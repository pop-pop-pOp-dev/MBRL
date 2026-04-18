from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from src.data.offline_dataset import Transition
from src.env.cityflow_signal_env import CityFlowSignalEnv
from src.env.reward import compute_synthetic_reward_from_states
from src.models.model_rollout import rollout_model
from src.models.policy_head import MultiDiscretePolicy
from src.models.uncertainty_ensemble import DynamicsEnsemble
from src.models.value_head import GraphValueHead
from src.rl.ppo_multidiscrete import TrajectoryStep, update_ppo, update_value_with_mixed_batch
from src.training.decision_selector import select_action_with_world_model
from src.training.offline_pretrain import behavior_clone_policy, collect_offline_transitions
from src.training.replay_buffer import SplitReplayBuffer
from src.training.sample_selection import select_model_samples
from src.training.train_world_model import build_ensemble, train_world_model, save_ensemble
from src.utils.device import resolve_device
from src.utils.seed import seed_everything


def _to_device_obs(obs: dict, device: torch.device):
    return (
        torch.tensor(obs['node_features'], dtype=torch.float32, device=device),
        torch.tensor(obs['edge_index'], dtype=torch.long, device=device),
        torch.tensor(obs['edge_attr'], dtype=torch.float32, device=device),
        torch.tensor(obs['action_mask'], dtype=torch.float32, device=device),
    )


def _build_action_mask_from_state(state: torch.Tensor, action_space_n: int, observation_spec) -> torch.Tensor:
    mask = torch.ones(state.size(0), action_space_n, device=state.device)
    latest = observation_spec.latest_dynamic(state)
    remaining_green = latest[:, 4]
    hold_only = remaining_green > 0.01
    mask[hold_only, 1:] = 0.0
    return mask


def _make_synthetic_reward_fn(cfg: Dict[str, Any], observation_spec):
    reward_cfg = cfg.get('env', {}).get('reward', {})
    def _reward(prev_state: torch.Tensor, next_state: torch.Tensor, actions: torch.Tensor):
        return compute_synthetic_reward_from_states(prev_state, next_state, actions, observation_spec, reward_cfg)
    return _reward


def _collect_real_trajectory(
    env: CityFlowSignalEnv,
    policy: MultiDiscretePolicy,
    value_net: GraphValueHead,
    device: torch.device,
    ensemble: DynamicsEnsemble | None = None,
    decision_cfg: Dict[str, Any] | None = None,
    observation_spec=None,
    reward_fn=None,
) -> List[TrajectoryStep]:
    obs, _ = env.reset()
    trajectory: List[TrajectoryStep] = []
    terminated = False
    while not terminated:
        node_x, edge_index, edge_attr, action_mask = _to_device_obs(obs, device)
        decision_info = None
        with torch.no_grad():
            if ensemble is not None and decision_cfg and bool(decision_cfg.get('enabled', False)):
                selection = select_action_with_world_model(
                    policy=policy,
                    value_net=value_net,
                    ensemble=ensemble,
                    node_x=node_x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    action_mask=action_mask,
                    observation_spec=observation_spec,
                    reward_fn=reward_fn,
                    action_mask_fn=lambda state: _build_action_mask_from_state(state, int(action_mask.shape[1]), observation_spec),
                    cfg=decision_cfg,
                )
                action_tensor = selection.action
                log_prob = selection.log_prob
                value = selection.value
                decision_info = selection
            else:
                policy_out = policy.sample(node_x, edge_index, edge_attr, action_mask)
                action_tensor = policy_out.actions
                log_prob = policy_out.log_prob
                value = float(value_net(node_x, edge_index, edge_attr).item())
        action_np = action_tensor.detach().cpu().numpy()
        next_obs, reward, terminated, truncated, _info = env.step(action_np)
        next_node_x, _, _, _next_action_mask = _to_device_obs(next_obs, device)
        with torch.no_grad():
            next_value = float(value_net(next_node_x, edge_index, edge_attr).item()) if not (terminated or truncated) else 0.0
        step = TrajectoryStep(
            state=obs['node_features'],
            action_mask=obs['action_mask'],
            action=action_np,
            reward=float(reward),
            done=float(terminated or truncated),
            log_prob=float(log_prob.detach().cpu().item()),
            value=float(value),
            next_state=next_obs['node_features'],
            next_action_mask=next_obs['action_mask'],
            next_value=next_value,
            source='real',
            uncertainty=float(getattr(decision_info, 'selected_uncertainty', 0.0)),
        )
        trajectory.append(step)
        obs = next_obs
        terminated = bool(terminated or truncated)
    return trajectory


def _step_to_transition(step: TrajectoryStep) -> Transition:
    return Transition(
        state=step.state,
        action=step.action,
        reward=float(step.reward),
        next_state=step.next_state,
        done=float(step.done),
        action_mask=step.action_mask,
        next_action_mask=step.next_action_mask,
        uncertainty=float(step.uncertainty),
        source=step.source,
    )


def _queue_mean_from_transition(transition: Transition, observation_spec) -> float:
    metrics = observation_spec.metrics_from_state(transition.state)
    return float(np.mean(metrics.queue))


def _select_start_transitions(
    replay: SplitReplayBuffer,
    strategy: str,
    model_start_count: int,
    observation_spec,
) -> List[Transition]:
    strategy = str(strategy)
    if strategy == 'recent_real_episode':
        return replay.sample_real(model_start_count, strategy='latest')
    all_real = replay.real_buffer.all()
    if not all_real:
        return []
    if strategy == 'high_queue_states':
        ordered = sorted(all_real, key=lambda item: _queue_mean_from_transition(item, observation_spec), reverse=True)
        return ordered[:model_start_count]
    if strategy == 'coverage_balanced':
        ordered = sorted(all_real, key=lambda item: _queue_mean_from_transition(item, observation_spec))
        if len(ordered) <= model_start_count:
            return ordered
        idxs = np.linspace(0, len(ordered) - 1, num=model_start_count, dtype=int)
        return [ordered[int(idx)] for idx in idxs]
    return replay.sample_real(model_start_count, strategy='random')


def _select_model_transitions(
    transitions: Sequence[Transition],
    keep_count: int,
    cfg: Dict[str, Any],
) -> List[Transition]:
    training_cfg = cfg.get('training', {})
    metrics = list(training_cfg.get('priority_metrics', ['uncertainty', 'reward', 'queue_proxy']))
    return select_model_samples(
        transitions=transitions,
        keep_count=keep_count,
        alpha=float(training_cfg.get('priority_alpha', 1.0)),
        beta=float(training_cfg.get('priority_beta', 1.0)),
        metrics=metrics,
        coverage_enabled=bool(training_cfg.get('coverage_selection', True)),
        bins=dict(training_cfg.get('coverage_bins', {'queue': 4, 'phase': 4, 'uncertainty': 4})),
        min_fraction=float(training_cfg.get('coverage_min_fraction', 0.1)),
    )


def _select_guidance_steps(
    steps: Sequence[TrajectoryStep],
    batch_size: int,
    strategy: str,
) -> List[TrajectoryStep]:
    batch_size = min(max(int(batch_size), 0), len(steps))
    if batch_size <= 0:
        return []
    if strategy == 'random':
        indices = np.random.choice(len(steps), size=batch_size, replace=False)
        return [steps[int(idx)] for idx in indices]
    return list(steps[-batch_size:])


def _update_policy_with_model_guidance(
    policy: MultiDiscretePolicy,
    value_net: GraphValueHead,
    ensemble: DynamicsEnsemble | None,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    real_steps: Sequence[TrajectoryStep],
    observation_spec,
    reward_fn,
    cfg: Dict[str, Any],
    policy_opt: torch.optim.Optimizer,
    device: torch.device,
    max_actions: int,
) -> Dict[str, float]:
    guidance_cfg = dict(cfg.get('policy_guidance', {}))
    if ensemble is None or not bool(guidance_cfg.get('enabled', False)):
        return {
            'guidance_policy_loss': 0.0,
            'guidance_batch_size': 0.0,
            'guidance_mean_uncertainty': 0.0,
            'guidance_mean_gap': 0.0,
        }
    candidate_steps = _select_guidance_steps(
        real_steps,
        batch_size=int(guidance_cfg.get('batch_size', 8)),
        strategy=str(guidance_cfg.get('state_strategy', 'latest')),
    )
    if not candidate_steps:
        return {
            'guidance_policy_loss': 0.0,
            'guidance_batch_size': 0.0,
            'guidance_mean_uncertainty': 0.0,
            'guidance_mean_gap': 0.0,
        }

    planner_cfg = {
        'mode': str(guidance_cfg.get('mode', 'first_action_rerank')),
        'candidate_count': int(guidance_cfg.get('candidate_count', 4)),
        'plan_count': int(guidance_cfg.get('plan_count', int(guidance_cfg.get('candidate_count', 4)))),
        'include_greedy': bool(guidance_cfg.get('include_greedy', True)),
        'horizon': int(guidance_cfg.get('horizon', 2)),
        'discount': float(guidance_cfg.get('discount', cfg.get('ppo', {}).get('gamma', 0.99))),
        'uncertainty_coef': float(guidance_cfg.get('uncertainty_coef', cfg.get('dynamics', {}).get('lambda_uncertainty', 0.0))),
        'pessimism_coef': float(guidance_cfg.get('pessimism_coef', cfg.get('dynamics', {}).get('pessimism_coef', 0.0))),
        'future_action_mode': str(guidance_cfg.get('future_action_mode', 'greedy')),
    }
    max_uncertainty = float(guidance_cfg.get('max_uncertainty', cfg.get('dynamics', {}).get('uncertainty_threshold', 1.0)))
    min_score_gap = float(guidance_cfg.get('min_score_gap', 0.0))
    coef = float(guidance_cfg.get('coef', 0.1))
    grad_clip = float(cfg.get('ppo', {}).get('grad_clip', 1.0))

    selected_states: List[np.ndarray] = []
    selected_masks: List[np.ndarray] = []
    selected_actions: List[np.ndarray] = []
    uncertainties: List[float] = []
    gaps: List[float] = []

    for step in candidate_steps:
        node_x = torch.tensor(step.state, dtype=torch.float32, device=device)
        action_mask = torch.tensor(step.action_mask, dtype=torch.float32, device=device)
        selection = select_action_with_world_model(
            policy=policy,
            value_net=value_net,
            ensemble=ensemble,
            node_x=node_x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            action_mask=action_mask,
            observation_spec=observation_spec,
            reward_fn=reward_fn,
            action_mask_fn=lambda state: _build_action_mask_from_state(state, max_actions, observation_spec),
            cfg=planner_cfg,
        )
        if float(selection.selected_uncertainty) > max_uncertainty:
            continue
        if float(selection.score_gap) < min_score_gap:
            continue
        selected_states.append(np.asarray(step.state, dtype=np.float32))
        selected_masks.append(np.asarray(step.action_mask, dtype=np.float32))
        selected_actions.append(selection.action.detach().cpu().numpy().astype(np.int64, copy=False))
        uncertainties.append(float(selection.selected_uncertainty))
        gaps.append(float(selection.score_gap))

    if not selected_states:
        return {
            'guidance_policy_loss': 0.0,
            'guidance_batch_size': 0.0,
            'guidance_mean_uncertainty': 0.0,
            'guidance_mean_gap': 0.0,
        }

    state_batch = torch.tensor(np.stack(selected_states), dtype=torch.float32, device=device)
    mask_batch = torch.tensor(np.stack(selected_masks), dtype=torch.float32, device=device)
    action_batch = torch.tensor(np.stack(selected_actions), dtype=torch.long, device=device)
    log_prob, _entropy = policy.evaluate_actions(state_batch, edge_index, edge_attr, mask_batch, action_batch)
    action_count = max(int(action_batch.shape[-1]), 1)
    guidance_loss = -log_prob / float(len(selected_states) * action_count)
    total_loss = float(coef) * guidance_loss
    policy_opt.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    policy_opt.step()
    return {
        'guidance_policy_loss': float(total_loss.detach().cpu().item()),
        'guidance_batch_size': float(len(selected_states)),
        'guidance_mean_uncertainty': float(np.mean(uncertainties)) if uncertainties else 0.0,
        'guidance_mean_gap': float(np.mean(gaps)) if gaps else 0.0,
    }


def train_mbrl_ppo(cfg: Dict[str, Any]) -> Dict[str, Any]:
    seed_everything(int(cfg.get('seed', 42)))
    device = resolve_device(cfg)
    env = CityFlowSignalEnv(cfg)
    obs, info = env.reset()
    observation_spec = info['observation_spec']
    node_x, edge_index, edge_attr, action_mask = _to_device_obs(obs, device)
    input_dim = int(node_x.shape[1])
    edge_dim = int(edge_attr.shape[1]) if edge_attr is not None else None
    max_actions = int(action_mask.shape[1])
    model_cfg = cfg.get('model', {})
    ppo_cfg = cfg.get('ppo', {})
    train_cfg = cfg.get('training', {})
    dynamics_cfg = cfg.get('dynamics', {})
    decision_cfg = dict(cfg.get('decision', {}))
    model_aug_cfg = dict(cfg.get('model_augmentation', {}))

    policy = MultiDiscretePolicy(
        input_dim=input_dim,
        hidden_dim=int(model_cfg.get('hidden_dim', 128)),
        max_actions=max_actions,
        edge_dim=edge_dim,
        heads=int(model_cfg.get('gat_heads', 4)),
        num_layers=int(model_cfg.get('gat_layers', 3)),
        dropout=float(model_cfg.get('dropout', 0.1)),
    ).to(device)
    value_net = GraphValueHead(
        input_dim=input_dim,
        hidden_dim=int(model_cfg.get('hidden_dim', 128)),
        edge_dim=edge_dim,
        heads=int(model_cfg.get('gat_heads', 4)),
        num_layers=int(model_cfg.get('gat_layers', 3)),
        dropout=float(model_cfg.get('dropout', 0.1)),
    ).to(device)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=float(ppo_cfg.get('lr', 3e-4)))
    value_opt = torch.optim.Adam(value_net.parameters(), lr=float(ppo_cfg.get('value_lr', 3e-4)))

    replay = SplitReplayBuffer(
        real_capacity=int(train_cfg.get('max_real_samples_per_update', 4096)),
        model_capacity=int(train_cfg.get('max_model_samples_per_update', 4096)),
    )

    offline_transitions = collect_offline_transitions(env, num_episodes=int(train_cfg.get('collect_episodes', 4)), cfg=cfg)
    for transition in offline_transitions:
        replay.add_real(transition)

    ensemble: DynamicsEnsemble | None = None
    if float(train_cfg.get('model_ratio', 0.3)) > 0.0:
        ensemble = build_ensemble(cfg, state_dim=input_dim, max_actions=max_actions, edge_dim=edge_dim, observation_spec=observation_spec)
        train_world_model(cfg, ensemble, edge_index, edge_attr, offline_transitions, device=device)
        behavior_clone_policy(
            cfg,
            policy,
            edge_index,
            edge_attr,
            offline_transitions,
            epochs=int(train_cfg.get('bc_epochs', 5)),
            batch_size=int(train_cfg.get('batch_size', 64)),
            device=device,
        )

    total_updates = int(ppo_cfg.get('total_updates', 10))
    metrics: List[Dict[str, float]] = []
    reward_fn = _make_synthetic_reward_fn(cfg, observation_spec)
    model_selection_strategy = str(train_cfg.get('model_selection_strategy', 'prioritized'))
    decision_guidance_used = 0.0
    augmentation_enabled = bool(model_aug_cfg.get('enabled', True))
    start_time = datetime.now()
    print(
        f"[train] start time={start_time.isoformat()} total_updates={total_updates} "
        f"augmentation_enabled={augmentation_enabled} decision_enabled={bool(decision_cfg.get('enabled', False))}",
        flush=True,
    )
    for update_idx in range(total_updates):
        update_begin = datetime.now()
        real_steps = _collect_real_trajectory(
            env,
            policy,
            value_net,
            device,
            ensemble=ensemble,
            decision_cfg=decision_cfg,
            observation_spec=observation_spec,
            reward_fn=reward_fn,
        )
        for step in real_steps:
            replay.add_real(_step_to_transition(step))
        if ensemble is not None and bool(decision_cfg.get('enabled', False)):
            decision_guidance_used = 1.0

        generated_model_transitions: List[Transition] = []
        retention = 0.0
        start_queue_mean = 0.0
        start_strategy = str(train_cfg.get('model_start_state_strategy', 'real_buffer'))
        if augmentation_enabled and ensemble is not None and float(train_cfg.get('model_ratio', 0.3)) > 0.0 and len(replay.real_buffer) > 0:
            model_start_count = max(1, int(train_cfg.get('model_start_state_count', 4)))
            start_transitions = _select_start_transitions(replay, strategy=start_strategy, model_start_count=model_start_count, observation_spec=observation_spec)
            if start_transitions:
                start_queue_mean = float(np.mean([_queue_mean_from_transition(item, observation_spec) for item in start_transitions]))
            retained = 0
            attempted = 0
            for start in start_transitions:
                start_state = torch.tensor(start.state, dtype=torch.float32, device=device)
                start_mask = torch.tensor(start.action_mask, dtype=torch.float32, device=device)
                synth = rollout_model(
                    ensemble=ensemble,
                    state=start_state,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    action_mask=start_mask,
                    horizon=int(dynamics_cfg.get('horizon', 5)),
                    policy_fn=lambda state, edge, edge_features, mask: policy.sample(state, edge, edge_features, mask, deterministic=False).actions,
                    action_mask_fn=lambda state: _build_action_mask_from_state(state, max_actions, observation_spec),
                    reward_fn=reward_fn,
                    uncertainty_threshold=float(dynamics_cfg.get('uncertainty_threshold', 0.25)),
                    lambda_uncertainty=float(dynamics_cfg.get('lambda_uncertainty', 0.1)),
                    uncertainty_mode=str(dynamics_cfg.get('uncertainty_mode', 'threshold_only')),
                    pessimism_coef=float(dynamics_cfg.get('pessimism_coef', 0.0)),
                )
                attempted += int(dynamics_cfg.get('horizon', 5))
                retained += len(synth)
                for item in synth:
                    state = item['state']
                    next_state = item['next_state']
                    action = item['action']
                    action_mask_t = item['action_mask']
                    next_action_mask_t = item['next_action_mask']
                    if not isinstance(state, torch.Tensor) or not isinstance(next_state, torch.Tensor) or not isinstance(action, torch.Tensor):
                        continue
                    generated_model_transitions.append(
                        Transition(
                            state=state.detach().cpu().numpy(),
                            action=action.detach().cpu().numpy(),
                            reward=float(item['reward']),
                            next_state=next_state.detach().cpu().numpy(),
                            done=0.0,
                            action_mask=action_mask_t.detach().cpu().numpy(),
                            next_action_mask=next_action_mask_t.detach().cpu().numpy(),
                            uncertainty=float(item['uncertainty']),
                            source='model',
                        )
                    )
            retention = float(retained) / float(max(attempted, 1))
            keep_count = min(len(generated_model_transitions), int(train_cfg.get('max_model_samples_per_update', 4096)))
            if model_selection_strategy in {'prioritized', 'priority_coverage'}:
                selected_model_transitions = _select_model_transitions(generated_model_transitions, keep_count=keep_count, cfg=cfg)
            else:
                selected_model_transitions = generated_model_transitions[:keep_count]
            for transition in selected_model_transitions:
                replay.add_model(transition)

        # Keep PPO policy updates on-policy: only the latest real rollout carries old log-probs from the behavior policy.
        stats = update_ppo(policy, value_net, edge_index, edge_attr, real_steps, ppo_cfg, policy_opt, value_opt, device=device)

        guidance_stats = _update_policy_with_model_guidance(
            policy=policy,
            value_net=value_net,
            ensemble=ensemble,
            edge_index=edge_index,
            edge_attr=edge_attr,
            real_steps=real_steps,
            observation_spec=observation_spec,
            reward_fn=reward_fn,
            cfg=cfg,
            policy_opt=policy_opt,
            device=device,
            max_actions=max_actions,
        )

        batch_size = int(ppo_cfg.get('rollout_steps', 256))
        real_ratio = float(train_cfg.get('real_ratio', 0.7))
        model_ratio = float(train_cfg.get('model_ratio', 0.3)) if augmentation_enabled else 0.0
        sampled_real: List[Transition] = []
        sampled_model: List[Transition] = []
        if augmentation_enabled:
            sampled_real, sampled_model = replay.sample_mixed_by_ratio(
                total_count=batch_size,
                real_ratio=real_ratio,
                model_ratio=model_ratio,
                real_strategy=str(model_aug_cfg.get('real_strategy', 'random')),
                model_strategy=str(model_aug_cfg.get('model_strategy', 'random')),
            )
        mixed_transitions = list(sampled_real) + list(sampled_model)
        mixed_steps: List[TrajectoryStep] = []
        for transition in mixed_transitions:
            node_state = torch.tensor(transition.state, dtype=torch.float32, device=device)
            next_node_state = torch.tensor(transition.next_state, dtype=torch.float32, device=device)
            with torch.no_grad():
                value = float(value_net(node_state, edge_index, edge_attr).item())
                next_value = 0.0 if float(transition.done) > 0.5 else float(value_net(next_node_state, edge_index, edge_attr).item())
            mixed_steps.append(
                TrajectoryStep(
                    state=transition.state,
                    action_mask=transition.action_mask,
                    action=transition.action,
                    reward=float(transition.reward),
                    done=float(transition.done),
                    log_prob=0.0,
                    value=value,
                    next_state=transition.next_state,
                    next_action_mask=transition.next_action_mask,
                    next_value=next_value,
                    source=transition.source,
                    uncertainty=float(transition.uncertainty),
                )
            )
        aux_stats = update_value_with_mixed_batch(
            value_net=value_net,
            edge_index=edge_index,
            edge_attr=edge_attr,
            transitions=mixed_steps,
            gamma=float(ppo_cfg.get('gamma', 0.99)),
            optimizer=value_opt,
            device=device,
        )
        metrics.append(
            {
                'update': float(update_idx),
                **stats,
                **guidance_stats,
                **aux_stats,
                'episode_return': float(sum(step.reward for step in real_steps)),
                'buffer_real_size': float(len(replay.real_buffer)),
                'buffer_model_size': float(len(replay.model_buffer)),
                'sampled_real_count': float(len(sampled_real)),
                'sampled_model_count': float(len(sampled_model)),
                'configured_real_ratio': real_ratio,
                'configured_model_ratio': model_ratio,
                'actual_model_ratio': float(len(sampled_model)) / float(max(len(sampled_real) + len(sampled_model), 1)),
                'model_rollout_keep_ratio': retention,
                'model_start_state_strategy': start_strategy,
                'selected_start_state_count': float(train_cfg.get('model_start_state_count', 4)),
                'start_state_queue_mean': start_queue_mean,
                'model_selection_strategy': 1.0 if model_selection_strategy in {'prioritized', 'priority_coverage'} else 0.0,
                'decision_guidance_enabled': decision_guidance_used,
                'model_augmentation_enabled': 1.0 if augmentation_enabled else 0.0,
            }
        )
        latest = metrics[-1]
        print(
            "[train] "
            f"update={update_idx + 1}/{total_updates} "
            f"return={latest['episode_return']:.4f} "
            f"policy_loss={latest.get('policy_loss', 0.0):.6f} "
            f"guidance_loss={latest.get('guidance_policy_loss', 0.0):.6f} "
            f"value_loss={latest.get('value_loss', 0.0):.6f} "
            f"aux_value_loss={latest.get('aux_value_loss', 0.0):.6f} "
            f"real_buf={int(latest['buffer_real_size'])} "
            f"model_buf={int(latest['buffer_model_size'])} "
            f"sampled_real={int(latest['sampled_real_count'])} "
            f"sampled_model={int(latest['sampled_model_count'])} "
            f"model_keep={latest['model_rollout_keep_ratio']:.4f} "
            f"elapsed_sec={(datetime.now() - update_begin).total_seconds():.2f}",
            flush=True,
        )

        if ensemble is not None and (update_idx + 1) % max(int(train_cfg.get('eval_every', 10)), 1) == 0:
            refresh_data = collect_offline_transitions(env, num_episodes=2, cfg=cfg)
            offline_transitions.extend(refresh_data)
            train_world_model(cfg, ensemble, edge_index, edge_attr, offline_transitions, device=device)
            cleared_model_count = len(replay.model_buffer)
            replay.clear_model()
            print(f"[train] world_model_refresh cleared_model_buffer={cleared_model_count}", flush=True)

    output_dir = Path(cfg.get('output_dir', './outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'policy': policy.state_dict(), 'value': value_net.state_dict()}, output_dir / 'model_last.pt')
    if ensemble is not None:
        save_ensemble(ensemble, output_dir / 'dynamics_ensemble.pt')
    total_elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[train] finished output_dir={output_dir} elapsed_sec={total_elapsed:.2f}", flush=True)
    return {'metrics': metrics, 'output_dir': str(output_dir)}
