from __future__ import annotations

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
from src.rl.ppo_multidiscrete import TrajectoryStep, update_ppo
from src.training.offline_pretrain import behavior_clone_policy, collect_offline_transitions
from src.training.replay_buffer import SplitReplayBuffer
from src.training.sample_selection import select_model_samples
from src.training.train_world_model import build_ensemble, train_world_model, save_ensemble
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



def _collect_real_trajectory(env: CityFlowSignalEnv, policy: MultiDiscretePolicy, value_net: GraphValueHead, device: torch.device) -> List[TrajectoryStep]:
    obs, _ = env.reset()
    trajectory: List[TrajectoryStep] = []
    terminated = False
    while not terminated:
        node_x, edge_index, edge_attr, action_mask = _to_device_obs(obs, device)
        with torch.no_grad():
            policy_out = policy.sample(node_x, edge_index, edge_attr, action_mask)
            value = float(value_net(node_x, edge_index, edge_attr).item())
        action_np = policy_out.actions.detach().cpu().numpy()
        next_obs, reward, terminated, truncated, _info = env.step(action_np)
        next_node_x, _, _, next_action_mask = _to_device_obs(next_obs, device)
        with torch.no_grad():
            next_value = float(value_net(next_node_x, edge_index, edge_attr).item()) if not (terminated or truncated) else 0.0
        trajectory.append(
            TrajectoryStep(
                state=obs['node_features'],
                action_mask=obs['action_mask'],
                action=action_np,
                reward=float(reward),
                done=float(terminated or truncated),
                log_prob=float(policy_out.log_prob.detach().cpu().item()),
                value=value,
                next_state=next_obs['node_features'],
                next_action_mask=next_obs['action_mask'],
                next_value=next_value,
                source='real',
                uncertainty=0.0,
            )
        )
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



def train_mbrl_ppo(cfg: Dict[str, Any]) -> Dict[str, Any]:
    seed_everything(int(cfg.get('seed', 42)))
    device = torch.device(str(cfg.get('device', 'cpu')))
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
    for update_idx in range(total_updates):
        real_steps = _collect_real_trajectory(env, policy, value_net, device)
        for step in real_steps:
            replay.add_real(_step_to_transition(step))

        generated_model_transitions: List[Transition] = []
        retention = 0.0
        start_queue_mean = 0.0
        start_strategy = str(train_cfg.get('model_start_state_strategy', 'real_buffer'))
        if ensemble is not None and float(train_cfg.get('model_ratio', 0.3)) > 0.0 and len(replay.real_buffer) > 0:
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

        batch_size = int(ppo_cfg.get('rollout_steps', 256))
        real_ratio = float(train_cfg.get('real_ratio', 0.7))
        model_ratio = float(train_cfg.get('model_ratio', 0.3))
        sampled_real, sampled_model = replay.sample_mixed_by_ratio(
            total_count=batch_size,
            real_ratio=real_ratio,
            model_ratio=model_ratio,
            real_strategy='random',
            model_strategy='random',
        )
        mixed_transitions = list(sampled_real) + list(sampled_model)
        mixed_steps: List[TrajectoryStep] = []
        for transition in mixed_transitions:
            node_state = torch.tensor(transition.state, dtype=torch.float32, device=device)
            next_node_state = torch.tensor(transition.next_state, dtype=torch.float32, device=device)
            action_mask_t = torch.tensor(transition.action_mask, dtype=torch.float32, device=device)
            action_t = torch.tensor(transition.action, dtype=torch.long, device=device)
            with torch.no_grad():
                log_prob, _ = policy.evaluate_actions(node_state, edge_index, edge_attr, action_mask_t, action_t)
                value = float(value_net(node_state, edge_index, edge_attr).item())
                next_value = 0.0 if float(transition.done) > 0.5 else float(value_net(next_node_state, edge_index, edge_attr).item())
            mixed_steps.append(
                TrajectoryStep(
                    state=transition.state,
                    action_mask=transition.action_mask,
                    action=transition.action,
                    reward=float(transition.reward),
                    done=float(transition.done),
                    log_prob=float(log_prob.detach().cpu().item()),
                    value=value,
                    next_state=transition.next_state,
                    next_action_mask=transition.next_action_mask,
                    next_value=next_value,
                    source=transition.source,
                    uncertainty=float(transition.uncertainty),
                )
            )
        stats = update_ppo(policy, value_net, edge_index, edge_attr, mixed_steps, ppo_cfg, policy_opt, value_opt, device=device)
        metrics.append(
            {
                'update': float(update_idx),
                **stats,
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
            }
        )

        if ensemble is not None and (update_idx + 1) % max(int(train_cfg.get('eval_every', 10)), 1) == 0:
            refresh_data = collect_offline_transitions(env, num_episodes=2, cfg=cfg)
            offline_transitions.extend(refresh_data)
            train_world_model(cfg, ensemble, edge_index, edge_attr, offline_transitions, device=device)

    output_dir = Path(cfg.get('output_dir', './outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'policy': policy.state_dict(), 'value': value_net.state_dict()}, output_dir / 'model_last.pt')
    if ensemble is not None:
        save_ensemble(ensemble, output_dir / 'dynamics_ensemble.pt')
    return {'metrics': metrics, 'output_dir': str(output_dir)}
