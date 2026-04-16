from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch

from src.env.cityflow_signal_env import CityFlowSignalEnv
from src.models.model_rollout import rollout_model
from src.models.policy_head import MultiDiscretePolicy
from src.models.uncertainty_ensemble import DynamicsEnsemble
from src.models.value_head import GraphValueHead
from src.rl.ppo_multidiscrete import TrajectoryStep, update_ppo
from src.training.offline_pretrain import behavior_clone_policy, collect_offline_transitions
from src.training.train_world_model import build_ensemble, save_ensemble, train_world_model
from src.utils.seed import seed_everything



def _to_device_obs(obs: dict, device: torch.device):
    return (
        torch.tensor(obs['node_features'], dtype=torch.float32, device=device),
        torch.tensor(obs['edge_index'], dtype=torch.long, device=device),
        torch.tensor(obs['edge_attr'], dtype=torch.float32, device=device),
        torch.tensor(obs['action_mask'], dtype=torch.float32, device=device),
    )



def _synthetic_reward(prev_state: torch.Tensor, next_state: torch.Tensor, _actions: torch.Tensor) -> float:
    prev_queue = float(prev_state[:, 0].mean().item())
    next_queue = float(next_state[:, 0].mean().item())
    return prev_queue - next_queue



def train_mbrl_ppo(cfg: Dict[str, Any]) -> Dict[str, Any]:
    seed_everything(int(cfg.get('seed', 42)))
    device = torch.device(str(cfg.get('device', 'cpu')))
    env = CityFlowSignalEnv(cfg)
    obs, _ = env.reset()
    node_x, edge_index, edge_attr, action_mask = _to_device_obs(obs, device)
    input_dim = int(node_x.shape[1])
    edge_dim = int(edge_attr.shape[1]) if edge_attr is not None else None
    max_actions = int(action_mask.shape[1])
    model_cfg = cfg.get('model', {})
    ppo_cfg = cfg.get('ppo', {})
    train_cfg = cfg.get('training', {})

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

    offline_transitions = collect_offline_transitions(env, num_episodes=int(train_cfg.get('collect_episodes', 4)))
    ensemble: DynamicsEnsemble | None = None
    if float(train_cfg.get('model_ratio', 0.3)) > 0.0:
        ensemble = build_ensemble(cfg, state_dim=input_dim, max_actions=max_actions, edge_dim=edge_dim)
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
    for update_idx in range(total_updates):
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
            trajectory.append(
                TrajectoryStep(
                    state=obs['node_features'],
                    action_mask=obs['action_mask'],
                    action=action_np,
                    reward=float(reward),
                    done=float(terminated or truncated),
                    log_prob=float(policy_out.log_prob.detach().cpu().item()),
                    value=value,
                )
            )
            obs = next_obs
            terminated = bool(terminated or truncated)

        if ensemble is not None and float(train_cfg.get('model_ratio', 0.3)) > 0.0:
            node_x, edge_index, edge_attr, action_mask = _to_device_obs(obs, device)
            synth = rollout_model(
                ensemble=ensemble,
                state=node_x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                action_mask=action_mask,
                horizon=int(cfg.get('dynamics', {}).get('horizon', 5)),
                policy_fn=lambda state, edge, edge_features, mask: policy.sample(state, edge, edge_features, mask, deterministic=False).actions,
                reward_fn=_synthetic_reward,
                uncertainty_threshold=float(cfg.get('dynamics', {}).get('uncertainty_threshold', 0.25)),
                lambda_uncertainty=float(cfg.get('dynamics', {}).get('lambda_uncertainty', 0.1)),
            )
            for item in synth:
                state = item['state']
                action = item['action']
                if not isinstance(state, torch.Tensor) or not isinstance(action, torch.Tensor):
                    continue
                with torch.no_grad():
                    log_prob, _ = policy.evaluate_actions(state, edge_index, edge_attr, action_mask, action)
                    value = float(value_net(state, edge_index, edge_attr).item())
                trajectory.append(
                    TrajectoryStep(
                        state=state.detach().cpu().numpy(),
                        action_mask=action_mask.detach().cpu().numpy(),
                        action=action.detach().cpu().numpy(),
                        reward=float(item['reward']),
                        done=0.0,
                        log_prob=float(log_prob.detach().cpu().item()),
                        value=value,
                    )
                )
        stats = update_ppo(policy, value_net, edge_index, edge_attr, trajectory, ppo_cfg, policy_opt, value_opt, device=device)
        metrics.append({'update': float(update_idx), **stats, 'episode_return': float(sum(step.reward for step in trajectory))})

        if ensemble is not None and (update_idx + 1) % max(int(train_cfg.get('eval_every', 10)), 1) == 0:
            refresh_data = collect_offline_transitions(env, num_episodes=2)
            offline_transitions.extend(refresh_data)
            train_world_model(cfg, ensemble, edge_index, edge_attr, offline_transitions, device=device)

    output_dir = Path(cfg.get('output_dir', './outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'policy': policy.state_dict(), 'value': value_net.state_dict()}, output_dir / 'model_last.pt')
    if ensemble is not None:
        save_ensemble(ensemble, output_dir / 'dynamics_ensemble.pt')
    return {'metrics': metrics, 'output_dir': str(output_dir)}
