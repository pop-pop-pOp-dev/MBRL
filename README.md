# CityFlow MBRL+ Signal Control

A CityFlow-based traffic signal control project using a hardened uncertainty-aware traffic MBPO / Dyna-style MBRL stack rather than heavy planner-based control.

## Defaults

- Backend: `CityFlow`
- Benchmark: `RoadnetSZ Fuhua`
- Control: centralized training with per-intersection multi-discrete phase actions
- Policy: `PPO`
- World model: `GAT + action fusion + GRU` with multi-step consistent training
- Uncertainty: ensemble variance gate plus ranked/pessimistic imagined sample selection
- Positioning: hardened route-1 traffic MBPO, not MPC or shooting-based planning

## Layout

- `configs/`: experiment and ablation configs
- `src/data/`: roadnet, phase, and offline dataset parsing
- `src/env/`: CityFlow env, observation, reward, and phase control
- `src/models/`: encoder, policy/value heads, dynamics, uncertainty
- `src/rl/`: multi-discrete PPO
- `src/training/`: offline pretrain, world model, replay, sample selection, and MBRL training
- `src/eval/`: evaluation, generalization, and robustness
- `scripts/`: runnable entrypoints
- `tests/`: unit tests

## Suggested workflow

```bash
python3 scripts/collect_offline.py --config configs/experiments/fuhua_mbrl_ppo.yaml
python3 scripts/train_world_model.py --config configs/experiments/fuhua_mbrl_ppo.yaml
python3 scripts/train_real_only.py --config configs/ablations/real_only.yaml
python3 scripts/train_mbrl.py --config configs/experiments/fuhua_mbrl_ppo.yaml
python3 scripts/evaluate.py --config configs/experiments/fuhua_mbrl_ppo.yaml
```

## Route-1 hardening highlights

- Replay-based mixed sampling instead of prefix slicing
- Prioritized and coverage-aware imagined sample selection
- Config-driven model start-state strategy and rollout schedule
- Threshold-ranked and pessimistic uncertainty usage for imagined data retention
- Still route-1 MBPO/Dyna style, not planner-based control

## Recommended ablations

- `configs/ablations/real_only.yaml`
- `configs/ablations/no_model.yaml`
- `configs/ablations/no_uncertainty.yaml`
- `configs/ablations/one_step_world_model.yaml`
- `configs/ablations/no_prior_constraints.yaml`
- `configs/ablations/prefix_mixing.yaml`
- `configs/ablations/no_priority_selection.yaml`
- `configs/ablations/no_coverage_selection.yaml`
