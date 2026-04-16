# CityFlow MBRL+ Signal Control

A CityFlow-based traffic signal control project using uncertainty-aware traffic MBPO / Dyna-style MBRL rather than heavy planner-based control.

## Defaults

- Backend: `CityFlow`
- Benchmark: `RoadnetSZ Fuhua`
- Control: centralized training with per-intersection multi-discrete phase actions
- Policy: `PPO`
- World model: `GAT + action fusion + GRU` with multi-step consistent training
- Uncertainty: ensemble variance gate
- Positioning: uncertainty-aware traffic MBPO, not MPC or shooting-based planning

## Layout

- `configs/`: experiment and ablation configs
- `src/data/`: roadnet, phase, and offline dataset parsing
- `src/env/`: CityFlow env, observation, reward, and phase control
- `src/models/`: encoder, policy/value heads, dynamics, uncertainty
- `src/rl/`: multi-discrete PPO
- `src/training/`: offline pretrain, world model, and MBRL training
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

## Key upgrades

- Unified synthetic and real reward structure
- Multi-step consistent world model training
- Ratio-controlled real/model mixing for PPO
- Traffic prior penalties for physical feasibility
- Broader offline coverage with heuristic and random perturbation mixture

## Recommended ablations

- `configs/ablations/real_only.yaml`
- `configs/ablations/no_model.yaml`
- `configs/ablations/no_uncertainty.yaml`
- `configs/ablations/one_step_world_model.yaml`
- `configs/ablations/no_prior_constraints.yaml`
