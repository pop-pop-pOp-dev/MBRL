# CityFlow MBRL+ Signal Control

A CityFlow-based traffic signal control project using a weak world model, strong policy learning, and uncertainty-gated imagination.

## Defaults

- Backend: `CityFlow`
- Benchmark: `RoadnetSZ Fuhua`
- Control: centralized training with per-intersection multi-discrete phase actions
- Policy: `PPO`
- World model: `GAT + action fusion + GRU`
- Uncertainty: `ensemble variance gate`

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

## Install

```bash
pip install -r requirements.txt
```

## Suggested workflow

```bash
python3 scripts/collect_offline.py --config configs/experiments/fuhua_mbrl_ppo.yaml
python3 scripts/train_world_model.py --config configs/experiments/fuhua_mbrl_ppo.yaml
python3 scripts/train_real_only.py --config configs/ablations/real_only.yaml
python3 scripts/train_mbrl.py --config configs/experiments/fuhua_mbrl_ppo.yaml
python3 scripts/evaluate.py --config configs/experiments/fuhua_mbrl_ppo.yaml
```

## Notes

- Env scripts raise a clear error when `cityflow` is missing.
- The project supports both built-in `RoadnetSZ Fuhua` assets and custom `roadnet.json + flow.json + config.json` inputs.
