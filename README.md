# CityFlow MBRL+ Signal Control

A CityFlow-based traffic signal control project centered on a hardened uncertainty-aware traffic MBPO / Dyna-style MBRL pipeline. The repository focuses on data-efficient signal control with graph dynamics modeling, uncertainty-aware imagined rollout, replay-based mixed optimization, and lightweight decision-time world-model guidance.

This project is intentionally **not** positioned as a full MPC / CEM / shooting-planner system. Instead, it explores a stronger route-1 MBRL design in which the world model first improves training data quality and then progressively starts to influence action selection through short-horizon candidate-plan reranking.

## Project Status

- Current stage: hardened route-1 MBRL prototype with decision-time sequence shooting
- Simulation backend: `CityFlow`
- Default benchmark: `RoadnetSZ Fuhua`
- Control form: centralized training with per-intersection multi-discrete phase actions
- Policy backbone: `PPO`
- World model: `GAT + action fusion + GRU` with multi-step consistent training
- Uncertainty: ensemble variance, ranked/pessimistic imagined rollout, and uncertainty-aware sample filtering
- Decision-time control: short-horizon candidate plan reranking with first-action execution

## Why This Repository Exists

The goal of this repository is to bridge the gap between:

- pure model-free traffic signal RL, which is often interaction-hungry
- and heavy planner-driven world-model control, which is stronger but much more expensive to build and stabilize

The result is a pragmatic middle route:

- a graph world model that predicts aggregated traffic state evolution
- uncertainty-aware imagination that does not blindly trust model rollout
- replay-based real/model sample mixing rather than naive synthetic append
- lightweight decision-time control that allows the model to directly influence actions without requiring a full MPC stack

## Statistical Scope

This repository now keeps a cleaner algorithmic boundary between policy optimization and model augmentation:

- PPO policy updates are applied only to the latest real on-policy rollout
- imagined and replayed samples are used for model augmentation and auxiliary value updates
- decision-time world-model guidance is controlled separately from training-time model augmentation

This separation is intentional so that ablations such as:

- `decision on / augmentation off`
- `decision off / augmentation on`
- `decision on / augmentation on`

remain interpretable.

## Core Design

### 1. State and Action

- State is represented at the intersection level
- Each node aggregates:
  - queue length
  - vehicle count
  - average speed
  - current phase
  - remaining green time
- Dynamic features are temporally stacked
- Static graph features are concatenated once, rather than repeated across the stack
- Action is multi-discrete:
  - each intersection chooses `hold` or a legal phase change

### 2. Graph World Model

- Graph encoder: `GAT`
- Action fusion: learnable action embedding per intersection
- Temporal model: `GRU`
- Training objective:
  - one-step reconstruction
  - multi-step rollout consistency
  - traffic prior penalties

The world model predicts aggregated next-state evolution rather than raw vehicle trajectories. This keeps the model lightweight enough for route-1 MBRL while still making it useful for imagination and short-horizon control.

### 3. Uncertainty-Aware Imagination

- Dynamics are modeled as an ensemble
- Imagined rollout uses:
  - variance threshold gating
  - uncertainty-aware reward penalty
  - pessimistic state estimates
  - ranked trajectory retention
- Imagined samples are not used naively:
  - they can be prioritized
  - they can be coverage-rebalanced
  - they can be filtered by uncertainty

### 4. Decision-Time World-Model Control

This repository now goes beyond “model for better training samples only”.

At decision time:

- the policy proposes candidate actions or candidate action sequences
- the world model rolls them forward for a short horizon
- the value head provides bootstrapped terminal evaluation
- the best candidate sequence is selected
- only the first action is executed in the real environment

This is a lightweight sequence-shooting style controller, not a full planner. It is meant to be stronger than first-action reranking while remaining much cheaper than full MPC.

## Training Pipeline

The default training pipeline is:

1. Collect offline data from heuristic and perturbed controllers
2. Pretrain the world model with multi-step consistent loss
3. Warm-start the policy by behavior cloning
4. Run online MBRL training
5. Mix real and imagined samples through replay-based ratio control
6. Periodically refresh the world model with new online data

In practice, the project combines:

- real environment rollout
- synthetic rollout from selected real-buffer start states
- prioritized model sample selection
- coverage-aware replay usage
- PPO updates on mixed batches

## Decision-Time Control Pipeline

The stronger decision-time branch currently works as follows:

1. Build several candidate first actions or candidate action sequences
2. Simulate them with the world model
3. Apply pessimistic rollout and uncertainty penalties
4. Add value bootstrap at the sequence tail
5. Rank candidate plans by short-horizon imagined return
6. Execute only the first action of the best plan

This gives the model a direct role in control while preserving the PPO training backbone.

## Repository Layout

- `configs/`: experiment and ablation configs
- `src/data/`: roadnet, phase, and offline dataset parsing
- `src/env/`: CityFlow env, observation, reward, and phase control
- `src/models/`: encoder, policy/value heads, dynamics, uncertainty
- `src/rl/`: multi-discrete PPO
- `src/training/`: offline pretrain, world model, replay, sample selection, decision selector, and MBRL training
- `src/eval/`: evaluation, generalization, and robustness
- `scripts/`: runnable entrypoints
- `tests/`: unit tests

## Important Modules

- `src/env/cityflow_signal_env.py`
  - wraps `CityFlow`
  - exposes graph observation, mask, reward, and phase constraints
- `src/models/graph_dynamics.py`
  - graph dynamics model used for imagined rollout
- `src/models/model_rollout.py`
  - uncertainty-aware rollout logic
- `src/training/replay_buffer.py`
  - split real/model replay buffers
- `src/training/sample_selection.py`
  - prioritized and coverage-aware imagined sample selection
- `src/training/decision_selector.py`
  - decision-time candidate plan scoring and selection
- `src/training/train_mbrl_ppo.py`
  - main route-1 MBRL training loop

## Default Configuration Themes

The default config in `configs/base.yaml` already includes several important families of controls:

- `dynamics`
  - rollout horizon
  - multi-step loss weight
  - teacher forcing ratio
  - rollout schedule
  - uncertainty mode
  - pessimism coefficient
- `training`
  - real/model mixing ratio
  - start-state strategy
  - sample priority weights
  - coverage bins
  - offline data mixture
- `model_augmentation`
  - enable or disable imagined training augmentation independently of decision-time guidance
  - choose replay sampling strategies for real/model auxiliary batches
- `decision`
  - enable decision-time model guidance
  - choose decision mode
  - candidate count / plan count
  - short-horizon discounting
  - uncertainty and pessimism penalties

## Suggested Workflow

```bash
python3 scripts/collect_offline.py --config configs/experiments/fuhua_mbrl_ppo.yaml
python3 scripts/train_world_model.py --config configs/experiments/fuhua_mbrl_ppo.yaml
python3 scripts/train_real_only.py --config configs/ablations/real_only.yaml
python3 scripts/train_mbrl.py --config configs/experiments/fuhua_mbrl_ppo.yaml
python3 scripts/evaluate.py --config configs/experiments/fuhua_mbrl_ppo.yaml
```

## Route-1 Hardening Highlights

- Replay-based mixed sampling instead of prefix slicing
- Prioritized and coverage-aware imagined sample selection
- Config-driven model start-state strategy and rollout schedule
- Threshold-ranked and pessimistic uncertainty usage for imagined data retention
- Decision-aware action-sequence shooting with short model rollout and first-action execution
- Still route-1 MBPO/Dyna style, not planner-based control

## Recommended Ablations

- `configs/ablations/real_only.yaml`
- `configs/ablations/no_model.yaml`
- `configs/ablations/no_uncertainty.yaml`
- `configs/ablations/one_step_world_model.yaml`
- `configs/ablations/no_prior_constraints.yaml`
- `configs/ablations/prefix_mixing.yaml`
- `configs/ablations/no_priority_selection.yaml`
- `configs/ablations/no_coverage_selection.yaml`

Recommended comparisons for this repository are not just “with model vs without model”, but also:

- one-step world model vs multi-step world model
- naive imagined usage vs priority/coverage-aware imagined usage
- threshold-only uncertainty vs ranked/pessimistic uncertainty
- training-only model usage vs decision-time world-model guidance

## Current Strengths

- Stronger route-1 completeness than a naive MBPO baseline
- Clear separation of real and imagined data in training
- Explicit uncertainty-aware rollout design
- Model sample selection with both priority and coverage terms
- World model already affects action selection rather than only replay augmentation

## Current Boundaries

This repository is still not the final answer to traffic world-model control. Key boundaries remain:

- coverage signatures are still lightweight rather than topology-rich
- online world-model refresh is incremental rather than fully adaptive
- decision-time control is sequence shooting, not full planner search
- no beam search, no tree expansion, no MPC fallback yet

These boundaries are intentional for the current research scope. The repository is designed to be a strong route-1 MBRL prototype rather than an all-in route-2/route-3 planning system.

## Next Logical Extensions

If the project continues beyond the current version, the most natural next upgrades are:

- richer coverage signatures with upstream/downstream congestion structure
- stronger online model refresh policies
- beam-search style candidate plan expansion
- lightweight MPC fallback for uncertain states
- explicit ablations showing the gain of each `MBRL+` component
