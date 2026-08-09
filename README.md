# Evaluating the Generalization of Pretrained Visual Representations for Sample-Efficient RL

**Research question:** To what extent do pretrained visual representations improve sample
efficiency and generalization in PPO-based reinforcement learning under limited interaction
budgets?

**Hypotheses:**
- H1: Pretrained visual representations improve sample efficiency compared to learning visual
  features from scratch.
- H2: The benefit of pretrained representations is largest under limited interaction budgets.
- H3: Different pretrained representations exhibit different trade-offs between sample efficiency,
  final performance, and generalization.

## Status

Implemented and validated:
- PPO training pipeline (Stable-Baselines3) on `MiniWorld-OneRoom-v0`, config-driven via YAML,
  with checkpointing at fixed interaction budgets (100k/250k/500k steps).
- Three pluggable feature extractors: CNN-from-scratch (baseline), frozen/fine-tunable ResNet18,
  frozen/fine-tunable DINOv2 (ViT-S/14, `[CLS]` token).
- Compute pilot run on all three encoders to measure real throughput before committing to the full
  experiment grid. Core comparison is CNN vs. ResNet18 (frozen and fine-tuned); DINOv2 (frozen) is
  run as a slower secondary comparison — see [Known issues](#known-issues).

- Dormant-neuron-ratio tracking for the encoder and the PPO heads, measured on a fixed reference
  batch of observations shared by every run. On by default; logged to TensorBoard under
  `dormant/` and to `results/dormant/<run_name>.jsonl` for later aggregation.

Not yet implemented:
- Generalization train/test split and evaluation harness.
- The full multi-seed run grid and result aggregation/plots.

## Setup

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Adjust the `cu130` tag to match your GPU driver if needed (see the comment at the top of
`requirements.txt`).

## Usage

Train a single run from a config:

```
python -m src.train configs/cnn_scratch_oneroom_seed0.yaml
```

Each config specifies the environment, encoder, PPO hyperparameters, and checkpoint schedule, e.g.:

```yaml
run_name: cnn_scratch_oneroom_seed0
env_id: MiniWorld-OneRoom-v0
seed: 0
n_envs: 4
encoder: cnn                 # cnn | resnet18 | dinov2
encoder_kwargs:
  features_dim: 512          # cnn only
total_timesteps: 500000
checkpoint_steps: [100000, 250000, 500000]
```

For `resnet18`/`dinov2`, `encoder_kwargs` takes `trainable: false|true` (frozen vs. fine-tuned)
instead of `features_dim`, which is fixed by the pretrained architecture.

Checkpoints and TensorBoard logs are written under `results/`.

View tensorboard logs with `tensorboard --logdir results/tensorboard`

## Repo structure

```
src/
  encoders/          # CNN, ResNet18, DINOv2 feature extractors (shared BaseFeaturesExtractor interface)
  envs/               # vectorized MiniWorld env factory
  train.py             # PPO training entrypoint
  dormant_ratio.py     # dormant-neuron tracking (hooks, fixed reference batch, SB3 callback)
  smoke_test_render.py # standalone MiniWorld rendering check
configs/              # one YAML per run; files prefixed `_smoke_` are throwaway benchmark/debug configs
tests/                # unit tests (`python -m tests.test_dormant_ratio`)
results/               # checkpoints + TensorBoard logs + dormant-ratio JSONL (gitignored)
```

## Known issues

**Frozen DINOv2 requires synchronous CUDA execution.** On our dev GPU, frozen DINOv2 occasionally
produced NaN outputs during live PPO training (never on offline replay of identical input data),
traced to an async execution race between the frozen encoder's forward kernels and the trainable
heads' backward kernels. `src/train.py` sets `CUDA_LAUNCH_BLOCKING=1` automatically when
`encoder: dinov2`, which reliably fixes it at a real throughput cost (roughly 4x slower). CNN and
ResNet18 runs are unaffected.
