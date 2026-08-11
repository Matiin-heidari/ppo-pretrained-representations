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
- Compute pilot run on all encoders to measure real throughput before committing to the full
  experiment grid. Core comparison is CNN vs. frozen ResNet18 at full budget (100k/250k/500k);
  fine-tuned ResNet18 is capped to the 100k budget only, and frozen DINOv2 runs as a slower stretch
  comparison — both due to a CUDA correctness issue on this hardware, see
  [Known issues](#known-issues).

- Dormant-neuron-ratio tracking for the encoder and the PPO heads, measured on a fixed reference
  batch of observations shared by every run. On by default; logged to TensorBoard under
  `dormant/` and to `results/dormant/<run_name>.jsonl` for later aggregation.
- Generalization train/test split (`src/envs/generalization.py`) and evaluation harness
  (`src/eval_generalization.py`) for H3 — see [Generalization](#generalization) below.

Not yet implemented / not yet run:
- No full-budget run has been launched yet — everything so far is smoke-tested at ≤8192 steps.
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

## Generalization

Training draws episodes from the lower half of MiniWorld's appearance-randomization ranges
(sky/light color, light position, light ambient, object color bias — see
`src/envs/generalization.py` for exactly which parameters and why); the upper half is held out for
evaluation. Toggle with the `generalization: {enabled: ...}` config block (on by default).

Evaluate a checkpoint on both halves:

```
python -m src.eval_generalization configs/cnn_scratch_oneroom_seed0.yaml --checkpoint-step 100000
```

Reports mean ± 95% CI return on the train-range and held-out test-range distributions, and the gap
between them, to `results/generalization/<run_name>_<step>.json`.

## Repo structure

```
src/
  encoders/                # CNN, ResNet18, DINOv2 feature extractors (shared BaseFeaturesExtractor interface)
  envs/
    make_env.py             # vectorized MiniWorld env factory
    generalization.py       # train/test appearance-randomization split (H3)
  train.py                 # PPO training entrypoint
  eval_generalization.py   # evaluates a checkpoint on the held-out split
  dormant_ratio.py         # dormant-neuron tracking (hooks, fixed reference batch, SB3 callback)
  smoke_test_render.py     # standalone MiniWorld rendering check
configs/                  # one YAML per run; files prefixed `_smoke_` are throwaway benchmark/debug configs
tests/                    # unit tests (`python -m tests.test_dormant_ratio`)
results/                   # checkpoints + TensorBoard logs + dormant-ratio/generalization JSON (gitignored)
```

## Known issues

**Frozen DINOv2 and fine-tuned ResNet18 both require synchronous CUDA execution.** On our dev GPU,
frozen DINOv2 occasionally produced NaN outputs during live PPO training, and fine-tuned ResNet18
separately crashed with `cudaErrorIllegalInstruction` — neither reproduces on offline replay of
identical input data, and both trace to an async execution race between a large pretrained
backbone's kernels and the trainable heads' kernels on the default CUDA stream. `src/train.py` sets
`CUDA_LAUNCH_BLOCKING=1` automatically whenever `encoder: dinov2` or `encoder_kwargs.trainable:
true`, which reliably fixes both. The throughput cost differs a lot by case: roughly 4x for DINOv2,
but roughly 35–40x for fine-tuned ResNet18 (~3–4 steps/sec vs. ~130 frozen) — expensive enough that
the fine-tuning ablation is budget-capped rather than run at full scale. Plain CNN and frozen
ResNet18/DINOv2 are unaffected and stay fast.

**A suspiciously slow benchmark may just be GPU contention, not your code.** We initially
misdiagnosed the ResNet18 fine-tuning slowdown above as another process hogging the GPU, because
`nvidia-smi` showed ~7.9/8GB VRAM used and 100% utilization — but that turned out to be true even
with no training script running at all. If a run seems unexpectedly slow, check `nvidia-smi` with
nothing of yours running before concluding it's your code.
