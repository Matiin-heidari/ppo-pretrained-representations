"""Dormant neuron tracking for the visual encoder and the PPO policy/value heads.

Implements the dormant-neuron score of Sokar et al.(ICML 2023) -- the metric that DRM, used by the
Williams & Polydoros work this project is built upon.

Normalizing per layer makes the score scale-free, which is what lets us compare the Nature-CNN
against ResNet18 against ViT-S/14 on one axis.

Measurements are taken on a *fixed* reference batch of observations that is identical across
every run, so that a change in the ratio means the network changed rather than the input
distribution changing.
"""
import json
import os
from collections import defaultdict

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

DEFAULT_TAUS = (0.0, 0.025, 0.1)

# The score is defined over post-nonlinearity activations. ReLU/GELU cover our three encoders
# (Nature-CNN and ResNet18 use ReLU, DINOv2's ViT MLP blocks use GELU); Tanh is included because
# SB3's default policy/value heads use it.
ACTIVATION_TYPES = (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SiLU, nn.GELU, nn.Tanh)

ENCODER_GROUP = "encoder"
HEAD_GROUP = "head"


def _group_for(layer_name: str) -> str:
    """Encoder vs. PPO heads. SB3 names the extractor `features_extractor` (and the aliases
    `pi_features_extractor` / `vf_features_extractor` when it is shared)."""
    return ENCODER_GROUP if "features_extractor" in layer_name else HEAD_GROUP


class DormantRatioTracker:
    """Accumulates per-unit mean absolute activations via forward hooks.

    Usage: `attach()`, run one or more forward passes calling `new_forward()` before each,
    then `compute()` / `summarize()`, then `detach()`. `reset()` clears accumulated stats
    between measurements.
    """

    def __init__(self, module: nn.Module, activation_types=ACTIVATION_TYPES):
        # named_modules() de-duplicates shared submodules by default, so a features extractor
        # exposed under several attribute names is only hooked once.
        self.activation_modules = [
            (name, m) for name, m in module.named_modules() if isinstance(m, activation_types)
        ]
        self._handles = []
        self._sum_abs = {}  # layer key -> Tensor[n_units], summed |activation|
        self._counts = {}  # layer key -> int, elements reduced into each unit
        self._call_index = defaultdict(int)

    # -- hook plumbing -------------------------------------------------------------------

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if not isinstance(output, th.Tensor) or output.dim() < 2:
                return
            # A module can be invoked more than once per forward pass (torchvision's BasicBlock
            # reuses a single nn.ReLU twice), so key on the call index too rather than merging
            # two distinct activation sites into one "layer".
            key = f"{name}#{self._call_index[name]}"
            self._call_index[name] += 1

            out = output.detach()
            if out.dim() == 4:
                # (N, C, H, W): a unit is a channel, so reduce over batch and space.
                reduce_dims = (0, 2, 3)
            else:
                # (N, F) or (N, tokens, F): the unit axis is last.
                reduce_dims = tuple(range(out.dim() - 1))

            sums = out.abs().sum(dim=reduce_dims).double()
            n_reduced = out.numel() // sums.numel()
            if key in self._sum_abs:
                self._sum_abs[key] += sums
                self._counts[key] += n_reduced
            else:
                self._sum_abs[key] = sums
                self._counts[key] = n_reduced

        return hook

    def attach(self):
        for name, module in self.activation_modules:
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def detach(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset(self):
        self._sum_abs.clear()
        self._counts.clear()
        self._call_index.clear()

    def new_forward(self):
        """Call before each forward pass so per-pass call indices line up across chunks."""
        self._call_index.clear()

    # -- the metric ----------------------------------------------------------------------

    def compute(self) -> dict:
        """Per-layer normalized scores s_i. By construction each layer's scores average to 1."""
        scores = {}
        for key in sorted(self._sum_abs):
            mean_abs = self._sum_abs[key] / max(self._counts[key], 1)
            layer_mean = mean_abs.mean()
            if layer_mean > 0:
                scores[key] = (mean_abs / layer_mean).cpu()
            else:
                # Whole layer is silent: every unit is dormant at any threshold.
                scores[key] = th.zeros_like(mean_abs).cpu()
        return scores

    def summarize(self, taus=DEFAULT_TAUS) -> dict:
        """Per-layer dormant ratios plus unit-weighted aggregates per group and overall."""
        scores = self.compute()
        taus = tuple(taus)

        per_layer = {}
        layer_units = {}
        dormant_counts = {tau: defaultdict(int) for tau in taus}
        total_units = defaultdict(int)

        for key, s in scores.items():
            n_units = s.numel()
            layer_units[key] = n_units
            per_layer[key] = {tau: float((s <= tau).sum()) / n_units for tau in taus}

            group = _group_for(key)
            total_units[group] += n_units
            total_units["all"] += n_units
            for tau in taus:
                n_dormant = int((s <= tau).sum())
                dormant_counts[tau][group] += n_dormant
                dormant_counts[tau]["all"] += n_dormant

        aggregate = {
            group: {tau: dormant_counts[tau][group] / n for tau in taus}
            for group, n in total_units.items()
            if n > 0
        }
        return {"per_layer": per_layer, "layer_units": layer_units, "aggregate": aggregate}


def collect_reference_observations(
    env_id: str, n_obs: int, seed: int, use_subproc: bool = False
) -> np.ndarray:
    """Roll a random policy to build the fixed evaluation batch.

    Uses its own env instance rather than the training env: stepping the training env from a
    callback would desync PPO's `_last_obs` from the real env state and corrupt the rollout.

    Observations are returned channel-first, matching what VecTransposeImage hands the policy
    during training, so the batch can be fed straight to `policy(obs)`.
    """
    # Imported lazily so this module can be used (and unit-tested) without MiniWorld installed.
    from stable_baselines3.common.preprocessing import (
        is_image_space,
        is_image_space_channels_first,
    )
    from stable_baselines3.common.vec_env import VecTransposeImage

    from src.envs.make_env import build_vec_env

    env = build_vec_env(env_id, n_envs=1, seed=seed, use_subproc=use_subproc)
    if is_image_space(env.observation_space) and not is_image_space_channels_first(
        env.observation_space
    ):
        env = VecTransposeImage(env)
    env.action_space.seed(seed)

    try:
        collected = [env.reset()]
        n_collected = collected[0].shape[0]
        while n_collected < n_obs:
            actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
            obs = env.step(actions)[0]
            collected.append(obs)
            n_collected += obs.shape[0]
    finally:
        env.close()

    return np.concatenate(collected, axis=0)[:n_obs]


class DormantRatioCallback(BaseCallback):
    """Logs dormant ratios over training, measured on a fixed reference batch.

    Measures at the end of a rollout (so the value lands in the same logger dump as that
    iteration's PPO stats) once at least `log_freq` env steps have passed, plus a baseline
    measurement before the first step.
    """

    def __init__(
        self,
        reference_obs: np.ndarray,
        log_freq: int,
        batch_size: int = 64,
        taus=DEFAULT_TAUS,
        primary_tau: float = 0.025,
        jsonl_path: str = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.reference_obs = reference_obs
        self.log_freq = log_freq
        self.batch_size = batch_size
        self.taus = tuple(taus)
        self.primary_tau = primary_tau
        self.jsonl_path = jsonl_path
        self._tracker = None
        self._last_log_step = None

    def _on_training_start(self) -> None:
        expected = tuple(self.model.policy.observation_space.shape)
        actual = tuple(self.reference_obs.shape[1:])
        if expected != actual:
            raise ValueError(
                f"Reference batch shape {actual} does not match the policy observation space "
                f"{expected}; the reference env is not wrapped like the training env."
            )
        self._tracker = DormantRatioTracker(self.model.policy)
        if self.jsonl_path:
            os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)
        # Baseline before any gradient step -- this is the measurement that shows whether a
        # frozen pretrained encoder is already largely inactive on MiniWorld renders.
        self._measure_and_log()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self._last_log_step >= self.log_freq:
            self._measure_and_log()

    def _measure(self) -> dict:
        policy = self.model.policy
        was_training = policy.training
        policy.set_training_mode(False)
        self._tracker.reset()
        self._tracker.attach()
        try:
            with th.no_grad():
                for start in range(0, len(self.reference_obs), self.batch_size):
                    chunk = self.reference_obs[start : start + self.batch_size]
                    self._tracker.new_forward()
                    policy(th.as_tensor(chunk, device=self.model.device), deterministic=True)
        finally:
            self._tracker.detach()
            policy.set_training_mode(was_training)
        return self._tracker.summarize(self.taus)

    def _measure_and_log(self) -> None:
        result = self._measure()
        self._last_log_step = self.num_timesteps

        for group, ratios in result["aggregate"].items():
            for tau in self.taus:
                self.logger.record(f"dormant/{group}_tau{tau}", ratios[tau])
        # Per-layer curves only at the primary threshold, to keep TensorBoard readable; the
        # JSONL below carries every layer at every tau for later aggregation and plots.
        # Excluded from stdout: SB3's table writer truncates keys to a fixed width and raises
        # on collisions, which deep paths like `...backbone.layer1.0.relu#0/#1` trigger.
        for key, ratios in result["per_layer"].items():
            self.logger.record(
                f"dormant_layers/{key}", ratios[self.primary_tau], exclude=("stdout",)
            )

        if self.jsonl_path:
            record = {
                "num_timesteps": self.num_timesteps,
                "aggregate": result["aggregate"],
                "per_layer": result["per_layer"],
                "layer_units": result["layer_units"],
            }
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        if self.verbose:
            overall = result["aggregate"]["all"][self.primary_tau]
            print(
                f"[dormant] step={self.num_timesteps} "
                f"tau={self.primary_tau} overall={overall:.4f} "
                + " ".join(
                    f"{g}={r[self.primary_tau]:.4f}"
                    for g, r in result["aggregate"].items()
                    if g != "all"
                )
            )
