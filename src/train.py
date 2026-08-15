import argparse
import os
import time

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.dormant_ratio import (
    DEFAULT_TAUS,
    DormantRatioCallback,
    collect_reference_observations,
)
from src.encoders import ENCODER_REGISTRY, restore_pretrained_weights
from src.envs.generalization import TRAIN_PARAMS
from src.envs.make_env import build_vec_env

RESULTS_DIR = "results"

# The reference batch for dormant-ratio tracking is deliberately independent of the run seed:
# every encoder and every seed is measured on the identical observations, which is what makes
# the ratios comparable across runs. Don't tie this to cfg["seed"].
DEFAULT_REFERENCE_SEED = 12345


class MilestoneCheckpointCallback(BaseCallback):
    """Saves a checkpoint the first time training crosses each step in `milestones`.

    Using exact milestones (100k/250k/500k) rather than a fixed save interval, since the
    experiment design treats these as the actual evaluation budgets (plan §6).
    """

    def __init__(self, milestones, save_dir, verbose=0):
        super().__init__(verbose)
        self.milestones = sorted(milestones)
        self.next_idx = 0
        self.save_dir = save_dir

    def _on_step(self) -> bool:
        while (
            self.next_idx < len(self.milestones)
            and self.num_timesteps >= self.milestones[self.next_idx]
        ):
            step = self.milestones[self.next_idx]
            path = os.path.join(self.save_dir, f"ckpt_{step}")
            self.model.save(path)
            if self.verbose:
                print(f"[checkpoint] saved {path} at num_timesteps={self.num_timesteps}")
            self.next_idx += 1
        return True


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(cfg: dict):
    is_finetuning = cfg.get("encoder_kwargs", {}).get("trainable", False)
    if cfg["encoder"] == "dinov2" or is_finetuning:
        # Reproduced on this machine: async CUDA kernels from a large pretrained backbone (its
        # own forward, or its backward when fine-tuned) race against the trainable heads' kernels
        # on the default stream. For frozen DINOv2 this silently corrupts a slice of the output
        # batch with NaN; for fine-tuned ResNet18 it crashed with `cudaErrorIllegalInstruction`.
        # Neither reproduces offline (identical inputs replayed outside the training loop are
        # fine) or is fixed by disabling TF32/cudnn.benchmark, and persists identically across
        # cu126, cu130, AND cu132 torch builds, and across a driver update (592.82 -> 610.88) --
        # this is not a single buggy kernel or a CUDA-13.0-specific compiler bug (we suspected
        # the latter -- CUDA 13.0/13.1 has a confirmed illegal-instruction compiler bug -- but it
        # persisted unchanged after upgrading to cu132, which has that bug fixed). Looks like a
        # deeper driver/scheduler issue on this machine. CUDA_LAUNCH_BLOCKING=1 serializes kernel
        # launches and reliably eliminates it (validated: DINOv2 over 8192 steps, ResNet18-finetune
        # over 4096 steps, zero failures on every torch/driver combo tried), at a real cost
        # (roughly 4-40x slower depending on how much of the step is the large backbone). Plain
        # CNN and *frozen* ResNet18/DINOv2 never hit this, so it's scoped to only the runs that
        # need it.
        #
        # Also tried `torch.use_deterministic_algorithms(True)` + CUBLAS_WORKSPACE_CONFIG as a
        # cheaper alternative (avoids atomicAdd-based nondeterministic kernels rather than
        # serializing every launch): ran clean with zero failures, but at the same throughput as
        # CUDA_LAUNCH_BLOCKING (3.7 vs 3.3-4 steps/s on ResNet18-finetune) -- no win, so not used.
        #
        # Not yet confirmed whether this is specific to this one machine/GPU or affects other
        # hardware too. PPO_SKIP_CUDA_RACE_FIX=1 disables the workaround so a teammate on
        # different hardware can check whether they hit the same bug at all -- see
        # temp/PROJECT_PLAN.md §2 for the check procedure. Leave unset for any real run.
        if os.environ.get("PPO_SKIP_CUDA_RACE_FIX") != "1":
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    run_name = cfg["run_name"]
    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints", run_name)
    tensorboard_dir = os.path.join(RESULTS_DIR, "tensorboard")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Collected before the training env exists so only one MiniWorld GL context is alive at a
    # time -- MiniWorld's renderer is the fragile part of this pipeline.
    dormant_cfg = cfg.get("dormant_ratio") or {}
    dormant_enabled = dormant_cfg.get("enabled", True)
    reference_obs = None
    if dormant_enabled:
        reference_obs = collect_reference_observations(
            env_id=cfg["env_id"],
            n_obs=dormant_cfg.get("n_reference_obs", 512),
            seed=dormant_cfg.get("reference_seed", DEFAULT_REFERENCE_SEED),
        )

    # Trains on the lower half of MiniWorld's appearance-randomization ranges (see
    # src/envs/generalization.py); src/eval_generalization.py checks the held-out upper half
    # for H3. Without this, training sees a single fixed appearance and there is nothing to
    # generalize *from*.
    generalization_enabled = cfg.get("generalization", {}).get("enabled", True)
    env = build_vec_env(
        env_id=cfg["env_id"],
        n_envs=cfg["n_envs"],
        seed=cfg["seed"],
        use_subproc=cfg.get("use_subproc", False),
        domain_rand=generalization_enabled,
        params=TRAIN_PARAMS if generalization_enabled else None,
    )

    encoder_cls = ENCODER_REGISTRY[cfg["encoder"]]
    policy_kwargs = dict(
        features_extractor_class=encoder_cls,
        features_extractor_kwargs=cfg.get("encoder_kwargs", {}),
        # SB3 normalizes image obs to [0,1] itself before calling the features extractor;
        # our extractors each already do their own /255 (and, for the pretrained ones,
        # ImageNet mean/std normalization on top), so SB3's own normalization must be off
        # to avoid dividing by 255 twice.
        normalize_images=False,
    )

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_dir,
        seed=cfg["seed"],
        verbose=1,
    )

    # SB3 builds the policy with ortho_init=True, which orthogonally re-initializes every
    # Conv2d/Linear inside the features extractor -- overwriting the pretrained ResNet18/DINOv2
    # weights that the extractor's __init__ just loaded. Putting them back here (rather than
    # setting ortho_init=False) keeps the PPO heads on their standard init and leaves the
    # from-scratch CNN correctly orthogonally initialized. See src/encoders/pretrained.py.
    n_restored = restore_pretrained_weights(model.policy)
    print(f"[encoder] {cfg['encoder']}: restored pretrained weights on {n_restored} extractor(s)")

    callbacks = [MilestoneCheckpointCallback(cfg["checkpoint_steps"], checkpoint_dir, verbose=1)]
    if dormant_enabled:
        callbacks.append(
            DormantRatioCallback(
                reference_obs=reference_obs,
                log_freq=dormant_cfg.get("log_freq", 10000),
                batch_size=dormant_cfg.get("batch_size", 64),
                taus=dormant_cfg.get("taus", list(DEFAULT_TAUS)),
                jsonl_path=os.path.join(RESULTS_DIR, "dormant", f"{run_name}.jsonl"),
                verbose=1,
            )
        )

    start = time.time()
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=callbacks,
        tb_log_name=run_name,
    )
    elapsed = time.time() - start
    steps_per_sec = cfg["total_timesteps"] / elapsed
    print(f"Finished {cfg['total_timesteps']} steps in {elapsed:.1f}s ({steps_per_sec:.1f} steps/s)")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to a YAML run config")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
