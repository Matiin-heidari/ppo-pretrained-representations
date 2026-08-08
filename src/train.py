import argparse
import os
import time

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.encoders import ENCODER_REGISTRY
from src.envs.make_env import build_vec_env

RESULTS_DIR = "results"


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
    if cfg["encoder"] == "dinov2":
        # Reproduced on this machine: DINOv2's forward races against the trainable heads'
        # backward kernels on the default CUDA stream and occasionally corrupts a slice of
        # the output batch with NaN (only reproduces in the live training loop, never on
        # offline replay of the identical input -- and persists across both cu126 and cu130
        # torch builds, so it isn't a single-toolkit regression). CUDA_LAUNCH_BLOCKING=1
        # serializes kernel launches and reliably eliminates it (validated over 8192 steps),
        # at a real cost (~5x slower). Scoped to DINOv2 runs only so CNN/ResNet18 stay fast.
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    run_name = cfg["run_name"]
    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints", run_name)
    tensorboard_dir = os.path.join(RESULTS_DIR, "tensorboard")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    env = build_vec_env(
        env_id=cfg["env_id"],
        n_envs=cfg["n_envs"],
        seed=cfg["seed"],
        use_subproc=cfg.get("use_subproc", False),
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

    callback = MilestoneCheckpointCallback(cfg["checkpoint_steps"], checkpoint_dir, verbose=1)

    start = time.time()
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=callback,
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
