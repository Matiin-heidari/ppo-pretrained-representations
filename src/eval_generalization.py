"""Generalization evaluation for H3: does a trained policy hold up on held-out visual conditions?

Evaluates a trained checkpoint on both halves of the train/test appearance split
(src/envs/generalization.py) and reports the gap between them. Both splits use a fixed eval seed,
independent of the run's training seed, so every checkpoint we ever evaluate is judged on the
exact same episodes -- that's what makes the train-vs-test gap comparable across encoders/seeds.
"""
import argparse
import json
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env.patch_gym import _convert_space

from src.envs.generalization import SPLIT_PARAMS
from src.envs.make_env import build_vec_env
from src.stats import t_crit_95
from src.train import RESULTS_DIR, load_config

# Fixed independently of any run's training seed, and shared across every checkpoint we ever
# evaluate -- so "train vs. test" comparisons are apples-to-apples across encoders/seeds.
DEFAULT_EVAL_SEED = 999983


def _build_eval_env(env_id: str, split: str, seed: int):
    env = build_vec_env(env_id, n_envs=1, seed=seed, domain_rand=True, params=SPLIT_PARAMS[split])
    if is_image_space(env.observation_space) and not is_image_space_channels_first(
        env.observation_space
    ):
        env = VecTransposeImage(env)
    return env


def _mean_ci95(values: np.ndarray) -> dict:
    n = len(values)
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "sem": sem, "ci95": t_crit_95(n) * sem, "n": n}


def _load_policy_only(checkpoint_path: str, device: str = "auto") -> PPO:
    """Load a checkpoint for inference, ignoring the saved optimizer state.

    `PPO.load` restores `policy.optimizer` too and requires its structure to match the one the
    freshly built policy happens to have. Fine-tune runs give the backbone its own param group
    (see `apply_backbone_lr_scale`), so those checkpoints carry a two-group optimizer and a
    plain `PPO.load` raises "loaded state dict has a different number of parameter groups".
    Evaluation only ever needs the weights, so drop the optimizer entry and load the rest --
    which also keeps old checkpoints readable if the optimizer layout changes again.
    """
    data, params, _pytorch_variables = load_from_zip_file(checkpoint_path, device=device)
    for key in ("observation_space", "action_space"):
        data[key] = _convert_space(data[key])

    model = PPO(policy=data["policy_class"], env=None, device=device, _init_setup_model=False)
    model.__dict__.update(data)
    model._setup_model()
    model.set_parameters({"policy": params["policy"]}, exact_match=False, device=device)
    model.policy.set_training_mode(False)
    return model


def evaluate_checkpoint(
    checkpoint_path: str,
    env_id: str,
    n_episodes: int = 30,
    seed: int = DEFAULT_EVAL_SEED,
    deterministic: bool = True,
) -> dict:
    model = _load_policy_only(checkpoint_path)

    results = {}
    for split in ("train", "test"):
        env = _build_eval_env(env_id, split, seed)
        try:
            returns, _lengths = evaluate_policy(
                model,
                env,
                n_eval_episodes=n_episodes,
                deterministic=deterministic,
                return_episode_rewards=True,
            )
        finally:
            env.close()
        results[split] = _mean_ci95(np.array(returns, dtype=float))

    results["generalization_gap"] = results["train"]["mean"] - results["test"]["mean"]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the run's YAML config")
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = cfg["run_name"]
    checkpoint_path = os.path.join(
        RESULTS_DIR, "checkpoints", run_name, f"ckpt_{args.checkpoint_step}.zip"
    )

    results = evaluate_checkpoint(
        checkpoint_path, cfg["env_id"], n_episodes=args.n_episodes, seed=args.seed
    )
    results["run_name"] = run_name
    results["checkpoint_step"] = args.checkpoint_step

    out_dir = os.path.join(RESULTS_DIR, "generalization")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_name}_{args.checkpoint_step}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"[{run_name} @ {args.checkpoint_step}] "
        f"train={results['train']['mean']:.3f}+/-{results['train']['ci95']:.3f}  "
        f"test={results['test']['mean']:.3f}+/-{results['test']['ci95']:.3f}  "
        f"gap={results['generalization_gap']:.3f}  -> {out_path}"
    )
