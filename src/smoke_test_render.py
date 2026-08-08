"""Day-1 feasibility check: does MiniWorld render on this machine at all?

Not part of the training pipeline. Run once, look at the saved frames, then move on.
"""
import gymnasium as gym
import miniworld  # noqa: F401  (registers MiniWorld-* envs with gymnasium)
import numpy as np
from PIL import Image

RESULTS_DIR = "results/smoke_test"


def main():
    import os

    os.makedirs(RESULTS_DIR, exist_ok=True)

    miniworld_ids = [k for k in gym.registry.keys() if k.startswith("MiniWorld-")]
    print(f"Found {len(miniworld_ids)} MiniWorld envs, e.g.: {miniworld_ids[:5]}")

    env_id = "MiniWorld-OneRoom-v0" if "MiniWorld-OneRoom-v0" in miniworld_ids else miniworld_ids[0]
    print(f"Using: {env_id}")

    env = gym.make(env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    print(f"obs shape: {obs.shape}, dtype: {obs.dtype}, action_space: {env.action_space}")

    frames = [obs]
    total_reward = 0.0
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if i % 10 == 0:
            frames.append(obs)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    for i, frame in enumerate(frames):
        Image.fromarray(np.asarray(frame)).save(f"{RESULTS_DIR}/frame_{i:02d}.png")

    print(f"Saved {len(frames)} frames to {RESULTS_DIR}/")
    print(f"Ran 50 random steps, total reward: {total_reward:.3f}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
