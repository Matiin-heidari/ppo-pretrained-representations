import miniworld  # noqa: F401  (registers MiniWorld-* envs with gymnasium)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def build_vec_env(env_id: str, n_envs: int, seed: int, use_subproc: bool = False):
    """Build a vectorized MiniWorld env for training.

    use_subproc=True runs each env in its own process (higher throughput on multi-core
    machines) but MiniWorld's OpenGL rendering is more fragile across processes on
    Windows -- verify it actually works here before relying on it (see pilot, plan §3).
    """
    vec_env_cls = SubprocVecEnv if use_subproc else DummyVecEnv
    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_env_cls,
        env_kwargs={"render_mode": "rgb_array"},
    )
