import miniworld  # noqa: F401  (registers MiniWorld-* envs with gymnasium)
from miniworld.params import DomainParams
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def build_vec_env(
    env_id: str,
    n_envs: int,
    seed: int,
    use_subproc: bool = False,
    domain_rand: bool = False,
    params: DomainParams = None,
):
    """Build a vectorized MiniWorld env for training or evaluation.

    use_subproc=True runs each env in its own process (higher throughput on multi-core
    machines) but MiniWorld's OpenGL rendering is more fragile across processes on
    Windows -- verify it actually works here before relying on it (see pilot, plan §3).

    domain_rand/params control the visual-appearance train/test split (src/envs/generalization.py,
    plan §4.1): pass domain_rand=True with generalization.TRAIN_PARAMS or .TEST_PARAMS to draw
    episodes from one side of the split. Left off by default so existing configs are unaffected.
    """
    vec_env_cls = SubprocVecEnv if use_subproc else DummyVecEnv
    env_kwargs = {"render_mode": "rgb_array"}
    if domain_rand:
        env_kwargs["domain_rand"] = True
    if params is not None:
        env_kwargs["params"] = params
    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_env_cls,
        env_kwargs=env_kwargs,
    )
