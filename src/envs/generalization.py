"""Train/test visual-appearance split for the H3 generalization claim.

MiniWorld's domain randomization (`domain_rand=True`) redraws a handful of continuous-range
appearance parameters every episode -- sky/light color, light position, light ambient, and a
per-object color bias -- plus wall/floor/ceiling textures (picked from the fixed asset set, not
governed by a range, so not something we can split this way). Episode goal/spawn position is
*always* randomized regardless of `domain_rand` and is shared between train and test: it is
ordinary task stochasticity, not the generalization axis under test here.

We build two disjoint `DomainParams` from MiniWorld's own `DEFAULT_PARAMS`: for each of the
appearance parameters, TRAIN gets the lower half of the default range and TEST gets the upper
half (split at the midpoint, per dimension). Physics/camera parameters (forward_step, turn_step,
bot_radius, cam_*) are left at the shared default range in both -- they aren't part of what H3
asks about (visual representation transfer), and splitting them too would confound the result.

Train on TRAIN_PARAMS, evaluate generalization on TEST_PARAMS: the model never sees that half of
color/lighting space during training.
"""
import numpy as np
from miniworld.params import DEFAULT_PARAMS, DomainParams

APPEARANCE_PARAMS = ["sky_color", "light_pos", "light_color", "light_ambient", "obj_color_bias"]


def _split_params(lower_half: bool) -> DomainParams:
    params = DEFAULT_PARAMS.copy()
    for name in APPEARANCE_PARAMS:
        p = params.params[name]
        lo, hi = np.asarray(p.min, dtype=float), np.asarray(p.max, dtype=float)
        mid = (lo + hi) / 2
        new_min, new_max = (lo, mid) if lower_half else (mid, hi)
        # keep `default` inside [new_min, new_max] so DomainParams.set()'s own asserts pass
        new_default = np.clip(np.asarray(p.default, dtype=float), new_min, new_max)
        params.set(name, new_default, new_min, new_max, type=p.type)
    return params


TRAIN_PARAMS = _split_params(lower_half=True)
TEST_PARAMS = _split_params(lower_half=False)

SPLIT_PARAMS = {"train": TRAIN_PARAMS, "test": TEST_PARAMS}
