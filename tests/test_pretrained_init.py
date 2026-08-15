"""Regression tests for pretrained-weight preservation through SB3 policy construction.

SB3 orthogonally re-initializes the features extractor when building the policy, which silently
replaces pretrained backbone weights with random ones (see src/encoders/pretrained.py). These
tests pin the behaviour down so the bug cannot come back unnoticed.

No MiniWorld and no PPO: an `ActorCriticPolicy` is built directly against a fake image
observation space. ResNet18 only -- DINOv2 would need a torch.hub fetch. Run with
`python -m tests.test_pretrained_init` (or pytest) from the repo root.
"""
import gymnasium as gym
import numpy as np
import torch as th
import torchvision.models as tv_models
from stable_baselines3.common.policies import ActorCriticPolicy

from src.encoders import (
    DINOv2FeaturesExtractor,
    PretrainedBackboneMixin,
    ResNet18FeaturesExtractor,
    restore_pretrained_weights,
)

OBS_SPACE = gym.spaces.Box(low=0, high=255, shape=(3, 60, 80), dtype=np.uint8)
ACTION_SPACE = gym.spaces.Discrete(3)


def _build_policy(seed: int = 0, trainable: bool = True) -> ActorCriticPolicy:
    th.manual_seed(seed)
    return ActorCriticPolicy(
        OBS_SPACE,
        ACTION_SPACE,
        lr_schedule=lambda _: 3e-4,
        features_extractor_class=ResNet18FeaturesExtractor,
        features_extractor_kwargs={"trainable": trainable},
        normalize_images=False,
    )


def _imagenet_reference():
    return tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)


def test_sb3_clobbers_pretrained_weights_without_restore():
    """Documents the hazard: without the fix the backbone is not the pretrained one."""
    policy = _build_policy()
    ref = _imagenet_reference()
    backbone = policy.features_extractor.backbone
    assert not th.allclose(backbone.conv1.weight, ref.conv1.weight)
    # ortho init zeroes biases; pretrained ResNet18 convs have no bias, so check a BN-free
    # signature instead: deep conv weights are replaced too, not just the stem.
    assert not th.allclose(backbone.layer4[1].conv2.weight, ref.layer4[1].conv2.weight)


def test_restore_recovers_exact_pretrained_weights():
    policy = _build_policy()
    assert restore_pretrained_weights(policy) == 1
    ref = _imagenet_reference()
    backbone = policy.features_extractor.backbone
    for name, tensor in ref.state_dict().items():
        if name.startswith("fc."):
            continue  # fc is replaced with Identity
        assert th.equal(backbone.state_dict()[name], tensor), f"{name} does not match ImageNet"


def test_restored_backbone_is_seed_independent():
    """The exact regression the bug produced: identical pretrained weights must give identical
    encoder measurements regardless of the run seed."""
    restored = []
    for seed in (0, 1, 2):
        policy = _build_policy(seed=seed)
        restore_pretrained_weights(policy)
        restored.append(policy.features_extractor.backbone.conv1.weight.detach().clone())
    assert th.equal(restored[0], restored[1])
    assert th.equal(restored[0], restored[2])


def test_frozen_encoder_is_also_restored():
    policy = _build_policy(trainable=False)
    assert restore_pretrained_weights(policy) == 1
    backbone = policy.features_extractor.backbone
    assert th.allclose(backbone.conv1.weight, _imagenet_reference().conv1.weight)
    assert not any(p.requires_grad for p in backbone.parameters())


def test_snapshot_does_not_enter_state_dict():
    """The snapshot must stay a plain attribute, or every checkpoint doubles in size."""
    policy = _build_policy()
    keys = policy.features_extractor.state_dict().keys()
    assert not any("_pretrained" in k for k in keys)


def test_restore_is_idempotent():
    policy = _build_policy()
    restore_pretrained_weights(policy)
    first = policy.features_extractor.backbone.conv1.weight.detach().clone()
    restore_pretrained_weights(policy)
    assert th.equal(policy.features_extractor.backbone.conv1.weight, first)


def test_both_pretrained_extractors_use_the_mixin():
    """DINOv2 is not exercised above (needs a hub fetch), so at least pin the wiring."""
    assert issubclass(ResNet18FeaturesExtractor, PretrainedBackboneMixin)
    assert issubclass(DINOv2FeaturesExtractor, PretrainedBackboneMixin)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(TESTS)} passed")
