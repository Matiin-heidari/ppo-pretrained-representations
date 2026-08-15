from src.encoders.cnn import CNNFeaturesExtractor
from src.encoders.dinov2 import DINOv2FeaturesExtractor
from src.encoders.pretrained import PretrainedBackboneMixin, ScaledLRAdam
from src.encoders.resnet18 import ResNet18FeaturesExtractor

ENCODER_REGISTRY = {
    "cnn": CNNFeaturesExtractor,
    "resnet18": ResNet18FeaturesExtractor,
    "dinov2": DINOv2FeaturesExtractor,
}

# SB3 exposes the extractor under up to three attributes; with the default
# share_features_extractor=True they are all the same object, so dedupe by identity.
_EXTRACTOR_ATTRS = ("features_extractor", "pi_features_extractor", "vf_features_extractor")


def restore_pretrained_weights(policy) -> int:
    """Put pretrained backbone weights back after SB3 has built (and re-initialized) the policy.

    Must be called once, immediately after the SB3 model is constructed -- see
    `src/encoders/pretrained.py` for why this is necessary. A no-op for the from-scratch CNN,
    which is supposed to keep SB3's orthogonal initialization.

    Returns the number of extractors restored, so callers can assert it matched expectations.
    """
    restored = 0
    for extractor in _distinct_extractors(policy):
        if isinstance(extractor, PretrainedBackboneMixin):
            extractor.restore_pretrained()
            restored += 1
    return restored


def _distinct_extractors(policy):
    seen = set()
    for attr in _EXTRACTOR_ATTRS:
        extractor = getattr(policy, attr, None)
        if extractor is not None and id(extractor) not in seen:
            seen.add(id(extractor))
            yield extractor


def apply_backbone_lr_scale(policy, base_lr: float, scale: float) -> bool:
    """Give a *trainable* pretrained backbone a lower learning rate than the PPO heads.

    Measured on this task: at a uniform 3e-4, 25k steps of PPO fine-tuning already rotate the
    ResNet18 feature vectors to cosine 0.76 against their ImageNet values (median weight drift
    5.4%, worst tensor 39%) -- i.e. over a 500k budget the pretrained initialization is
    substantially overwritten, and the fine-tune arm drifts toward being a from-scratch run
    under another name. At a 0.1x backbone scale the same 25k steps keep cosine at 0.85 with
    median weight drift 0.6%, which is adaptation rather than replacement.

    No-op for frozen encoders (nothing to scale) and for the from-scratch CNN, which is
    supposed to train at the full rate. Returns whether a scaled optimizer was installed.
    """
    if scale == 1.0:
        return False
    backbones = [
        e.backbone
        for e in _distinct_extractors(policy)
        if isinstance(e, PretrainedBackboneMixin) and getattr(e, "trainable", False)
    ]
    if not backbones:
        return False

    backbone_params = [p for bb in backbones for p in bb.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in policy.parameters() if id(p) not in backbone_ids]
    policy.optimizer = ScaledLRAdam(
        [
            {"params": backbone_params, "lr_scale": scale},
            {"params": other_params, "lr_scale": 1.0},
        ],
        lr=base_lr,
        eps=1e-5,  # matches SB3's default for Adam
    )
    return True
