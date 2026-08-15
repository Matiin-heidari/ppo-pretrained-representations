from src.encoders.cnn import CNNFeaturesExtractor
from src.encoders.dinov2 import DINOv2FeaturesExtractor
from src.encoders.pretrained import PretrainedBackboneMixin
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
    seen = set()
    for attr in _EXTRACTOR_ATTRS:
        extractor = getattr(policy, attr, None)
        if extractor is None or id(extractor) in seen:
            continue
        seen.add(id(extractor))
        if isinstance(extractor, PretrainedBackboneMixin):
            extractor.restore_pretrained()
            restored += 1
    return restored
