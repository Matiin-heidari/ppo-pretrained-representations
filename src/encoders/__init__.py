from src.encoders.cnn import CNNFeaturesExtractor
from src.encoders.dinov2 import DINOv2FeaturesExtractor
from src.encoders.resnet18 import ResNet18FeaturesExtractor

ENCODER_REGISTRY = {
    "cnn": CNNFeaturesExtractor,
    "resnet18": ResNet18FeaturesExtractor,
    "dinov2": DINOv2FeaturesExtractor,
}
