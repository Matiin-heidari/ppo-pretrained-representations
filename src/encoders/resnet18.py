import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.encoders.pretrained import PretrainedBackboneMixin

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESNET18_EMBED_DIM = 512


class ResNet18FeaturesExtractor(PretrainedBackboneMixin, BaseFeaturesExtractor):
    """ImageNet-pretrained ResNet18. Frozen by default (Tier 1 core comparison); set
    trainable=True for the frozen-vs-fine-tuned ablation (plan §4.3, promoted from optional
    to core per supervisor feedback). Output is the pooled 512-d feature (fc replaced with
    Identity), fed into PPO's default MLP head.
    """

    def __init__(self, observation_space, trainable: bool = False):
        super().__init__(observation_space, RESNET18_EMBED_DIM)
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.trainable = trainable

        self.register_buffer("mean", th.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", th.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

        self._snapshot_pretrained()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trainable:
            # keep BatchNorm running stats fixed even if SB3 flips the outer module to train()
            self.backbone.eval()
        return self

    def _preprocess(self, observations: th.Tensor) -> th.Tensor:
        x = observations.float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self._preprocess(observations)
        if self.trainable:
            return self.backbone(x)
        with th.no_grad():
            return self.backbone(x)
