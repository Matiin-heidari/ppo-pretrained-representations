import torch as th
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.encoders.pretrained import PretrainedBackboneMixin

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DINOV2_EMBED_DIM = 384  # dinov2_vits14, smallest variant (compute feasibility, plan §4.2)


class DINOv2FeaturesExtractor(PretrainedBackboneMixin, BaseFeaturesExtractor):
    """DINOv2 ViT-S/14, frozen by default (Tier 1); trainable=True is the Tier-2 stretch
    fine-tune ablation. Uses the [CLS] token as the pooled feature (plan §4.2 answers
    supervisor feedback point 5 on how ViT features feed the PPO heads).
    """

    def __init__(self, observation_space, trainable: bool = False):
        super().__init__(observation_space, DINOV2_EMBED_DIM)
        self.backbone = th.hub.load("facebookresearch/dinov2", "dinov2_vits14")
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
            self.backbone.eval()
        return self

    def _preprocess(self, observations: th.Tensor) -> th.Tensor:
        x = observations.float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def _forward_backbone(self, x: th.Tensor) -> th.Tensor:
        return self.backbone.forward_features(x)["x_norm_clstoken"]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self._preprocess(observations)
        if self.trainable:
            return self._forward_backbone(x)
        with th.no_grad():
            return self._forward_backbone(x)
