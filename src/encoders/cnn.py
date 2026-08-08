import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    """CNN-from-scratch baseline (Nature-CNN-style), trained end to end with the policy.

    Expects channel-first uint8 observations (SB3's VecTransposeImage handles the
    HWC -> CHW conversion automatically for image observation spaces).
    """

    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # named as attributes (not a single nn.Sequential) so the dormant-ratio
        # hooks (see src/dormant_ratio.py, added later) can attach to each ReLU by name.
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()

        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            n_flatten = self._conv_forward(sample).shape[1]

        self.linear = nn.Linear(n_flatten, features_dim)
        self.relu_out = nn.ReLU()

    def _conv_forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.relu1(self.conv1(observations))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        return self.flatten(x)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.float() / 255.0
        x = self._conv_forward(x)
        return self.relu_out(self.linear(x))
