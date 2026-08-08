"""Unit tests for the dormant-neuron score.

Deliberately free of MiniWorld/PPO: builds tiny networks with a known number of forced-dead
units and checks the tracker recovers exactly that. Run with `python -m tests.test_dormant_ratio`
(or pytest) from the repo root.
"""
import torch as th
import torch.nn as nn

from src.dormant_ratio import DormantRatioTracker


class TinyConvNet(nn.Module):
    """Conv (4 channels) -> ReLU -> pool -> Linear (6 units) -> ReLU."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4, 6)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv(x))
        x = self.flatten(self.pool(x))
        return self.relu2(self.fc(x))


def _kill_units(layer, indices):
    """Force the given output units to emit 0 for any input (dead ReLU)."""
    with th.no_grad():
        for i in indices:
            layer.weight[i].zero_()
            layer.bias[i] = -1e4


def _run(module, tracker, x, n_chunks=2):
    tracker.attach()
    try:
        with th.no_grad():
            for chunk in x.chunk(n_chunks):
                tracker.new_forward()
                module(chunk)
    finally:
        tracker.detach()


def test_dead_units_are_detected():
    th.manual_seed(0)
    net = TinyConvNet()
    # Make surviving units clearly positive so they are not dormant by accident.
    with th.no_grad():
        net.conv.bias.fill_(1.0)
        net.fc.bias.fill_(1.0)
    _kill_units(net.conv, [1])  # 1 of 4 channels
    _kill_units(net.fc, [0, 2, 5])  # 3 of 6 features

    tracker = DormantRatioTracker(net)
    _run(net, tracker, th.randn(16, 3, 8, 8))
    result = tracker.summarize(taus=(0.0, 0.025, 0.1))

    assert result["per_layer"]["relu1#0"][0.0] == 1 / 4
    assert result["per_layer"]["relu2#0"][0.0] == 3 / 6
    # Unit-weighted aggregate over all 10 units, 4 of them dead.
    assert result["aggregate"]["all"][0.0] == 4 / 10
    assert result["layer_units"] == {"relu1#0": 4, "relu2#0": 6}


def test_scores_average_to_one_and_are_scale_free():
    """Per-layer normalization is what makes different-width encoders comparable, so scaling
    a layer's activations must not change its dormant ratio."""
    th.manual_seed(0)
    net = TinyConvNet()
    with th.no_grad():
        net.conv.bias.fill_(1.0)
        net.fc.bias.fill_(1.0)
    _kill_units(net.fc, [3])
    x = th.randn(8, 3, 8, 8)

    tracker = DormantRatioTracker(net)
    _run(net, tracker, x)
    baseline = tracker.summarize()
    for scores in tracker.compute().values():
        assert th.isclose(scores.mean(), th.tensor(1.0, dtype=scores.dtype), atol=1e-5)

    with th.no_grad():  # scale the whole fc layer up by 1000x
        net.fc.weight *= 1000.0
        net.fc.bias *= 1000.0
    tracker.reset()
    _run(net, tracker, x)
    assert tracker.summarize()["per_layer"] == baseline["per_layer"]


def test_all_silent_layer_is_fully_dormant():
    """A layer with zero total activation would divide by zero; it must read as 100% dormant."""
    th.manual_seed(0)
    net = TinyConvNet()
    _kill_units(net.fc, range(6))

    tracker = DormantRatioTracker(net)
    _run(net, tracker, th.randn(8, 3, 8, 8))
    result = tracker.summarize()
    assert result["per_layer"]["relu2#0"][0.0] == 1.0


def test_repeated_module_calls_are_kept_separate():
    """torchvision's BasicBlock reuses one nn.ReLU twice per forward; the two call sites must
    not be merged into a single 'layer'."""

    class Reused(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)
            self.relu = nn.ReLU()  # single module, two call sites

        def forward(self, x):
            return self.relu(self.fc2(self.relu(self.fc1(x))))

    th.manual_seed(0)
    net = Reused()
    with th.no_grad():
        net.fc1.bias.fill_(1.0)
        net.fc2.bias.fill_(1.0)
    _kill_units(net.fc1, [0])
    _kill_units(net.fc2, [1, 2])

    tracker = DormantRatioTracker(net)
    _run(net, tracker, th.randn(8, 4), n_chunks=2)
    result = tracker.summarize()

    assert set(result["per_layer"]) == {"relu#0", "relu#1"}
    assert result["per_layer"]["relu#0"][0.0] == 1 / 4
    assert result["per_layer"]["relu#1"][0.0] == 2 / 4


def test_token_shaped_activations_reduce_over_tokens():
    """ViT activations are (batch, tokens, dim); a unit is a dim, not a token."""

    class TokenNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(5, 8)
            self.act = nn.GELU()

        def forward(self, x):
            return self.act(self.fc(x))

    th.manual_seed(0)
    net = TokenNet()
    tracker = DormantRatioTracker(net)
    _run(net, tracker, th.randn(6, 7, 5), n_chunks=2)  # 6 batch, 7 tokens, 5 dim

    scores = tracker.compute()["act#0"]
    assert scores.numel() == 8  # one score per output dim, not per token


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(TESTS)} passed")
