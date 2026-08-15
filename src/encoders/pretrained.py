"""Keeps pretrained backbone weights alive through SB3's policy construction.

SB3's `ActorCriticPolicy._build` runs orthogonal initialization over `self.features_extractor`
whenever `ortho_init=True` (its default): every `nn.Conv2d`/`nn.Linear` weight is replaced by a
fresh orthogonal matrix with gain sqrt(2) and every bias is zeroed. That happens *after* the
extractor's `__init__` has loaded the pretrained checkpoint, so a "pretrained" encoder silently
ends up randomly initialized -- and for a frozen encoder those random weights are then frozen.
Only `Conv2d`/`Linear` are touched, so BatchNorm/LayerNorm buffers survive, which makes the
damage easy to miss: the module still looks like a loaded checkpoint.

We keep `ortho_init=True` rather than switching it off, because it is also what gives the PPO
policy/value heads their standard initialization (notably `action_net` at gain 0.01, which keeps
the initial policy near-uniform) and what correctly initializes the from-scratch CNN baseline.
Instead each pretrained extractor snapshots its backbone at construction and puts it back once
the policy exists -- see `restore_pretrained_weights` in `src/encoders/__init__.py`.
"""
import torch as th


class ScaledLRAdam(th.optim.Adam):
    """Adam with a per-parameter-group learning-rate multiplier applied inside `step()`.

    The obvious way to fine-tune a backbone more slowly than the heads -- build Adam with two
    param groups at different `lr` -- does not work under SB3. Before every update it calls
    `_update_learning_rate`, which writes the single scheduled learning rate into *every*
    param group, so the per-group values are silently discarded and the run trains as if
    nothing had been configured. Applying the multiplier at step time survives that.
    """

    def step(self, closure=None):
        originals = [group["lr"] for group in self.param_groups]
        for group in self.param_groups:
            group["lr"] = group["lr"] * group.get("lr_scale", 1.0)
        try:
            return super().step(closure)
        finally:
            for group, lr in zip(self.param_groups, originals):
                group["lr"] = lr


class PretrainedBackboneMixin:
    """Mixin for feature extractors wrapping a pretrained `self.backbone`.

    Call `_snapshot_pretrained()` at the end of `__init__`, once `self.backbone` is loaded.
    """

    def _snapshot_pretrained(self) -> None:
        # Plain attribute, not a registered buffer: it must not end up in `state_dict()`, or
        # every checkpoint would carry a second copy of the backbone.
        self._pretrained_backbone_state = {
            name: tensor.detach().clone()
            for name, tensor in self.backbone.state_dict().items()
        }

    def restore_pretrained(self) -> None:
        """Undo any re-initialization applied to the backbone since construction.

        Safe to call more than once, and device-agnostic: `load_state_dict` copies into the
        existing parameters, so a CPU snapshot restores onto a CUDA backbone fine.
        """
        self.backbone.load_state_dict(self._pretrained_backbone_state)
