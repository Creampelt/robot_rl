from __future__ import annotations

import torch
from tensordict import TensorDict

from robot_rl.modules import HiddenState

from .mlp_model import MLPModel, _OnnxMLPModel, _TorchMLPModel


class DiscriminatorModel(MLPModel):
    """Discriminator MLP-based neural model. See :class:`MLPModel`."""

    def forward(
        self,
        obs: TensorDict,
        *args: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        raw_logits: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the MLP model. If raw_logits is False, will apply sigmoid to the MLP output.

        ..note::
            The `stochastic_output` flag only has an effect if the model has a distribution (i.e., ``distribution_cfg``
            was provided) and defaults to ``False``, meaning that even stochastic models will return deterministic
            outputs by default.
        """
        out = super().forward(obs, *args, masks=masks, hidden_state=hidden_state, stochastic_output=stochastic_output)
        if not raw_logits:
            out = torch.sigmoid(out)
        return out


class _TorchDiscriminatorModel(_TorchMLPModel):
    """Exportable discriminator MLP model for JIT."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on pre-concatenated observations."""
        out = super().forward(x)
        return torch.sigmoid(out)


class _OnnxDiscriminatorModel(_OnnxMLPModel):
    """Exportable discriminator MLP model for ONNX."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on pre-concatenated observations."""
        out = super().forward(x)
        return torch.sigmoid(out)
