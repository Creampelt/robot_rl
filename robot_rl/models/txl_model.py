from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from robot_rl.models import MLPModel
from robot_rl.modules import HiddenState, TransformerXL


class TXLModel(MLPModel):
    """Transformer-XL-based neural model.

    Uses a :class:`~robot_rl.modules.transformer_xl.TransformerXL` memory module to process 1D observation
    groups before passing the resulting latent to an MLP head. Drop-in alternative to
    :class:`~robot_rl.models.rnn_model.RNNModel` for environments that benefit from longer memory (e.g.,
    meta-RL where trials span multiple episodes).
    """

    is_recurrent: bool = True
    """Whether the model contains a stateful memory module."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (1024, 1024),
        activation: str = "gelu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        txl_hidden_dim: int = 256,
        txl_num_heads: int = 4,
        txl_dropout: float = 0.0,
        txl_mem_len: int = 64,
        txl_max_seq_len: int = 64,
        memory_only: bool = False,
    ) -> None:
        """Initialize the TXL-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Per-layer feed-forward widths of the TXL stack. ``len(hidden_dims)`` is the number
                of TXL layers and each entry is the feed-forward width of that layer. (TXLModel does not have
                an MLP head: the transformer output is the model output, with at most a single linear
                projection to ``output_dim`` when ``memory_only=False``.)
            activation: Unused (the TXL stack's internal feed-forward activation is GELU, fixed). Accepted
                for cfg-class symmetry with :class:`MLPModel` / :class:`RNNModel`.
            obs_normalization: Whether to normalize the observations before feeding them to the TXL.
            distribution_cfg: Configuration dictionary for the output distribution. Only used when
                ``memory_only=False``.
            txl_hidden_dim: TXL model/latent width. Must be divisible by ``txl_num_heads``.
            txl_num_heads: Number of attention heads per layer.
            txl_dropout: Dropout probability used in attention and feed-forward blocks.
            txl_mem_len: Rolling memory length used during rollout. Set to 0 to disable memory (stateless
                causal transformer over a single token).
            txl_max_seq_len: Largest training segment length the relative-position bias table must cover. Must
                be at least the longest padded trajectory seen at update time.
            memory_only: When ``True``, skip the optional output head and return the TXL latent directly.
                Used when this model serves as a shared memory module under :class:`MetaRlCfg.memory`.
        """
        self.latent_dim = txl_hidden_dim

        # When memory_only, bypass MLP-head
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=[],  # Single layer input -> output
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            memory_only=memory_only,
        )

        self.memory_module = TransformerXL(
            input_size=self.obs_dim,
            d_model=txl_hidden_dim,
            nhead=txl_num_heads,
            feedforward_dims=hidden_dims,
            dropout=txl_dropout,
            mem_len=txl_mem_len,
            max_seq_len=txl_max_seq_len,
        )

    def get_latent(
        self, obs: TensorDict, *args: torch.Tensor, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by passing normalized observations through the TXL memory module."""
        latent = super().get_latent(obs, *args)
        return self.memory_module(latent, masks, hidden_state)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the TXL memory cache."""
        self.memory_module.reset(dones, hidden_state)  # type: ignore[arg-type]

    def get_hidden_state(self) -> HiddenState:
        """Return the per-layer rolling memory cache as a tuple of ``[mem_len, num_envs, d_model]`` tensors."""
        return self.memory_module.memory  # type: ignore[return-value]

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the TXL memory. Typically a no-op during rollout because memory is detached on write."""
        self.memory_module.detach_hidden_state(dones)

    def as_jit(self) -> nn.Module:
        """TorchScript export for TXL is not implemented yet (deferred; see CLAUDE.md)."""
        raise NotImplementedError("TorchScript export for TXLModel is not implemented yet.")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """ONNX export for TXL is not implemented yet (deferred; see CLAUDE.md)."""
        raise NotImplementedError("ONNX export for TXLModel is not implemented yet.")

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.latent_dim
