from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from collections.abc import Sequence

from robot_rl.utils import unpad_trajectories


class _RelPosMultiheadAttention(nn.Module):
    """Multi-head attention with learned relative-position bias (Shaw/Dai-style)."""

    def __init__(self, d_model: int, nhead: int, max_rel_dist: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.max_rel_dist = max_rel_dist

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        # One learnable scalar per (head, relative distance). Distance range: [-(max-1), +(max-1)].
        self.rel_bias = nn.Parameter(torch.zeros(nhead, 2 * max_rel_dist - 1))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend from ``query`` to ``key_value`` with relative-position bias.

        Shapes: ``query`` ``[S_q, B, D]``; ``key_value`` ``[S_kv, B, D]``; ``attn_mask`` broadcastable to
        ``[B, H, S_q, S_kv]`` as bool (True = masked out) or float (added to scores).
        """
        s_q, batch, _ = query.shape
        s_kv = key_value.shape[0]

        q = self.q_proj(query).reshape(s_q, batch, self.nhead, self.d_head).permute(1, 2, 0, 3)
        kv = self.kv_proj(key_value).reshape(s_kv, batch, self.nhead, 2 * self.d_head).permute(1, 2, 0, 3)
        k, v = kv.chunk(2, dim=-1)

        scale = 1.0 / (self.d_head**0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S_q, S_kv]

        # Relative-position bias. Query position i aligns to key position (S_kv - S_q + i); key position j is j.
        # This handles both rollout (S_q=1, S_kv=mem_len+1 -> query is the newest position) and batch mode
        # (S_q=S_kv=seq_len -> query/key positions coincide).
        q_pos = torch.arange(s_q, device=query.device) + (s_kv - s_q)
        k_pos = torch.arange(s_kv, device=query.device)
        rel = q_pos[:, None] - k_pos[None, :]  # [S_q, S_kv] in [-(S_kv-1), S_q-1]
        rel_idx = (rel + (self.max_rel_dist - 1)).clamp_(0, 2 * self.max_rel_dist - 2)
        bias = self.rel_bias[:, rel_idx]  # [H, S_q, S_kv]
        scores = scores + bias.unsqueeze(0)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B, H, S_q, d_head]
        out = out.permute(2, 0, 1, 3).reshape(s_q, batch, self.d_model)
        return self.out_proj(out)


class _TransformerXLLayer(nn.Module):
    """Pre-norm transformer block with memory-prefix keys/values."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, max_rel_dist: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _RelPosMultiheadAttention(d_model, nhead, max_rel_dist, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the block. ``mem`` ``[mem_len, B, D]`` or ``None`` prepends to keys/values before attention."""
        x_norm = self.norm1(x)
        kv = torch.cat([self.norm1(mem), x_norm], dim=0) if mem is not None and mem.size(0) > 0 else x_norm
        x = x + self.dropout(self.attn(x_norm, kv, attn_mask=attn_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class TransformerXL(nn.Module):
    """Transformer-XL memory module."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 4,
        feedforward_dims: Sequence[int] = (1024, 1024),
        dropout: float = 0.0,
        mem_len: int = 64,
        max_seq_len: int = 64,
    ) -> None:
        """Build the TXL stack with a rolling memory cache of length ``mem_len`` per layer.

        ``feedforward_dims`` is a per-layer list: ``len(feedforward_dims)`` is the number of TXL layers and
        each entry is the feed-forward width of the corresponding layer.
        """
        super().__init__()
        if mem_len < 0:
            raise ValueError(f"mem_len must be non-negative, got {mem_len}.")
        if max_seq_len < 1:
            raise ValueError(f"max_seq_len must be at least 1, got {max_seq_len}.")
        if len(feedforward_dims) < 1:
            raise ValueError(f"feedforward_dims must have at least one entry, got {list(feedforward_dims)}.")

        self.d_model = d_model
        self.num_layers = len(feedforward_dims)
        self.mem_len = mem_len
        self.max_seq_len = max_seq_len
        # Worst-case attention span governs the relative-bias table size.
        max_rel_dist = mem_len + max_seq_len + 1

        self.input_proj = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList([
            _TransformerXLLayer(d_model, nhead, ff_dim, dropout, max_rel_dist) for ff_dim in feedforward_dims
        ])
        self.norm_out = nn.LayerNorm(d_model)
        # Per-layer memory cache. Each element: [mem_len, batch, d_model]. None before the first rollout forward
        # or after a full reset.
        self.memory: tuple[torch.Tensor, ...] | None = None

    def forward(
        self,
        input: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: tuple[torch.Tensor, ...] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch to rollout or batch mode based on whether ``masks`` is provided."""
        if masks is None:
            return self._forward_rollout(input)
        return self._forward_batch(input, masks, hidden_state)

    def _forward_rollout(self, input: torch.Tensor) -> torch.Tensor:
        """Single-step inference. Updates and reads ``self.memory`` in place."""
        x = self.input_proj(input).unsqueeze(0)  # [1, batch, d_model]

        # Pre-allocate full-size zero memory so the saved-hidden-state buffer has a stable shape.
        if self.memory is None and self.mem_len > 0:
            batch = x.shape[1]
            self.memory = tuple(
                torch.zeros(self.mem_len, batch, self.d_model, device=x.device, dtype=x.dtype)
                for _ in range(self.num_layers)
            )

        prev_mem: list[torch.Tensor | None] = list(self.memory) if self.memory is not None else [None] * self.num_layers
        new_mem: list[torch.Tensor] = []

        for i, layer in enumerate(self.layers):
            mem_i = prev_mem[i]
            if self.mem_len > 0:
                # New entry is the current pre-layer activation, detached so gradients don't cross rollout steps.
                x_det = x.detach()
                combined = torch.cat([mem_i, x_det], dim=0) if mem_i is not None and mem_i.size(0) > 0 else x_det
                new_mem.append(combined[-self.mem_len :])
            x = layer(x, mem=mem_i)

        if self.mem_len > 0:
            self.memory = tuple(new_mem)

        x = self.norm_out(x)
        return x.squeeze(0)

    def _forward_batch(
        self,
        input: torch.Tensor,
        masks: torch.Tensor,
        hidden_state: tuple[torch.Tensor, ...] | torch.Tensor | None,
    ) -> torch.Tensor:
        """Run a padded trajectory segment with ``hidden_state`` as the memory prefix.

        Inputs longer than ``max_seq_len`` are split into chunks; per-layer memory carries (detached) between
        chunks so gradients flow only within each chunk's window.
        """
        x_proj = self.input_proj(input)  # [seq_len, batch, d_model]
        seq_len, batch_size, _ = x_proj.shape

        # The recurrent generator unwraps single-layer hidden state to a bare tensor; re-wrap for iteration.
        # Also cast to input dtype so per-layer cat with autocast'd activations doesn't error.
        if hidden_state is not None:
            hs = [hidden_state] if torch.is_tensor(hidden_state) else list(hidden_state)
            mem: list[torch.Tensor | None] = [h.to(x_proj.dtype) for h in hs]
        else:
            mem = [None] * self.num_layers

        outputs: list[torch.Tensor] = []
        for chunk_start in range(0, seq_len, self.max_seq_len):
            chunk_end = min(chunk_start + self.max_seq_len, seq_len)
            chunk = x_proj[chunk_start:chunk_end]  # [chunk_len, batch, d_model]
            chunk_masks = masks[chunk_start:chunk_end]  # [chunk_len, batch]
            chunk_len = chunk.shape[0]
            chunk_mem_len = mem[0].size(0) if mem[0] is not None else 0

            # Causal + key-padding mask over [batch, 1, chunk_len, mem_len + chunk_len]. Memory positions are
            # always attendable (they're rollout activations or zero, both safe).
            causal = torch.ones(chunk_len, chunk_len, dtype=torch.bool, device=input.device).triu(1)
            key_pad = ~chunk_masks.transpose(0, 1)  # [batch, chunk_len]
            if chunk_mem_len > 0:
                mem_causal = torch.zeros(chunk_len, chunk_mem_len, dtype=torch.bool, device=input.device)
                full_causal = torch.cat([mem_causal, causal], dim=1)
                mem_key_pad = torch.zeros(batch_size, chunk_mem_len, dtype=torch.bool, device=input.device)
                full_key_pad = torch.cat([mem_key_pad, key_pad], dim=1)
            else:
                full_causal = causal
                full_key_pad = key_pad
            attn_mask = full_causal[None, None, :, :] | full_key_pad[:, None, None, :]

            new_mem: list[torch.Tensor] = []
            h = chunk
            for i, layer in enumerate(self.layers):
                mem_i = mem[i]
                if self.mem_len > 0:
                    h_det = h.detach()
                    combined = torch.cat([mem_i, h_det], dim=0) if mem_i is not None and mem_i.size(0) > 0 else h_det
                    new_mem.append(combined[-self.mem_len :])
                h = layer(h, mem=mem_i, attn_mask=attn_mask)

            if self.mem_len > 0:
                mem = new_mem  # type: ignore[assignment]
            outputs.append(h)

        x = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        x = self.norm_out(x)
        return unpad_trajectories(x, masks)

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        """Reset memory for all envs (``dones=None``) or per-env (``dones[i]==1`` zeros env i's memory in place)."""
        if dones is None:
            self.memory = hidden_state  # None by default
            return
        if hidden_state is not None:
            raise NotImplementedError(
                "Resetting the memory of done environments with a custom hidden state is not implemented."
            )
        if self.memory is None:
            return
        done_mask = dones == 1
        if not done_mask.any():
            return
        for mem in self.memory:
            mem[:, done_mask, :] = 0.0

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach memory from the computation graph. No-op if memory is already detached (typical during rollout)."""
        if self.memory is None:
            return
        if dones is None:
            self.memory = tuple(mem.detach() for mem in self.memory)
            return
        done_mask = dones == 1
        if not done_mask.any():
            return
        for mem in self.memory:
            mem[:, done_mask, :] = mem[:, done_mask, :].detach()
