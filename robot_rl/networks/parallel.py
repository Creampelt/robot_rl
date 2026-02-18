from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_parallel: int,
        bias: bool = True,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_parallel = num_parallel
        self.use_linear = num_parallel == 1

        if self.use_linear:
            weight_dim = (output_dim, input_dim)
            bias_dim = (output_dim,)
        else:
            weight_dim = (num_parallel, input_dim, output_dim)
            bias_dim = (num_parallel, 1, output_dim)

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(weight_dim, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(bias_dim, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        gain = nn.init.calculate_gain("relu")
        if self.use_linear:
            nn.init.orthogonal_(self.weight.data, gain=gain)
        else:
            tensor = self.weight.data
            n_parallel = tensor.size(0)
            rows = tensor.size(1)
            cols = tensor.numel() // n_parallel // rows
            flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)

            qs = []
            for flat_tensor in torch.unbind(flattened, dim=0):
                if rows < cols:
                    flat_tensor.t_()

                # Compute the qr factorization
                q, r = torch.linalg.qr(flat_tensor)
                # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
                d = torch.diag(r, 0)
                ph = d.sign()
                q *= ph

                if rows < cols:
                    q.t_()
                qs.append(q)

            qs = torch.stack(qs, dim=0)
            with torch.no_grad():
                tensor.view_as(qs).copy_(qs)
                tensor.mul_(gain)
        self.bias.data.fill_(0.0)

    def forward(self, x) -> torch.Tensor:
        if self.use_linear:
            return F.linear(x, self.weight, self.bias)
        else:
            return torch.baddbmm(self.bias, x, self.weight)

    def extra_repr(self) -> str:
        return f"in_features={self.input_dim}, out_features={self.output_dim}, num_parallel={self.num_parallel}, bias={self.bias is not None}"


class ParallelLayerNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        num_parallel: int,
        eps: float = 1e-5,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.num_parallel = num_parallel
        self.eps = eps
        self.use_linear = num_parallel == 1

        factory_kwargs = {"device": device, "dtype": dtype}
        param_dim = self.normalized_shape if self.use_linear else (num_parallel, 1, *self.normalized_shape)
        self.weight = nn.Parameter(torch.empty(param_dim, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(param_dim, **factory_kwargs))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, eps=self.eps) * self.weight + self.bias

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, num_parallel={self.num_parallel}"
