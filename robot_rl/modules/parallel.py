import torch
import torch.nn as nn


class ParallelLinear(nn.Module):
    """Module that implements Linear across parallel networks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_parallel: int,
        bias: bool = True,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the module."""
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_parallel = num_parallel

        weight_dim = (num_parallel, in_features, out_features)
        bias_dim = (num_parallel, 1, out_features)

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(weight_dim, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(bias_dim, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear layer."""
        return torch.baddbmm(self.bias, x, self.weight)

    def extra_repr(self) -> str:
        """Return the extra representation of the module."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, num_parallel={self.num_parallel}, "
            f"bias={self.bias is not None}"
        )


class ParallelLayerNorm(nn.Module):
    """Module that implements LayerNorm across parallel networks."""

    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        num_parallel: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the module."""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.num_parallel = num_parallel
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            param_dim = (num_parallel, 1, *self.normalized_shape)
            self.weight = nn.Parameter(torch.empty(param_dim, **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(param_dim, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters, if applicable."""
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization."""
        out = nn.functional.layer_norm(x, self.normalized_shape, eps=self.eps)
        if self.elementwise_affine:
            out = out * self.weight
            if self.bias is not None:
                out = out + self.bias
        else:
            out = out.reshape(self.num_parallel, 1, *self.normalized_shape)
        return out

    def extra_repr(self) -> str:
        """Return the extra representation of the module."""
        return (
            f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, "
            f"num_parallel={self.num_parallel}"
        )
