import torch
from torch import nn
from typing import Optional

class Func(nn.Module):
    def __init__(
        self,
        learnable_scale: bool = False,
        init_scale: float = 1.0,
        eps: float = 1e-12,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.name = name or self.__class__.__name__
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(init_scale)), persistent=True)

    @staticmethod
    def _stable_softplus(x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.exp(-torch.abs(x))) + torch.clamp(x, min=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(device=x.device, dtype=x.dtype)
        pos = self._stable_softplus(x)
        neg = self._stable_softplus(-x)
        numerator = pos - neg
        denominator = pos + neg + self.eps
        out = numerator / denominator
        out = out * scale
        return out
    def extra_repr(self) -> str:
        learnable = isinstance(self.scale, nn.Parameter)
        return (
            f"name={self.name!r}, "
            f"learnable_scale={learnable}, "
            f"init_scale={float(self.scale.data):.4f}, "
            f"eps={self.eps}"
        )
