import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    https://arxiv.org/pdf/1910.07467, page 3
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))  # or scale

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        root_mean_square = torch.sqrt(
            torch.sum(x**2, dim=-1, keepdim=True) / (x.shape[-1] + self.eps)
        )
        return root_mean_square

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms_norm = (x / self._rms(x)) * self.gamma
        return rms_norm
