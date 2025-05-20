import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    SwiGLU feed-forward module.

    This implements the SwiGLU variant of the GLU activation function
    with a pair of projection matrices and SiLU (Swish) activation.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int):
        """
        Initialize the SwiGLU module.

        Args:
            hidden_dim (int): Input and output dimension
            ffn_dim (int): Intermediate dimension for the feed-forward network
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim

        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SwiGLU module.

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, hidden_dim]
        """
        # Gate and up projections
        gate = self.w_gate(x)  # [batch_size, seq_len, ffn_dim]
        up = self.w_up(x)  # [batch_size, seq_len, ffn_dim]

        # Applying SiLU (Swish) activation to the gate
        activated_gate = F.silu(gate)  # [batch_size, seq_len, ffn_dim]

        # Element-wise multiplication
        intermediate = activated_gate * up  # [batch_size, seq_len, ffn_dim]

        # Down projection
        output = self.w_down(intermediate)  # [batch_size, seq_len, hidden_dim]

        return output
