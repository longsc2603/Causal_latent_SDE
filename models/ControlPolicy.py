import math
import torch
from torch import nn


class ControlPolicy(nn.Module):
    """Neural control policy :math:`u_\phi(t, z, c)` used by LACD.

    The policy receives the current latent state ``z`` (shape ``[batch, z_dim]``), a scalar
    time ``t`` (broadcast across the batch) and an optional context vector ``c`` that can
    encode observation-specific information. It outputs a control vector with the same
    dimensionality as ``z`` which will be scaled by the diffusion term before being applied
    to the drift.
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int = 64,
        context_dim: int = 0,
        num_hidden_layers: int = 2,
        activation: str = "elu",
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        self.z_dim = z_dim
        self.context_dim = context_dim
        self.time_embed_dim = 32

        in_dim = z_dim + self.time_embed_dim + max(context_dim, 0)
        hidden_sizes = [hidden_dim] * num_hidden_layers

        layers = []
        prev_dim = in_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size, bias=bias, device=device))
            if activation == "elu":
                layers.append(nn.ELU(inplace=True))
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            prev_dim = hidden_size
        layers.append(nn.Linear(prev_dim, z_dim, bias=bias, device=device))

        self.network = nn.Sequential(*layers)

        # Initialize weights with a small gain to keep early controls near zero.
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _get_time_embedding(self, t):
        # Sinusoidal embedding
        half_dim = self.time_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return emb

    def forward(self, t, z, context=None):
        if z.dim() == 1:
            z = z.unsqueeze(0)

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=z.dtype, device=z.device)
        if t.dim() == 0:
            t = t.repeat(z.size(0))
        if t.dim() == 1 and t.shape[0] == 1 and z.size(0) > 1:
            t = t.repeat(z.size(0))
        if t.dim() != 1 or t.shape[0] != z.shape[0]:
            raise ValueError("Time input must broadcast to batch dimension of z.")
        
        t_feat = self._get_time_embedding(t)

        features = [z, t_feat]

        if context is not None:
            if not isinstance(context, torch.Tensor):
                context = torch.tensor(context, dtype=z.dtype, device=z.device)
            if context.dim() == 1:
                context = context.unsqueeze(0)
            if context.shape[0] == 1 and z.size(0) > 1:
                context = context.expand(z.size(0), -1)
            if context.shape[0] != z.shape[0]:
                raise ValueError("Context tensor must match batch size of z after broadcasting.")
            features.append(context)

        stacked = torch.cat(features, dim=1)
        u = self.network(stacked)
        return u
