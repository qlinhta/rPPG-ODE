from typing import List, Optional
import torch.nn as nn
import stribor as st
from torch import Tensor


class CouplingFlow(nn.Module):
    def __init__(self, dim: int, n_layers: int, hidden_dims: List[int], time_net: nn.Module,
                 time_hidden_dim: Optional[int] = None):
        super().__init__()
        transforms = [st.ContinuousAffineCoupling(
            latent_net=st.net.MLP(dim + 1, hidden_dims, 2 * dim),
            time_net=getattr(st.net, time_net)(2 * dim, hidden_dim=time_hidden_dim),
            mask='none' if dim == 1 else f'ordered_{i % 2}') for i in range(n_layers)]
        self.flow = st.Flow(transforms=transforms)

    def forward(self, x: Tensor, t: Tensor, t0: Optional[Tensor] = None) -> Tensor:
        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)
        if t0 is not None:
            x = self.flow.inverse(x, t=t0)[0]
        return self.flow(x, t=t)[0]


class ResNetFlow(nn.Module):
    def __init__(self, dim: int, n_layers: int, hidden_dims: List[int], time_net: str,
                 time_hidden_dim: Optional[int] = None, invertible: Optional[bool] = True):
        super().__init__()
        self.layers = nn.ModuleList([st.net.ResNetFlow(dim, hidden_dims, n_layers, activation='ReLU',
                                                       final_activation=None, time_net=time_net,
                                                       time_hidden_dim=time_hidden_dim, invertible=invertible) for _ in
                                     range(n_layers)])

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)
        for layer in self.layers:
            x = layer(x, t)
        return x
