from typing import Callable, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.inits import uniform


def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch_size: Tensor,
    num_node,
):
    batch_node_num = torch.full((batch_size,), num_node)
    cum_num_nodes = torch.cat(
        [batch_node_num.new_zeros(1),
         batch_node_num.cumsum(dim=0)[:-1]], dim=0).cuda()
    num_nodes = torch.tensor(num_node)
    dense_x = x.view(-1, num_nodes)

    _, perm0 = dense_x.sort(dim=-1, descending=True)
    perm = perm0 + cum_num_nodes.view(-1, 1)

    perm = perm.view(-1)
    perm_per = perm0.view(-1)

    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)

    index = torch.cat([
        torch.arange(k, dtype=torch.long, device=x.device) + i * num_nodes for i in range(batch_size)
    ], dim=0)

    perm = perm[index]
    perm_per = perm_per[index]

    return perm, perm_per


class TopKPooling(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        multiplier: float = 1.,
        nonlinearity: Union[str, Callable] = 'tanh',
    ):
        super().__init__()

        if isinstance(nonlinearity, str):
            nonlinearity = getattr(torch, nonlinearity)

        self.in_channels = in_channels
        self.ratio = ratio
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        uniform(self.in_channels, self.weight)

    def forward(self, x: Tensor, posi):

        b, n, f = x.shape
        x = x.reshape(b*n, f)

        position = posi.reshape(b*n, 3)
        score = (x * self.weight).sum(dim=-1)
        score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))

        perm, perm_FC = topk(score, self.ratio, b, n)
        perm = perm.type(torch.long)

        x = x[perm] * score[perm].view(-1, 1)
        position = position[perm]
        x = self.multiplier * x if self.multiplier != 1 else x
        x = x.reshape(b, -1, x.shape[1])

        position = position.reshape(b, -1, 3)

        return x, position