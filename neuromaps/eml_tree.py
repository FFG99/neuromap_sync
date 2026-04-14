"""Дифференцируемое EML-дерево (reduction по листьям)."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def eml_stable(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a = torch.clamp(a, -40.0, 40.0)
    br = F.softplus(b) + eps
    return torch.exp(a) - torch.log(br)


class TrainableEMLReductionTree(nn.Module):
    """
    Листья: softmax по терминалам [1, u…, p…]. Узлы: eml_stable(left, right).
    """

    def __init__(
        self,
        n_terminals: int,
        n_leaves: int = 8,
        *,
        eps: float = 1e-6,
    ):
        super().__init__()
        if n_leaves < 2 or (n_leaves & (n_leaves - 1)) != 0:
            raise ValueError("n_leaves must be a power of 2 and >= 2")
        self.n_terminals = n_terminals
        self.n_leaves = n_leaves
        self.eps = eps
        self.leaf_logits = nn.Parameter(torch.zeros(n_leaves, n_terminals))

    def forward(self, terminals: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.leaf_logits, dim=-1)
        cur: List[torch.Tensor] = [
            (w[i : i + 1] * terminals).sum(dim=-1, keepdim=True) for i in range(self.n_leaves)
        ]
        while len(cur) > 1:
            nxt = []
            for i in range(0, len(cur), 2):
                nxt.append(eml_stable(cur[i], cur[i + 1], self.eps))
            cur = nxt
        return cur[0]
