from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

import torch
from torch import Tensor
from torch_linear_assignment import assignment_to_indices, batch_linear_assignment


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _compiler_disable(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    return cast("Callable[_P, _R]", torch.compiler.disable(fn))


@_compiler_disable
def linear_assignment(cost: Tensor) -> tuple[Tensor, Tensor]:
    if cost.device.type == "mps":
        # MPS does not support batch_linear_assignment, so we need to use a workaround
        # by moving the cost matrix to CPU and then back to MPS.
        row_ind, col_ind = assignment_to_indices(
            batch_linear_assignment(cost[None].cpu())
        )
        return row_ind[0].to(cost.device), col_ind[0].to(cost.device)

    row_ind, col_ind = assignment_to_indices(batch_linear_assignment(cost[None]))
    return row_ind[0], col_ind[0]
