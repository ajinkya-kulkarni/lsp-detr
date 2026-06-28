import torch
from torch import Tensor
from torchmetrics import Metric

from lsp_detr.misc import linear_assignment


def panoptic_quality(
    preds: Tensor,
    targets: Tensor,
    iou_threshold: float = 0.5,
    eps: float = 1e-6,
) -> tuple[float, float, float]:
    preds = preds.flatten(1).float()
    targets = targets.flatten(1).float()

    intersection = targets @ preds.T
    union = targets.sum(dim=1, keepdim=True) + preds.sum(dim=1) - intersection
    iou = intersection / union

    row_ind, col_ind = linear_assignment(-iou)
    iou = iou[row_ind, col_ind]

    iou = iou[iou > iou_threshold]
    tp = iou.numel()

    dq = tp / (0.5 * preds.size(0) + 0.5 * targets.size(0) + eps)
    sq = iou.sum() / (tp + eps)
    pq = dq * sq

    return float(dq), float(sq), float(pq)


class PanopticQuality(Metric):
    def __init__(self, iou_threshold: float = 0.5) -> None:
        super().__init__()
        self.iou_threshold = iou_threshold
        self.dq: Tensor
        self.sq: Tensor
        self.pq: Tensor
        self.count: Tensor
        self.add_state("dq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.numel() == 0 and targets.numel() == 0:
            return

        dq, sq, pq = panoptic_quality(preds, targets, self.iou_threshold)

        self.dq += dq
        self.sq += sq
        self.pq += pq
        self.count += 1

    def compute(self) -> dict[str, Tensor]:
        if self.count == 0:
            zero = torch.zeros_like(self.pq)
            return {"dq": zero, "sq": zero, "pq": zero}

        return {
            "dq": self.dq / self.count,
            "sq": self.sq / self.count,
            "pq": self.pq / self.count,
        }
