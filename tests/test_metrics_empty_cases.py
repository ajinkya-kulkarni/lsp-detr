import torch

from lsp_detr.metrics.f1_score import F1Score
from lsp_detr.metrics.panoptic_quality import PanopticQuality


def test_panoptic_quality_handles_all_empty_batches() -> None:
    metric = PanopticQuality()

    metric.update(
        torch.empty((0, 4, 4), dtype=torch.bool),
        torch.empty((0, 4, 4), dtype=torch.bool),
    )

    assert metric.compute()["pq"].item() == 0


def test_panoptic_quality_penalizes_false_positives_without_targets() -> None:
    metric = PanopticQuality()

    metric.update(
        torch.ones((1, 4, 4), dtype=torch.bool),
        torch.empty((0, 4, 4), dtype=torch.bool),
    )

    assert metric.compute()["pq"].item() == 0


def test_f1_score_handles_empty_predictions_or_targets() -> None:
    metric = F1Score()

    metric.update(
        torch.empty((0, 4, 4), dtype=torch.bool),
        torch.ones((1, 4, 4), dtype=torch.bool),
    )
    metric.update(
        torch.ones((1, 4, 4), dtype=torch.bool),
        torch.empty((0, 4, 4), dtype=torch.bool),
    )

    result = metric.compute()
    assert result["Precision"].item() == 0
    assert result["Recall"].item() == 0
    assert result["F1"].item() == 0
