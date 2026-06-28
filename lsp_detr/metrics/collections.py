from torchmetrics import MetricCollection

from lsp_detr.metrics.f1_score import F1Score
from lsp_detr.metrics.panoptic_quality import PanopticQuality


def build_validation_metrics() -> MetricCollection:
    return MetricCollection({"pq": PanopticQuality()})


def build_test_metrics() -> MetricCollection:
    return MetricCollection(
        {"F1": F1Score(radius=12), "pq": PanopticQuality()},
        compute_groups=False,
    )
