from torchmetrics import MetricCollection

from lsp_detr.lightning_module import LSPDetrModule
from lsp_detr.metrics.collections import (
    build_test_metrics,
    build_validation_metrics,
)


def _metric_names(metrics: MetricCollection) -> set[str]:
    return {str(key) for key in metrics}


def test_lightning_module_is_training_entry_point() -> None:
    assert LSPDetrModule.__module__ == "lsp_detr.lightning_module"


def test_validation_metric_builder_preserves_metric_names() -> None:
    assert _metric_names(build_validation_metrics()) == {"pq"}


def test_test_metric_builder_preserves_metric_names() -> None:
    assert _metric_names(build_test_metrics()) == {
        "F1",
        "pq",
    }
