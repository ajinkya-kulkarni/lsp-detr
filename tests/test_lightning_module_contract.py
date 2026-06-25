from typing import cast

from torchmetrics import MetricCollection

from lsp_detr.det_meta_arch import DETMetaArch, NestedMetricCollection
from lsp_detr.lightning_module import LSPDetrModule
from lsp_detr.metrics.collections import (
    build_binary_test_metrics,
    build_category_test_metrics,
    build_multiclass_test_metrics,
    build_validation_metrics,
)


def _metric_names(metrics: NestedMetricCollection) -> set[str]:
    return {str(key) for key in cast("MetricCollection", metrics.metric)}


def test_det_meta_arch_remains_hydra_compatibility_target() -> None:
    assert DETMetaArch.__module__ == "lsp_detr.det_meta_arch"
    assert issubclass(DETMetaArch, LSPDetrModule)


def test_validation_metric_builder_preserves_metric_names() -> None:
    assert _metric_names(build_validation_metrics()) == {"bPQ"}


def test_binary_test_metric_builder_preserves_metric_names() -> None:
    assert _metric_names(build_binary_test_metrics()) == {
        "AJI",
        "AP_0,5",
        "AP_0,7",
        "AP_0,9",
        "AP_0,5:0,05:0,9",
        "F1",
        "bPQ",
        "bMPQ",
    }


def test_multiclass_test_metric_builder_preserves_metric_names() -> None:
    assert _metric_names(build_multiclass_test_metrics(num_classes=2)) == {
        "mPQ",
        "mMPQ",
    }


def test_category_test_metric_builder_preserves_metric_names() -> None:
    assert _metric_names(build_category_test_metrics()) == {
        "F1",
        "PQ",
        "MPQ",
    }
