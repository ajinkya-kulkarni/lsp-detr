from collections import defaultdict
from copy import deepcopy
from typing import Any, cast

import torch
from torch.nn import ModuleDict
from torchmetrics import Metric, MetricCollection

from lsp_detr.metrics.aji import AggregatedJaccardIndex
from lsp_detr.metrics.ap_2018_dsb import AveragePrecision2018DSB
from lsp_detr.metrics.binary_panoptic_quality import BinaryPanopticQuality
from lsp_detr.metrics.f1_score import F1Score
from lsp_detr.metrics.panoptic_quality import PanopticQuality


type MetricTemplate = Metric | MetricCollection


class NestedMetricCollection(ModuleDict):
    def __init__(self, metric: MetricTemplate) -> None:
        super().__init__()
        self.metric: MetricTemplate = metric

    def update(self, *args: Any, key: str | None = None, **kwargs: Any) -> None:
        if key is None:
            super().update(*args, **kwargs)
            return

        if key not in self:
            self.add_module(key, deepcopy(self.metric))

        cast("MetricTemplate", self[key]).update(*args, **kwargs)

    def compute(self) -> dict[str, Any]:
        divided_metrics = {
            k: cast("MetricTemplate", v).compute()
            for k, v in self.items()
            if k != "metric"
        }

        out = defaultdict(list)
        for key, metrics in divided_metrics.items():
            out["key"].append(key)
            for subkey, value in metrics.items():
                out[subkey].append(value.item())

        return out

    def reset(self) -> None:
        for metric in self.values():
            cast("MetricTemplate", metric).reset()


def _binary_instance_metrics() -> dict[str, MetricTemplate]:
    return {
        "AJI": AggregatedJaccardIndex(),
        "AP_0,5": AveragePrecision2018DSB(iou_thresholds=[0.5]),
        "AP_0,7": AveragePrecision2018DSB(iou_thresholds=[0.7]),
        "AP_0,9": AveragePrecision2018DSB(iou_thresholds=[0.9]),
        "AP_0,5:0,05:0,9": AveragePrecision2018DSB(
            iou_thresholds=torch.arange(0.5, 1, 0.05)
        ),
        "F1": F1Score(radius=12),
        "bPQ": BinaryPanopticQuality(),
        "bMPQ": BinaryPanopticQuality(masked=True),
    }


def build_validation_metrics() -> NestedMetricCollection:
    return NestedMetricCollection(MetricCollection({"bPQ": BinaryPanopticQuality()}))


def build_binary_test_metrics() -> NestedMetricCollection:
    return NestedMetricCollection(
        MetricCollection(_binary_instance_metrics(), compute_groups=False)
    )


def build_multiclass_test_metrics(num_classes: int) -> NestedMetricCollection:
    return NestedMetricCollection(
        MetricCollection(
            {
                "mPQ": PanopticQuality(num_classes=num_classes),
                "mMPQ": PanopticQuality(num_classes=num_classes, masked=True),
            },
            compute_groups=False,
        )
    )


def build_category_test_metrics() -> NestedMetricCollection:
    return NestedMetricCollection(
        MetricCollection(
            {
                "F1": F1Score(radius=12),
                "PQ": BinaryPanopticQuality(),
                "MPQ": BinaryPanopticQuality(masked=True),
            },
            compute_groups=False,
        )
    )
