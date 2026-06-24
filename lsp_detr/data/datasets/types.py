from collections.abc import Sequence
from typing import Any, Protocol


class SegmentationData(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ...


class TissueSegmentationData(SegmentationData, Protocol):
    tissue_names: Sequence[str]


class TestSegmentationData(TissueSegmentationData, Protocol):
    category_names: Sequence[str]
