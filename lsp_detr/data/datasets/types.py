from collections.abc import Sequence
from typing import Any, Protocol


class SegmentationData(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict[str, Any]: ...


class TissueSegmentationData(SegmentationData, Protocol):
    @property
    def tissue_names(self) -> Sequence[str]: ...


class TestSegmentationData(TissueSegmentationData, Protocol):
    @property
    def category_names(self) -> Sequence[str]: ...
