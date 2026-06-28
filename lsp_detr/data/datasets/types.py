from typing import Any, Protocol


class SegmentationData(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict[str, Any]: ...
