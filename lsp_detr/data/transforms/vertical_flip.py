from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray


if TYPE_CHECKING:

    class _VerticalFlipBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

        def apply_to_mask(
            self, mask: NDArray[Any], *args: Any, **params: Any
        ) -> NDArray[Any]: ...

else:
    from albumentations.augmentations.geometric.flip import (
        VerticalFlip as _VerticalFlipBase,
    )


class VerticalFlip(_VerticalFlipBase):
    def apply_to_mask(
        self, mask: NDArray[Any], *args: Any, **params: Any
    ) -> NDArray[Any]:
        if mask.size == 0:
            return mask
        return super().apply_to_mask(mask, *args, **params)
