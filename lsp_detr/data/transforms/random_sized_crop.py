from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray


if TYPE_CHECKING:

    class _RandomSizedCropBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

        def apply_to_mask(
            self,
            mask: NDArray[Any],
            crop_coords: tuple[int, int, int, int],
            **params: Any,
        ) -> NDArray[Any]: ...

else:
    from albumentations.augmentations.crops.transforms import (
        RandomSizedCrop as _RandomSizedCropBase,
    )


class RandomSizedCrop(_RandomSizedCropBase):
    def apply_to_mask(
        self,
        mask: NDArray[Any],
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> NDArray[Any]:
        if mask.size == 0:
            return mask
        return super().apply_to_mask(mask, crop_coords, **params)
