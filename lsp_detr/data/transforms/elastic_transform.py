from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray


if TYPE_CHECKING:

    class _ElasticTransformBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

        def apply_to_mask(
            self,
            mask: NDArray[Any],
            map_x: NDArray[Any],
            map_y: NDArray[Any],
            **params: Any,
        ) -> NDArray[Any]: ...

else:
    from albumentations.augmentations.geometric.distortion import (
        ElasticTransform as _ElasticTransformBase,
    )


class ElasticTransform(_ElasticTransformBase):
    def apply_to_mask(
        self,
        mask: NDArray[Any],
        map_x: NDArray[Any],
        map_y: NDArray[Any],
        **params: Any,
    ) -> NDArray[Any]:
        if mask.size == 0:
            return mask
        return super().apply_to_mask(mask, map_x, map_y, **params)
