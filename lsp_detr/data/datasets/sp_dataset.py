import albumentations as A
import numpy as np
import torch
from albumentations.core.composition import TransformsSeqType
from albumentations.pytorch import ToTensorV2
from stardist import star_distances
from torch import Tensor

from lsp_detr.data.datasets.types import SegmentationData
from lsp_detr.misc import masks2centroids


class SPDataset(torch.utils.data.Dataset[tuple[Tensor, dict[str, Tensor]]]):
    def __init__(
        self,
        data: SegmentationData,
        transforms: TransformsSeqType | None = None,
        n_rays: int = 64,
    ) -> None:
        self.data = data
        self.transforms = A.Compose(transforms or [])
        self.n_rays = n_rays
        self._to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        sample = self.data[idx]
        image = sample["image"]
        masks = sample["instances"]
        labels = np.zeros(masks.shape[-1], dtype=np.uint8)

        # Apply transforms
        transformed = self.transforms(image=image, mask=masks)
        image = transformed["image"]
        masks = transformed["mask"].transpose(2, 0, 1)

        # Drop empty masks after augmentations
        keep = masks.any(axis=(1, 2))
        masks = masks[keep]
        labels = labels[keep]

        lower_bound, _ = star_distances(masks, self.n_rays)
        radial_distances = lower_bound

        image = self._to_tensor(image=image)["image"]
        masks = torch.from_numpy(masks)

        return image, {
            "masks": masks,
            "labels": torch.from_numpy(labels).long(),
            "radial_distances": torch.from_numpy(radial_distances),
            "centroids": masks2centroids(masks, normalize=True),
        }
