from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from albumentations.core.composition import TransformsSeqType
from lightning import LightningDataModule
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lsp_detr.data.datasets import PredictDataset, SPDataset, TestingDataset
from lsp_detr.data.utils import collate_fn


class NuFuseDataset(Dataset[dict[str, Any]]):
    """Dataset for NuFuse .tif / .npy instance segmentation pairs.

    Each sample consists of:
        - ``{name}.tif``: 256x256 RGB image
        - ``{name}.npy``: 256x256 label map where each pixel value is an instance id
          (0 = background)
    """

    def __init__(self, root: str | Path, max_samples: int | None = None) -> None:
        self.root = Path(root)
        self.image_paths = sorted(self.root.glob("*.tif"))
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

        self.tissue_names = ["unknown"]
        self.category_names: list[str] = []

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path = self.image_paths[idx]
        mask_path = img_path.with_suffix(".npy")

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        label_map = np.load(mask_path)

        instance_ids = np.unique(label_map)
        instance_ids = instance_ids[instance_ids != 0]

        if len(instance_ids) == 0:
            masks = np.empty((image.shape[0], image.shape[1], 0), dtype=np.uint8)
        else:
            masks = np.stack(
                [(label_map == i).astype(np.uint8) for i in instance_ids], axis=-1
            )

        return {
            "image": image,
            "instances": masks,
            "categories": np.zeros(len(instance_ids), dtype=np.uint8),
            "tissue": 0,
        }


class NuFuse(LightningDataModule):
    name = "nufuse"

    def __init__(
        self,
        root: str,
        batch_size: int,
        num_radial_distances: int,
        max_samples: int | None = None,
        num_workers: int = 0,
        train_transforms: TransformsSeqType | None = None,
        eval_transforms: TransformsSeqType | None = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_radial_distances = num_radial_distances
        self.max_samples = max_samples
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms

    def setup(self, stage: str) -> None:
        data = NuFuseDataset(self.root, max_samples=self.max_samples)

        match stage:
            case "fit":
                self.train_dataset = SPDataset(
                    data, self.train_transforms, self.num_radial_distances
                )
                self.val_dataset = SPDataset(
                    data, self.eval_transforms, self.num_radial_distances
                )
            case "validate":
                self.val_dataset = SPDataset(
                    data, self.eval_transforms, self.num_radial_distances
                )
            case "test":
                self.test_dataset = TestingDataset(data, self.eval_transforms)
            case "predict":
                self.predict_dataset = PredictDataset(data, self.eval_transforms)

    def _dataloader(
        self, dataset: Dataset[Any], shuffle: bool = False
    ) -> DataLoader[tuple[Tensor, list[dict[str, Any]]]]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0,
            drop_last=shuffle,
            pin_memory=True,
        )

    def train_dataloader(self) -> Iterable[tuple[Tensor, list[dict[str, Any]]]]:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> Iterable[tuple[Tensor, list[dict[str, Any]]]]:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> Iterable[tuple[Tensor, list[dict[str, Any]]]]:
        return self._dataloader(self.test_dataset)

    def predict_dataloader(self) -> Iterable[tuple[Tensor, list[dict[str, Any]]]]:
        return self._dataloader(self.predict_dataset)
