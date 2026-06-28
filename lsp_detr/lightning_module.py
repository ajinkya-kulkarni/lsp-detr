import re
from typing import TYPE_CHECKING, Any, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import Tensor

from lsp_detr.configuration import LSPDetrConfig
from lsp_detr.image_processing import LSPDetrImageProcessor
from lsp_detr.metrics.collections import (
    build_test_metrics,
    build_validation_metrics,
)
from lsp_detr.modeling import SetCriterion
from lsp_detr.modeling.lsp_detr import LSPDetrModel


if TYPE_CHECKING:
    from collections.abc import Mapping

    class _LSPDetrModuleBase(LSPDetrModel):
        trainer: Any
        logger: Any
        current_epoch: int

        def log(self, name: str, value: Any, *args: Any, **kwargs: Any) -> None: ...

        def log_dict(
            self, dictionary: Mapping[str, Any], *args: Any, **kwargs: Any
        ) -> None: ...

else:

    class _LSPDetrModuleBase(LSPDetrModel, LightningModule):
        pass


class LSPDetrModule(_LSPDetrModuleBase):
    def __init__(
        self,
        warmup_epochs: int,
        num_classes: int,
        criterion: SetCriterion,
        **config: Any,
    ) -> None:
        super().__init__(LSPDetrConfig(**config, num_classes=num_classes))
        self.criterion = criterion
        self.warmup_epochs = warmup_epochs

        self.backbone.train()  # to enable the drop path
        self.processor = LSPDetrImageProcessor()

        self.val_metrics = build_validation_metrics()
        self.test_metrics = build_test_metrics()
        self._aux_pattern = re.compile(r"_\d+$")

    def training_step(self, batch: tuple[Tensor, Any]) -> Tensor:
        inputs, targets = batch
        outputs = self(inputs)

        losses = self.criterion(outputs, targets)

        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            batch_size=len(inputs),
            on_step=True,
            prog_bar=True,
        )

        losses = {
            key: loss * self.criterion.weight_dict[name]
            for key, loss in losses.items()
            if (name := self._aux_pattern.sub("", key)) in self.criterion.weight_dict
        }

        return sum(losses.values(), torch.zeros((), device=inputs.device))

    def validation_step(self, batch: tuple[Tensor, Any]) -> None:
        inputs, targets = batch
        outputs = self(inputs)

        losses = self.criterion(outputs, targets)
        self.log_dict(
            {f"validation/{k}": v for k, v in losses.items()},
            batch_size=len(inputs),
            on_epoch=True,
            prog_bar=True,
        )

        results = self.processor.post_process(cast("dict[str, Tensor]", outputs))
        height = inputs.shape[-2]
        width = inputs.shape[-1]
        results = self.processor.post_process_instance(
            results, height=height, width=width
        )

        for b, result in enumerate(results):
            self.val_metrics.update(result["masks"], targets[b]["masks"])

    def on_validation_epoch_end(self) -> None:
        self.log("validation/pq", self.val_metrics.compute()["pq"])
        self._print_metrics("validation")
        self.val_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self._print_metrics("train")

    def _print_metrics(self, phase: str) -> None:
        prefix = f"{phase}/"
        items: list[str] = []
        for key in sorted(self.trainer.callback_metrics):
            if key.startswith(prefix):
                name = key[len(prefix) :]
                value = self.trainer.callback_metrics[key]
                display_value = value.item() if isinstance(value, Tensor) else value
                items.append(f"{name}={float(display_value):.6f}")
        if items:
            print(f"Epoch {self.current_epoch} - {phase}: {' '.join(items)}")

    def test_step(self, batch: tuple[Tensor, list[dict[str, Any]]]) -> None:
        inputs, targets = batch
        outputs = self(inputs)

        results = self.processor.post_process(cast("dict[str, Tensor]", outputs))
        height = inputs.shape[-2]
        width = inputs.shape[-1]
        results = self.processor.post_process_instance(
            results, height=height, width=width
        )
        for b, result in enumerate(results):
            self.test_metrics.update(result["masks"], targets[b]["masks"])

    def on_test_epoch_end(self) -> None:
        logger = cast("Any", self.logger)
        logger.log_table(
            self.test_metrics.compute(),
            "test_metrics.json",
        )
        self.test_metrics.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-4 / self.trainer.accumulate_grad_batches,
            weight_decay=1e-4,
        )

        max_epochs = self.trainer.max_epochs
        if max_epochs is None:
            raise ValueError("Trainer max_epochs must be set for CosineLRScheduler.")

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=max_epochs,
            lr_min=1e-6 / self.trainer.accumulate_grad_batches,
            warmup_lr_init=1e-7 / self.trainer.accumulate_grad_batches,
            warmup_t=self.warmup_epochs,
        )
        return cast("OptimizerLRScheduler", ([optimizer], [scheduler]))

    def lr_scheduler_step(self, scheduler: Any, metric: Any | None) -> None:
        cast("CosineLRScheduler", scheduler).step(epoch=self.current_epoch)
