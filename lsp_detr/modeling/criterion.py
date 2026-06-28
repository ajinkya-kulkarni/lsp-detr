from typing import TYPE_CHECKING, TypedDict, cast

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torchvision.ops import sigmoid_focal_loss


if TYPE_CHECKING:
    from collections.abc import Callable


def l1_log_space_loss(outputs: Tensor, tgt_bound: Tensor) -> Tensor:
    """L1 loss in log-space for radial distances."""
    return (outputs - tgt_bound.log()).abs().nanmean().nan_to_num(0.0)


_pad_sequence = cast("Callable[..., Tensor]", torch.compiler.disable(pad_sequence))


class Outputs(TypedDict):
    logits: Tensor
    radial_distances: Tensor
    points: Tensor


class ModelOutputs(Outputs):
    absolute_points: Tensor
    embeddings: Tensor
    aux_outputs: list[Outputs]


class Target(TypedDict):
    labels: Tensor
    masks: Tensor
    centroids: Tensor
    radial_distances: Tensor


class SetCriterion(nn.Module):
    def __init__(
        self, num_classes: int, matcher: nn.Module, weight_dict: dict[str, float]
    ) -> None:
        """Create the criterion.

        Args:
            num_classes: Number of object categories, omitting the special no-object category.
            matcher: Module able to compute a matching between targets and proposals.
            weight_dict: Dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

    def loss_labels(
        self,
        outputs: Outputs,
        targets: list[Target],
        indices: list[tuple[Tensor, Tensor]],
    ) -> dict[str, Tensor]:
        src_logits = outputs["logits"]
        idx = self._get_src_permutation_indices(indices)

        tgt_classes_o = torch.cat(
            [t["labels"][j] for t, (_, j) in zip(targets, indices, strict=True)]
        )

        tgt_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        tgt_classes[idx] = tgt_classes_o
        tgt_one_hot = F.one_hot(tgt_classes).type_as(src_logits)

        return {
            "loss_ce": sigmoid_focal_loss(src_logits, tgt_one_hot, reduction="mean")
        }

    def loss_descriptors(
        self,
        outputs: Outputs,
        targets: list[Target],
        indices: list[tuple[Tensor, Tensor]],
    ) -> dict[str, Tensor]:
        src_idx = self._get_src_permutation_indices(indices)
        tgt_idx = self._get_tgt_permutation_indices(indices)

        src_points = outputs["points"]
        src_radial_distances = outputs["radial_distances"]

        tgt_radial_distance_map = torch.stack([t["radial_distances"] for t in targets])
        tgt_centroids = _pad_sequence(
            [t["centroids"] for t in targets], batch_first=True
        )

        with torch.no_grad():
            grid = src_points * 2 - 1  # Normalize to [-1, 1]
            # Sample the target radial distances at the predicted reference points
            tgt_bound = F.grid_sample(
                tgt_radial_distance_map,
                grid.unsqueeze(1),  # (B, 1, N, 2)
                align_corners=False,
                padding_mode="zeros",  # border results in NaN with Infs
            )
            tgt_bound = rearrange(tgt_bound, "B R 1 N -> B N R")

        return {
            "loss_centroids": F.l1_loss(
                src_points[src_idx], tgt_centroids[tgt_idx].to(src_points)
            ).nan_to_num(0.0),  # if empty
            "loss_radial_distances": l1_log_space_loss(
                outputs=src_radial_distances[src_idx],
                tgt_bound=tgt_bound[src_idx].to(src_radial_distances),
            ),
        }

    def _get_src_permutation_indices(
        self, indices: list[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, Tensor]:
        # permute predictions following indices
        batch_indices = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, src_indices

    def _get_tgt_permutation_indices(
        self, indices: list[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, Tensor]:
        # permute targets following indices
        batch_indices = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    def forward(
        self, outputs: ModelOutputs, targets: list[Target]
    ) -> dict[str, Tensor]:
        """This performs the loss computation.

        Args:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute all the requested losses
        losses = {
            **self.loss_labels(outputs, targets, indices),
            **self.loss_descriptors(outputs, targets, indices),
        }

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        for i, aux_outputs in enumerate(outputs["aux_outputs"]):
            indices = self.matcher(aux_outputs, targets)
            loss_labels = self.loss_labels(aux_outputs, targets, indices)
            loss_descriptors = self.loss_descriptors(aux_outputs, targets, indices)
            losses.update(
                {f"{k}_{i}": v for k, v in (loss_labels | loss_descriptors).items()}
            )

        return losses
