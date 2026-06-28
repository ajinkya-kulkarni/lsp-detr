import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor


class LSPDetrImageProcessor:
    def post_process(self, outputs: dict[str, Tensor]) -> list[dict[str, Tensor]]:
        """Converts the raw output into polygons.

        Returns:
            A list of dictionaries, each containing:
                - "polygons": A tensor of shape (N, num_radial_distances, 2) representing the polygons.
                - "labels": A tensor of shape (N,) representing the labels for each polygon.
        """
        radial_distances = outputs["radial_distances"].exp()

        t = torch.linspace(
            0, 1, radial_distances.size(-1) + 1, device=radial_distances.device
        )[:-1]
        cos = torch.cos(2 * torch.pi * t)
        sin = torch.sin(2 * torch.pi * t)

        polar = radial_distances.unsqueeze(-1) * torch.stack([sin, cos], dim=-1)
        polygons = outputs["absolute_points"].unsqueeze(-2) + polar

        labels = outputs["logits"].argmax(dim=-1)
        non_no_object_indices = labels != outputs["logits"].size(-1) - 1

        return [
            {"polygons": polygons[b, indices], "labels": labels[b, indices]}
            for b, indices in enumerate(non_no_object_indices)
        ]

    def post_process_instance(
        self,
        results: list[dict[str, Tensor]],
        height: int,
        width: int,
    ) -> list[dict[str, Tensor]]:
        """Converts the output into actual instance segmentation predictions.

        Args:
            results: Results list obtained by `post_process`, to which "masks" results will be added.
            height: Height of the input image.
            width: Width of the input image.
        """
        for i, result in enumerate(results):
            masks = torch.zeros(
                (len(result["polygons"]), height, width),
                dtype=torch.bool,
                device=result["polygons"].device,
            )

            for j, polygon in enumerate(result["polygons"]):
                img = Image.fromarray(masks[j].cpu().numpy())
                canvas = ImageDraw.Draw(img)
                canvas.polygon(xy=polygon.flatten().tolist(), outline=1, fill=1)
                masks[j] = torch.from_numpy(np.asarray(img).copy()).to(
                    device=masks.device, dtype=torch.bool
                )

            # Area-sort painter: resolve prediction overlaps (largest first)
            areas = masks.sum(dim=(1, 2))
            sorted_indices = torch.argsort(areas, descending=True)
            occupied = torch.zeros(
                (height, width), dtype=torch.bool, device=masks.device
            )
            for idx in sorted_indices:
                masks[idx] = masks[idx] & ~occupied
                occupied = occupied | masks[idx]

            results[i]["masks"] = masks

        return results
