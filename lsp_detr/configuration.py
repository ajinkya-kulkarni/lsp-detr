from typing import TypedDict


class STAConfig(TypedDict):
    kernel: int
    q_tile: int
    kv_tile: int


class LSPDetrConfig:
    def __init__(
        self,
        backbone: str = "swinv2_tiny_window16_256",
        pretrained_backbone: bool = True,
        dim: int = 384,
        num_heads: int = 12,
        num_classes: int = 1,
        query_block_size: float = 14,
        feature_levels: tuple[int, ...] = (2, 1, 0, 2, 1, 0),
        num_radial_distances: int = 64,
        self_sta_config: STAConfig | None = None,
        cross_sta_config: tuple[STAConfig, ...] = (
            {"kernel": 5, "q_tile": 3, "kv_tile": 8},
            {"kernel": 5, "q_tile": 3, "kv_tile": 4},
            {"kernel": 5, "q_tile": 3, "kv_tile": 2},
        ),
    ) -> None:
        if self_sta_config is None:
            self_sta_config = {"kernel": 3, "q_tile": 3, "kv_tile": 3}

        self.backbone = backbone
        self.pretrained_backbone = pretrained_backbone
        self.dim = dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.query_block_size = query_block_size
        self.feature_levels = feature_levels
        self.num_radial_distances = num_radial_distances
        self.self_sta_config = self_sta_config
        self.cross_sta_config = cross_sta_config
