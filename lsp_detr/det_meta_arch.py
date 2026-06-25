from lsp_detr.lightning_module import LSPDetrModule
from lsp_detr.metrics.collections import NestedMetricCollection


class DETMetaArch(LSPDetrModule):
    """Hydra-compatible entry point for the LSP-DETR Lightning module."""


__all__ = ["DETMetaArch", "LSPDetrModule", "NestedMetricCollection"]
