import torch
from diffdrr.drr import DRR
from jaxtyping import Float

from .renderer import DeformableRenderer


class DenseTranslationField(DeformableRenderer):
    def __init__(
        self,
        drr: DRR,
        displacements: Float[torch.Tensor, "B D H W 3"] = None,
    ):
        super().__init__(drr, warp="nonrigid", displacements=displacements)
