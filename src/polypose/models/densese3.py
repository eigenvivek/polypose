import torch
from diffdrr.drr import DRR
from jaxtyping import Float

from .renderer import DeformableRenderer


class DenseSE3Field(DeformableRenderer):
    def __init__(
        self,
        drr: DRR,
        se3_rot: Float[torch.Tensor, "N 3"] = None,
        se3_xyz: Float[torch.Tensor, "N 3"] = None,
    ):
        super().__init__(drr, warp="se3", se3_rot=se3_rot, se3_xyz=se3_xyz)

    @property
    def se3_rot(self):
        return self.warp.se3_rot

    @property
    def se3_xyz(self):
        return self.warp.se3_xyz
