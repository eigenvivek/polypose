import torch
from diffdrr.drr import DRR
from jaxtyping import Float

from .renderer import DeformableRenderer


class PolyPose(DeformableRenderer):
    def __init__(
        self,
        drr: DRR,
        weights: Float[torch.Tensor, "K D H W"],
        poses_rot: Float[torch.Tensor, "K 3"] = None,
        poses_xyz: Float[torch.Tensor, "K 3"] = None,
    ):
        super().__init__(
            drr,
            warp="polyrigid",
            weights=weights,
            poses_rot=poses_rot,
            poses_xyz=poses_xyz,
        )

    @property
    def poses_rot(self):
        return self.warp.poses_rot

    @property
    def poses_xyz(self):
        return self.warp.poses_xyz


class DenseTranslationField(DeformableRenderer):
    def __init__(
        self,
        drr: DRR,
        displacements: Float[torch.Tensor, "B D H W 3"] = None,
    ):
        super().__init__(drr, warp="nonrigid", displacements=displacements)


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
