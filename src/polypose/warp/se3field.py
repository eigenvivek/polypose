import torch
from diffdrr.drr import DRR
from diffdrr.pose import convert
from jaxtyping import Float

from ..loss import elastic as _elastic
from .warp import Warp


class SE3Field(Warp):
    """Compute a dense SE(3) field and apply it to a volume and segmentation mask."""

    def __init__(
        self,
        drr: DRR,  # DRR containing a subject (volume and mask) to warp
        se3_rot: Float[torch.Tensor, "N 3"] = None,  # Rotation parameters
        se3_xyz: Float[torch.Tensor, "N 3"] = None,  # Translation parameters
    ):
        super().__init__(drr)
        N = self.D * self.H * self.W
        self.se3_rot = torch.nn.Parameter(se3_rot if se3_rot is not None else torch.randn(N, 3) * 1e-8)
        self.se3_xyz = torch.nn.Parameter(se3_xyz if se3_xyz is not None else torch.randn(N, 3) * 1e-8)

    def warp(self):
        """Sample the displacement field at the identity points."""
        x = self.pts.reshape(-1, 1, 3)
        x = self.pose(x)
        return x.reshape(1, self.D, self.H, self.W, 3)

    @property
    def pose(self):
        pose = convert(self.se3_rot, self.se3_xyz, parameterization="se3_log_map")
        return self.drr.affine.compose(pose).compose(self.drr.affine_inverse)

    @property
    def elastic(self):
        return _elastic(-self.warp())
