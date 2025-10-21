import torch
import torch.nn.functional as F
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, convert
from jaxtyping import Float

from .warp import Warp


class PolyRigid(Warp):
    """Compute a polyrigid warp and apply it to a volume and segmentation mask."""

    def __init__(
        self,
        drr: DRR,  # DRR containing a subject (volume and mask) to warp
        weights: Float[torch.Tensor, "K D H W"],  # Weights for the polyrigid warp
        poses_rot: Float[torch.Tensor, "K 3"] = None,  # Rotation parameters
        poses_xyz: Float[torch.Tensor, "K 3"] = None,  # Translation parameters
    ):
        super().__init__(drr)

        # Interpolate the weights to match the shape of the volume
        self.register_buffer("weights", weights)
        self.weights = F.interpolate(
            self.weights[None],
            (self.D, self.H, self.W),
            mode="trilinear",
            align_corners=False,
        )[0]
        self.K, *_ = self.weights.shape

        # Initialize the log transforms for the articulated structures in the volume
        self.poses_rot = torch.nn.Parameter(poses_rot if poses_rot is not None else torch.randn(self.K, 3) * 1e-8)
        self.poses_xyz = torch.nn.Parameter(poses_xyz if poses_xyz is not None else torch.randn(self.K, 3) * 1e-8)

    @property
    def pose(self) -> RigidTransform:
        """Compute the average log transform at every point in space and map to the manifold."""
        poses = torch.concat([self.poses_rot, self.poses_xyz], dim=-1)
        logs = torch.einsum("cdhw,cn->dhwn", self.weights, poses).reshape(-1, 6)
        pose = convert(*logs.split([3, 3], dim=1), parameterization="se3_log_map")
        return self.drr.affine.compose(pose).compose(self.drr.affine_inverse)

    def warp(self):
        """Sample the displacement field at the identity points."""
        x = self.pts.reshape(-1, 1, 3)
        x = self.pose(x)
        return x.reshape(1, self.D, self.H, self.W, 3)
