import torch
import torch.nn.functional as F
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, convert
from jaxtyping import Float

from .loss import divergence as _divergence
from .loss import elastic as _elastic
from .loss import jacdet as _jacdet


class Warp(torch.nn.Module):
    """Base class for all 3D deformation fields."""

    def __init__(self, drr: DRR):
        super().__init__()

        # Load the (possibly downsampled) volume and segmentation mask
        self.drr = drr
        self.density = self.drr.density.permute(2, 1, 0)[None, None]
        self.mask = self.drr.mask.permute(2, 1, 0)[None, None]
        *_, self.W, self.H, self.D = self.density.shape

        # Initialize identity points for sampling the displacement field
        X, Y, Z = torch.meshgrid(
            torch.arange(self.D),
            torch.arange(self.H),
            torch.arange(self.W),
            indexing="ij",
        )
        pts = torch.stack([X, Y, Z], dim=-1).to(torch.float32)
        self.register_buffer("pts", pts)
        self.register_buffer("shape", torch.tensor([self.D, self.H, self.W]))
        self.register_buffer("volume", self.drr.subject.volume.data[0].permute(2, 1, 0)[None, None])

    def normalize(self, x):
        return 2 * x / self.shape - 1

    def warp(self):
        raise NotImplementedError("Subclasses must implement this method")

    def forward(self):
        pts = self.normalize(self.warp())
        warped_density = self._warp_volume(self.density, pts)
        warped_mask = self._warp_mask(self.mask, pts)
        return warped_density, warped_mask

    def _warp_volume(self, volume, pts):
        dtype = volume.dtype
        if volume.dtype != pts.dtype:
            volume = volume.to(pts.dtype)
        return F.grid_sample(volume, pts, align_corners=False, mode="bilinear", padding_mode="border").squeeze().to(dtype)

    def _warp_mask(self, mask, pts):
        dtype = mask.dtype
        if mask.dtype != pts.dtype:
            mask = mask.to(pts.dtype)
        return F.grid_sample(mask, pts, align_corners=False, mode="nearest").squeeze().to(dtype)

    @torch.no_grad
    def warp_subject(self):
        pts = self.normalize(self.warp())
        warped_volume = self._warp_volume(self.volume, pts)
        warped_mask = self._warp_mask(self.mask, pts)
        return warped_volume, warped_mask

    @property
    def jacdet(self):
        """Compute the Jacobian determinant of the warp."""
        return _jacdet(-self.warp())


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


class NonRigid(Warp):
    """Compute a dense translation field and apply it to a volume and segmentation mask."""

    def __init__(
        self,
        drr: DRR,  # DRR containing a subject (volume and mask) to warp
        displacements: Float[torch.Tensor, "B D H W 3"] = None,  # Displacement field
    ):
        super().__init__(drr)
        if displacements is None:
            displacements = torch.zeros(1, self.D, self.H, self.W, 3)
        else:
            displacements = F.interpolate(
                displacements.permute(0, -1, 1, 2, 3),
                (self.D, self.H, self.W),
                mode="trilinear",
                align_corners=False,
            )
            displacements = displacements.permute(0, 2, 3, 4, 1)
        self.displacements = torch.nn.Parameter(displacements)

    def warp(self):
        """Sample the displacement from the identity points."""
        return self.pts + self.displacements

    @property
    def divergence(self):
        """Compute the divergence of the displacement field."""
        return _divergence(-self.warp())


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
