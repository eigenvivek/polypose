import torch
import torch.nn.functional as F
from diffdrr.drr import DRR

from ..loss import jacdet as _jacdet


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
