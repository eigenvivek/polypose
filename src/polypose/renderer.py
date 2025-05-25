import torch
from diffdrr.drr import DRR
from torchio import LabelMap, ScalarImage

from .warp import NonRigid, PolyRigid, SE3Field


class DeformableRenderer(torch.nn.Module):
    """
    Render a DRR from a volume warped with a displacement field.
    """

    def __init__(
        self,
        drr: DRR,  # DRR containing a subject (volume and mask) to warp
        warp: str,  # Type of warp to use (polyrigid, nonrigid, se3)
        **kwargs,  # Additional arguments for the warp
    ):
        super().__init__()
        self.drr = drr
        if warp == "polyrigid":
            self.warp = PolyRigid(drr, **kwargs)
        elif warp == "nonrigid":
            self.warp = NonRigid(drr, **kwargs)
        elif warp == "se3":
            self.warp = SE3Field(drr, **kwargs)
        else:
            raise ValueError(f"Invalid warp: {warp}")

    def forward(self, pose, **kwargs):
        """Render a DRR from the warped density and mask."""
        warped_density, warped_mask = self.warp()
        source, target = self.drr.detector(pose, calibration=None)
        img = self.render(warped_density, warped_mask, source, target, **kwargs)
        img = self.drr.reshape_transform(img, batch_size=len(pose))
        return img

    def render(self, density, mask, source, target, **kwargs):
        img = (target - source).norm(dim=-1).unsqueeze(1)
        source = self.drr.affine_inverse(source)
        target = self.drr.affine_inverse(target)
        img = self.drr.renderer(density, source, target, img, mask=mask, **kwargs)
        return img

    @torch.no_grad
    def warp_subject(self, affine=None, volume_dtype=torch.float32, mask_dtype=torch.uint8):
        """Warp the original volume (HU) and segmentation mask."""
        if affine is None:
            affine = self.drr.subject.volume.affine
        warped_volume, warped_mask = self.warp.warp_subject()
        warped_volume = ScalarImage(tensor=warped_volume[None].cpu().to(volume_dtype), affine=affine)
        warped_mask = LabelMap(tensor=warped_mask[None].cpu().to(mask_dtype), affine=affine)
        return warped_volume, warped_mask
