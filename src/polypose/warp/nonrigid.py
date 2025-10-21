import torch
import torch.nn.functional as F
from diffdrr.drr import DRR
from jaxtyping import Float

from ..loss import divergence as _divergence
from .warp import Warp


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
