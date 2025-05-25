from typing import List

import torch
from diffdrr.metrics import (
    GradientNormalizedCrossCorrelation2d,
    MultiscaleNormalizedCrossCorrelation2d,
)
from jaxtyping import Float


class ImageLoss(torch.nn.Module):
    """Initialize the image similarity metric"""

    def __init__(
        self,
        beta: float = 0.5,  # Mixing parameter between the two similarity metrics
        mncc_patch_size: int = 9,  # Patch size for Multiscale Normalized Cross Correlation
        mncc_weights: List[float] = [0.5, 0.5],  # Weights for the global and local scales
        gncc_patch_size: int = 11,  # Patch size for Gradient Normalized Cross Correlation
        gncc_sigma: float = 10,  # Sigma for Gradient Normalized Cross Correlation
    ):
        super().__init__()
        self.sim1 = MultiscaleNormalizedCrossCorrelation2d([None, mncc_patch_size], mncc_weights)
        self.sim2 = GradientNormalizedCrossCorrelation2d(patch_size=gncc_patch_size, sigma=gncc_sigma).cuda()
        self.beta = beta

    def imagesim(self, x, y):
        if self.beta == 0:
            return self.sim2(x, y)
        elif self.beta == 1:
            return self.sim1(x, y)
        else:
            return self.beta * self.sim1(x, y) + (1 - self.beta) * self.sim2(x, y)

    def forward(self, gt, img):
        img = img.sum(dim=1, keepdim=True)
        return self.imagesim(gt, img)


def jacobian(J: Float[torch.Tensor, "B D H W 3"]) -> Float[torch.Tensor, "B D H W 3 3"]:
    """
    Compute the Jacobian of the flow field with finite differences.
    """
    dy = J[:, 1:, :-1, :-1] - J[:, :-1, :-1, :-1]
    dx = J[:, :-1, 1:, :-1] - J[:, :-1, :-1, :-1]
    dz = J[:, :-1, :-1, 1:] - J[:, :-1, :-1, :-1]
    return dx, dy, dz


def jacdet(J: Float[torch.Tensor, "B D H W 3"]) -> Float[torch.Tensor, "B D H W"]:
    """
    Compute the Jacobian determinant of the flow field.

    The flow field (input points + displacement field) should be in units of voxels (i.e., already normalized to the volume size).

    Adapted from https://github.com/Kidrauh/neural-atlasing/blob/216f624aea3708589e60beee5285eb6781acb981/sinf/utils/util.py#L172-L184
    """
    dx, dy, dz = jacobian(J)
    Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
    Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
    Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
    Jdet = Jdet0 - Jdet1 + Jdet2
    return Jdet


def divergence(J: Float[torch.Tensor, "B D H W 3"]) -> Float[torch.Tensor, "B D H W 3"]:
    """
    Compute the divergence of the flow field.
    """
    dx, dy, dz = jacobian(J)
    return (dx + dy + dz).abs().square()


def elastic(J, eps=1e-6):
    dx, dy, dz = jacobian(J)
    jac = torch.stack([dx, dy, dz], dim=-1)
    _, sigma, _ = torch.svd(jac)
    log_sigma = (sigma + eps).log().norm(dim=-1)
    return log_sigma
