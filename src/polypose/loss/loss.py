from typing import List

import torch
from diffdrr.metrics import (
    GradientNormalizedCrossCorrelation2d,
    MultiscaleNormalizedCrossCorrelation2d,
)


class ImageLoss(torch.nn.Module):
    """Initialize the image similarity metric"""

    def __init__(
        self,
        beta: float = 0.5,  # Mixing parameter between the two similarity metrics
        mncc_patch_size: int = 9,  # Patch size for Multiscale Normalized Cross Correlation
        mncc_weights: List[float] = [0.5, 0.5],  # Weights for the global and local scales
        gncc_patch_size: int = 11,  # Patch size for Gradient Normalized Cross Correlation
        gncc_sigma: float = 0.0,  # Sigma for Gradient Normalized Cross Correlation
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
