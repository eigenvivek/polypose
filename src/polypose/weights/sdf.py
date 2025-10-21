import cupy as cp
import torch
from cupyx.scipy.ndimage import distance_transform_edt as _distance_transform_edt


def distance_transform_edt(
    x: torch.Tensor,  # Binary 3D segmentation mask
    spacing: float | tuple[float] = None,  # Voxel spacing
):
    """
    If outside x, min Euclidean distance to surface of x
    If inside x, 0
    """
    x = cp.asarray(x)
    x = _distance_transform_edt(x, sampling=spacing)
    return torch.as_tensor(x)


def signed_distance_field(
    x: torch.Tensor,  # Binary 3D segmentation mask
    spacing: float | tuple[float] = None,  # Voxel spacing
):
    """
    If outside x, min Euclidean distance to surface of x
    if inside x, negative min Euclidean distance to surface of x
    """
    inside = distance_transform_edt(x, spacing)
    outside = distance_transform_edt(~x, spacing)
    return inside - outside
