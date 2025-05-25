import cupy as cp
import torch
from cupyx.scipy.ndimage import distance_transform_edt as _distance_transform_edt
from torchio import Subject
from tqdm import tqdm


def inverse_distance(D: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute the inverse of the distance from a point to a surface."""
    return 1 / (eps * D**2 + 1)


def gravity(D: torch.Tensor, mass: float) -> torch.Tensor:
    """Compute the gravitational attraction of a point to a body with a given mass."""
    return mass / (D**2 + 1)


def exp(D: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute the exponentiated signed distance field."""
    return (eps * D).exp()


def compute_weights(
    subject: Subject,  # torchio subject with a labelmap
    labels: list | list[list],  # A list labels for each rigid body
    weightfn: str = "gravity",  # "gravity" or "invdf"
    normalize: bool = True,  # Normalize the weights to sum to 1 in each voxel
    **kwargs,
):
    """
    Given a voxelgrid labelmap and a list of lists defining the individual rigid bodies,
    compute the weight field for the polyrigid warp.
    """
    # Get the labelmap and voxelgrid spacing
    mask = subject.mask.data.squeeze()
    assert (spacing := subject.volume.spacing) == subject.mask.spacing

    # Create a mask for each label and compute the distance transform
    masks = []
    edtmaps = []
    for label in tqdm(labels):
        if isinstance(label, int):
            label = [label]
        structure = torch.stack([mask == idx for idx in label]).any(dim=0)
        if weightfn == "exp":
            edtmap = signed_distance_field(structure, spacing=spacing)
        else:
            edtmap = distance_transform_edt(~structure, spacing=spacing)
        masks.append(structure)
        edtmaps.append(edtmap)

    # Create a segmentation mask containing all labels
    segmentations = torch.stack([i * mask for i, mask in enumerate(masks, start=1)]).sum(dim=0)

    # Compute the mass of each structure
    masses = [mask.sum() for mask in masks]
    masses = torch.tensor(masses)
    masses = masses / masses.sum()

    # Get the weight field for each structure
    weights = []
    for mass, edtmap in zip(masses, edtmaps):
        if weightfn == "gravity":
            weight = gravity(edtmap, mass.item())
        elif weightfn == "invdf":
            weight = inverse_distance(edtmap, **kwargs)
        elif weightfn == "exp":
            weight = exp(edtmap, **kwargs)
        else:
            raise ValueError(f"weightfn must be 'gravity', 'invdf', or 'exp', not {weightfn}")
        weights.append(weight)
    weights = torch.stack(weights)

    # Optionally, normalize
    if normalize:
        weights = weights / weights.sum(dim=0, keepdim=True)

    return segmentations, weights.to(torch.float32)


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
