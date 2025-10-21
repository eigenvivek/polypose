import numpy as np
import SimpleITK as sitk
import torch
from jaxtyping import Float


def load_ants_displacements(flow_path: str, invert: bool = False) -> Float[torch.Tensor, "B D H W 3"]:
    """
    Read an ANTs transform as a displacement field for polypose.models.DenseTranslation

    Displacement fields in ANTs are in world coordinates. Therefore, we need to convert them
    to voxel coordinates using the inverse affine matrix. Since we are transforming directions
    vectors, we don't need the translational component of the affine matrix.

    Adapted from: https://github.com/ANTsX/ANTsPy/issues/427#issuecomment-1449783073
    """
    # Load the displacement field
    disp = sitk.ReadImage(flow_path)
    if invert:
        disp = sitk.InvertDisplacementField(
            disp,
            maximumNumberOfIterations=20,
            maxErrorToleranceThreshold=0.01,
            meanErrorToleranceThreshold=0.0001,
            enforceBoundaryCondition=True,
        )

    # Get the direction and spacing of the displacement field
    direction = torch.tensor(disp.GetDirection()).reshape(3, 3)
    spacing = torch.diag(torch.tensor(disp.GetSpacing()))

    # Get the inverse affine matrix
    affine = torch.matmul(direction, spacing)
    affine_inv = torch.linalg.inv(affine)

    # Convert the displacement field from XYZ (SITK) to ZXY (numpy/torch)
    disp_arr = sitk.GetArrayFromImage(disp)
    disp_arr = np.transpose(disp_arr, axes=(3, 2, 1, 0))
    disp_tensor = torch.from_numpy(disp_arr).float()

    # Convert from world coordinates to voxel coordinates
    disp_tensor = torch.einsum("ij,jhwd->ihwd", affine_inv, disp_tensor)
    return disp_tensor.unsqueeze(0).permute(0, 2, 3, 4, 1)
