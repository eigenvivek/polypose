import torch
from jaxtyping import Float


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
