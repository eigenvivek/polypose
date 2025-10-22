from pathlib import Path

import click
import torch
from diffdrr.pose import RigidTransform
from torchio import ScalarImage
from xvr.dicom import read_xray

from models import fit_densese3, fit_densexyz, fit_polypose
from utils import KWARGS, get_training_frames, load_dataset

torch.manual_seed(42)


def load_xrays(xrays, frames):
    # Load ground truth X-ray images and camera poses
    imgs = []
    poses = []
    for idx in frames:
        xray, *_ = read_xray(f"{xrays}/{idx:03d}.dcm", crop=100)
        pose, *_ = torch.load(f"{xrays}/{idx:03d}.pt", weights_only=False)["pose"]
        imgs.append(xray)
        poses.append(pose)
    gt = torch.concat(imgs).cuda()
    poses = RigidTransform(torch.stack(poses)).cuda()

    return gt, poses


def save(subject_id, model_name, volume, model):
    # Save the warped volumes
    affine = ScalarImage(volume).affine
    warped_volume, warped_mask = model.warp_subject(affine=affine)
    warped_volume.save(f"results/{subject_id}/{model_name}_volume.nii.gz")
    warped_mask.save(f"results/{subject_id}/{model_name}_mask.nii.gz")

    # Save the warp and its Jacobian determinant
    torch.save(model, f"results/{subject_id}/{model_name}.ckpt")
    jacdet = (model.warp.jacdet < 0).to(torch.float32).mean().item() * 100
    with open(f"results/{subject_id}/{model_name}.txt", "w") as f:
        f.write(f"{jacdet:.4f}")


@click.command()
@click.option("--subject_id", type=click.IntRange(1, 6))
@click.option("--model_name", type=click.Choice(["polypose", "densexyz", "densese3"]))
def main(subject_id, model_name):
    # Load the required data
    volume, mask, xrays, _ = load_dataset(subject_id)
    frames = get_training_frames(subject_id)
    gt, poses = load_xrays(xrays, frames)

    # Run the registration
    if model_name == "polypose":
        model = fit_polypose(gt, poses, volume, mask, KWARGS)
    elif model_name == "densexyz":
        model = fit_densexyz(gt, poses, volume, mask, KWARGS)
    elif model_name == "densese3":
        model = fit_densese3(gt, poses, volume, mask, KWARGS)
    else:
        raise ValueError(f"Unrecognized model_name {model_name}")

    # Save the output
    Path(f"results/{subject_id}").mkdir(parents=True, exist_ok=True)
    save(subject_id, model_name, volume, model)


if __name__ == "__main__":
    main()
