import warnings

import pandas as pd
import torch
from diffdrr.pose import RigidTransform
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from tqdm import tqdm
from xvr.dicom import read_xray
from xvr.renderer import initialize_drr
from xvr.utils import XrayTransforms

from polypose.loss import ImageLoss

from utils import KWARGS, get_training_frames, load_dataset

warnings.filterwarnings("ignore", category=UserWarning, module="monai.metrics")
warnings.filterwarnings("ignore", category=FutureWarning, module="monai.utils.deprecate_utils")


def load_xrays(subject_id, xrays, segmentations, frames):
    # Load ground truth X-ray images, segmentations, and camera poses
    imgs = []
    segs = []
    cams = []

    for idx in range(37):
        if idx in frames or (subject_id == 1 and idx == 3):
            continue
        try:
            xray, *_ = read_xray(f"{xrays}/{idx:03d}.dcm", crop=100)
            pose, *_ = torch.load(f"{xrays}/{idx:03d}.pt", weights_only=False)["pose"]
        except FileNotFoundError:
            break

        seg = torch.load(f"{segmentations}/{idx:03d}.pt", weights_only=False).squeeze()
        seg = seg[50:-50, 50:-50][None, None].to(torch.float32).cuda()
        seg = resize(seg, (179, 179), interpolation=InterpolationMode.NEAREST_EXACT)

        imgs.append(xray.cuda())
        cams.append(RigidTransform(pose).cuda())
        segs.append(seg)

    imgs = torch.concat(imgs)
    segs = torch.concat(segs)

    return imgs, segs, cams


def load_drrs(subject_id, volume, mask, drr_kwargs):
    drr_kwargs["labels"] = "0,1,2,3,4,5,6,7"

    drr_original = initialize_drr(
        volume,
        mask,
        **drr_kwargs,
    )
    drr_polypose = initialize_drr(
        volume=f"results/{subject_id}/polypose_volume.nii.gz",
        mask=f"results/{subject_id}/polypose_mask.nii.gz",
        **drr_kwargs,
    )
    drr_densexyz = initialize_drr(
        volume=f"results/{subject_id}/densexyz_volume.nii.gz",
        mask=f"results/{subject_id}/densexyz_mask.nii.gz",
        **drr_kwargs,
    )
    drr_densese3 = initialize_drr(
        volume=f"results/{subject_id}/densese3_volume.nii.gz",
        mask=f"results/{subject_id}/densese3_mask.nii.gz",
        **drr_kwargs,
    )
    drr_original.rescale_detector_(0.125)
    drr_polypose.rescale_detector_(0.125)
    drr_densexyz.rescale_detector_(0.125)
    drr_densese3.rescale_detector_(0.125)
    xt = XrayTransforms(drr_original.detector.height)

    return drr_original, drr_polypose, drr_densexyz, drr_densese3, xt


@torch.no_grad()
def predict(cams, drr_original, drr_polypose, drr_densexyz, drr_densese3):
    preds_original = []
    preds_polypose = []
    preds_densexyz = []
    preds_densese3 = []
    for pose in tqdm(cams, ncols=100):
        preds_original.append(drr_original(pose, mask_to_channels=True))
        preds_polypose.append(drr_polypose(pose, mask_to_channels=True))
        preds_densexyz.append(drr_densexyz(pose, mask_to_channels=True))
        preds_densese3.append(drr_densese3(pose, mask_to_channels=True))

    preds_original = torch.concat(preds_original)
    preds_polypose = torch.concat(preds_polypose)
    preds_densexyz = torch.concat(preds_densexyz)
    preds_densese3 = torch.concat(preds_densese3)

    return preds_original, preds_polypose, preds_densexyz, preds_densese3


def labels_to_channels(seg, labels=range(0, 7)):
    return torch.concat([seg == idx for idx in labels], dim=1)


def reformat(segs):
    lfemur = segs[:, [5]] > 0
    rfemur = segs[:, [6]] > 0
    pelvis = segs[:, [1, 2, 3, 4]].sum(dim=1, keepdim=True) > 0
    pelvis = ~(lfemur | rfemur) & pelvis
    return torch.concat([pelvis, lfemur, rfemur], dim=1)


def surface(pred, true):
    dice = DiceMetric(reduction="none")
    haus = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="none")

    pred = reformat(pred)
    true = reformat(true)

    dice_metric = dice(pred, true).cpu()
    haus_metric = haus(pred, true).cpu()

    too_small = (true.sum(dim=[-1, -2]) < 500).cpu()
    dice_metric[too_small] = torch.nan
    haus_metric[too_small] = torch.nan

    return dice_metric, haus_metric


def bakedf(model_name, columns, *metrics):
    metrics = torch.concat(metrics, dim=-1)
    df = pd.DataFrame(metrics, columns=columns)
    df["model"] = model_name
    return df


def evaluate(subject_id):
    # Load the required data and DRRs
    volume, mask, xrays, segmentations = load_dataset(subject_id)
    frames = get_training_frames(subject_id)
    imgs, segs, cams = load_xrays(subject_id, xrays, segmentations, frames)
    drr_original, drr_polypose, drr_densexyz, drr_densese3, xt = load_drrs(subject_id, volume, mask, KWARGS)

    # Render test images through the warp fields
    preds_original, preds_polypose, preds_densexyz, preds_densese3 = predict(cams, drr_original, drr_polypose, drr_densexyz, drr_densese3)

    # Calculate mNCC
    mncc = ImageLoss()
    mncc_original = mncc(xt(preds_original.sum(dim=1, keepdim=True)), xt(imgs)).cpu().unsqueeze(1)
    mncc_polypose = mncc(xt(preds_polypose.sum(dim=1, keepdim=True)), xt(imgs)).cpu().unsqueeze(1)
    mncc_densexyz = mncc(xt(preds_densexyz.sum(dim=1, keepdim=True)), xt(imgs)).cpu().unsqueeze(1)
    mncc_densese3 = mncc(xt(preds_densese3.sum(dim=1, keepdim=True)), xt(imgs)).cpu().unsqueeze(1)

    # Calculate Dice and HD95
    dice_original, haus_original = surface(preds_original, labels_to_channels(segs))
    dice_polypose, haus_polypose = surface(preds_polypose, labels_to_channels(segs))
    dice_densexyz, haus_densexyz = surface(preds_densexyz, labels_to_channels(segs))
    dice_densese3, haus_densese3 = surface(preds_densese3, labels_to_channels(segs))

    # Construct a dataframe of results
    columns = ["mncc", "dice_pelvis", "dice_lfemur", "dice_rfemur", "hd95_pelvis", "hd95_lfemur", "hd95_rfemur"]
    df1 = bakedf("original", columns, mncc_original, dice_original, haus_original)
    df2 = bakedf("polypose", columns, mncc_polypose, dice_polypose, haus_polypose)
    df3 = bakedf("densexyz", columns, mncc_densexyz, dice_densexyz, haus_densexyz)
    df4 = bakedf("densese3", columns, mncc_densese3, dice_densese3, haus_densese3)
    df = pd.concat([df1, df2, df3, df4])

    df["model"] = pd.Categorical(
        df["model"],
        categories=["original", "polypose", "densexyz", "densese3"],
        ordered=True,
    )
    df["subject_id"] = subject_id
    df.to_csv(f"results/{subject_id}/metrics.csv", index=False)
    return df


def main():
    # Get mNCC, Dice, and HD95 for every subject
    dfs = []
    for subject_id in range(1, 7):
        df = evaluate(subject_id)
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv("results/metrics.csv")

    # Get %jacdets for every warp
    jacdets = []
    for subject_id in range(1, 7):
        for model_name in ["polypose", "densexyz", "densese3"]:
            with open(f"results/{subject_id}/{model_name}.txt", "r") as f:
                jacdet = float(f.read())
                jacdets.append([subject_id, model_name, jacdet])
    df = pd.DataFrame(jacdets, columns=["subject_id", "model", "jacdet"])
    df.to_csv("results/jacdet.csv", index=False)


if __name__ == "__main__":
    main()
