import torch
from diffdrr.data import read
from tqdm import tqdm
from xvr.renderer import initialize_drr
from xvr.utils import XrayTransforms

from polypose import DenseSE3Field, DenseTranslationField, PolyPose
from polypose.loss import ImageLoss
from polypose.weights import compute_weights


def fit_polypose(
    gt,  # X-rays
    poses,  # C-arm poses for X-rays
    volume,  # Preoperative CT
    mask,  # CT segmentation mask
    drr_kwargs,  # X-ray imaging parameters
):
    # Compute the weight field
    subject = read(volume, mask)
    _, weights = compute_weights(subject, [[1, 2, 3, 4, 7], [5], [6]])

    # Run the registration
    imagesim = ImageLoss()
    for idx, (resample, n_itrs) in enumerate(zip([4.0, 2.0, None], [51, 51, 51])):
        drr_kwargs["read_kwargs"] = {
            "bone_attenuation_multiplier": 2.0,
            "resample_target": resample,
        }
        drr = initialize_drr(volume, mask, **drr_kwargs)
        drr.rescale_detector_(0.125)
        xt = XrayTransforms(drr.detector.height)

        if idx == 0:
            model = PolyPose(drr, weights).cuda()
        else:
            model = PolyPose(drr, weights, model.poses_rot, model.poses_xyz).cuda()
        optimizer = torch.optim.Adam(
            [
                {"params": [model.poses_rot], "lr": 1e-2},
                {"params": [model.poses_xyz], "lr": 1e-0},
            ],
            maximize=True,
        )

        for itr in (pbar := tqdm(range(n_itrs), ncols=100)):
            img = model(poses)
            loss = imagesim(xt(gt), xt(img))
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.mean().item()})

    return model


def fit_densexyz(
    gt,  # X-rays
    poses,  # C-arm poses for X-rays
    volume,  # Preoperative CT
    mask,  # CT segmentation mask
    drr_kwargs,  # X-ray imaging parameters
):
    # Load the model
    drr_kwargs["read_kwargs"] = {"bone_attenuation_multiplier": 2.0}
    drr = initialize_drr(volume, mask, **drr_kwargs)
    drr.rescale_detector_(0.125)
    xt = XrayTransforms(drr.detector.height)

    model = DenseTranslationField(drr).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, maximize=True)

    # Run the registration
    imagesim = ImageLoss()
    for itr in (pbar := tqdm(range(501), ncols=100)):
        img = model(poses)
        loss = imagesim(xt(gt), xt(img)) - 1e-2 * model.warp.divergence.mean()
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.mean().item()})

    return model


def fit_densese3(
    gt,  # X-rays
    poses,  # C-arm poses for X-rays
    volume,  # Preoperative CT
    mask,  # CT segmentation mask
    drr_kwargs,  # X-ray imaging parameters
):
    # Load the model
    drr_kwargs["read_kwargs"] = {"bone_attenuation_multiplier": 2.0}
    drr = initialize_drr(volume, mask, **drr_kwargs)
    drr.rescale_detector_(0.125)
    xt = XrayTransforms(drr.detector.height)

    model = DenseSE3Field(drr).cuda()
    optimizer = torch.optim.Adam(
        [
            {"params": [model.se3_rot], "lr": 1e-2},
            {"params": [model.se3_xyz], "lr": 1e-0},
        ],
        maximize=True,
    )

    # Run the registration
    imagesim = ImageLoss()
    for itr in (pbar := tqdm(range(501), ncols=100)):
        img = model(poses)
        loss = imagesim(xt(gt), xt(img)) - 1e-3 * model.warp.elastic.abs().mean()
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.mean().item()})

    return model
