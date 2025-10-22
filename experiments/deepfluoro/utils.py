from pathlib import Path

from huggingface_hub import snapshot_download

# X-ray imaging parameters
KWARGS = dict(
    labels=None,
    orientation="PA",
    height=1436,
    width=1436,
    sdd=1020.0,
    delx=0.194,
    dely=0.194,
    x0=0.0,
    y0=0.0,
    reverse_x_axis=True,
    renderer="trilinear",
    drr_kwargs={"voxel_shift": 0.0},
    read_kwargs={"bone_attenuation_multiplier": 2.0},
)


def get_training_frames(subject_id):
    """Return two frames that capture the right and left femurs, respectively."""
    if subject_id == 1:
        frames = [10, 9]
    elif subject_id == 2:
        frames = [29, 31]
    elif subject_id == 3:
        frames = [17, 22]
    elif subject_id == 4:
        frames = [0, 33]
    elif subject_id == 5:
        frames = [2, 47]
    elif subject_id == 6:
        frames = [9, 7]
    else:
        raise ValueError(f"subject_id must be in 1...6, now {subject_id}")
    return frames


def load_dataset(subject_id, dataset="deepfluoro"):
    # Download the dataset (or load a cache)
    data = f"{dataset}/subject{subject_id:02d}"
    download_path = snapshot_download(repo_id="eigenvivek/xvr-data", repo_type="dataset", allow_patterns=[data])
    datapath = Path(download_path) / data

    # Make paths to the relevant images
    volume = datapath / "volume.nii.gz"
    mask = datapath / "mask.nii.gz"
    xrays = datapath / "xrays"
    segs = datapath / "segmentations"

    return volume, mask, xrays, segs
