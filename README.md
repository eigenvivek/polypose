# PolyPose

[![Paper shield](https://img.shields.io/badge/arXiv-2505.19256-red.svg)](https://arxiv.org/abs/2505.19256)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
<a href="https://colab.research.google.com/drive/1ui17wtfjxcjEM9QqMhD4sQXZB2NlMdlS?usp=sharing"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)


![PolyPose](.github/teaser.webp)

*Deformable 2D/3D registration via polyrigid transforms ([project page](https://polypose.csail.mit.edu/)).*

## Highlights

PolyPose is a fully deformable 2D/3D registration framework.

- 🔭 PolyPose is effective in both **sparse-view** and **limited-angle** registration.
- 🦾 PolyPose accurately solves this highly ill-constrained problem with polyrigid transforms.
- 🫀 PolyPose has been tested on multiple anatomical structures from different clinical specialties.

![PolyPose](.github/baselines.webp)

## Tutorial

After [setting up the environment](#setup), check out the tutorial notebook in [`notebooks/pelvis.ipynb`](notebooks/pelvis.ipynb) for a demonstration of PolyPose.

*Note*: 

- This tutorial requires ≥24 GB of VRAM.
- We are working on a tutorial with a smaller memory footprint that can be run on Google Colab (coming soon!).

## Setup

PolyPose depends on the following packages:
```
torch
diffdrr    # Differentiable X-ray rendering
xvr        # Rigid 2D/3D registration
monai      # Evaluation metrics
cupy       # GPU-accelerated distance field computations
jaxtyping  # Extensive type hints!
```

Download the package:
```
git clone https://github.com/eigenvivek/polypose
cd polypose
```

You can install the required packages using `virtualenv`:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or you can set up the packages using [`uv`](https://docs.astral.sh/uv/):
```
# Install uv, if necessary
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up the virtual environment with all dev requirements
uv sync --all-extras

# Install pre-commit hooks locally
uvx pre-commit install
```

## Experiments

To run the experiments in `PolyPose` on the `DeepFluoro` dataset, run the following scripts.

```
# Download the DeepFluoro dataset
uv run hf download eigenvivek/xvr-data --repo-type dataset

# Run PolyPose and baselines
cd experiments/deepfluoro/
sbatch run.sh

# Run the evaluation script once all jobs are finished
uv run python eval.py
```

`run.sh` is written with SLURM and is configured to run in parallel on a cluster of RTX A6000s.

## Citing `PolyPose`

If you find `PolyPose` useful for your work, please cite our [paper](https://arxiv.org/abs/2505.19256):

```
@article{gopalakrishnan2025polypose,
  title={PolyPose: Deformable 2D/3D Registration via Polyrigid Transforms},
  author={Gopalakrishnan, Vivek and Dey, Neel and Golland, Polina},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```
