from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from diffdrr.visualization import plot_drr
from jaxtyping import Float
from matplotlib.colors import ListedColormap
from PIL import Image
from torchio import Subject


def plot(
    gt: Float[torch.Tensor, "B 1 H W"],  # Fixed X-rays
    img: Float[torch.Tensor, "B 1 H W"],  # Moving DRRs
    losses: Optional[Float[torch.Tensor, " B"]] = None,  # NCC for each image
    savepath: Optional[str] = None,  # If not None, save frame to savepath
    dpi: int = 100,  # DPI for each frame
) -> None:
    render = img.sum(dim=1, keepdim=True)
    imgs = torch.concat([gt, render])

    *_, h, w = gt.shape
    weight = (w / h) ** 0.95

    ncols = len(img)
    figsize = (2 * ncols * weight, 4)
    if losses is not None:
        title = [None for _ in range(ncols)]
        title += [f"{loss.item():.3f}" for loss in losses]
    else:
        title = None

    _, axs = plt.subplots(ncols=ncols, nrows=2, figsize=figsize, dpi=dpi)
    if axs.ndim == 1:
        axs = axs.reshape(-1, 1)
    plot_drr(imgs, axs=axs.flatten(), ticks=False, title=title)
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    axs[0, 0].set_ylabel("Fixed")
    axs[1, 0].set_ylabel("Moving")

    if savepath is None:
        plt.show()
    else:
        plt.gcf().set_size_inches(*figsize)
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0)
        plt.close()


def gif(imgpath: str, savepath: str, fps: int = 10, loop: bool = True) -> None:
    imgpath = Path(imgpath)
    imgs = [Image.open(p).convert("RGB") for p in sorted(imgpath.glob("*.png"))]
    imgs[0].save(
        savepath,
        save_all=True,
        append_images=imgs[1:],
        duration=1000 // fps,
        loop=0 if loop else 1,
        optimize=True,
        quality=95,  # High quality
    )


def plot_weights(
    subject: Subject,  # torchio subject with a volume
    segmentations: Float[torch.Tensor, "D H W"],  # Multilabel labelmap
    weights: Float[torch.Tensor, "K D H W"],  # Weights for each voxel
    dim: int = 0,  # Dimension of the volume to index along
    index: int = None,  # Index on a given dimension
    cmap: str = "gist_rainbow",  # Name of matplotlib colormap
    rot90: int = 1,  # Number of times to rotate slices by 90 degrees
) -> None:
    # Make sure the tensors are on CPU
    segmentations = segmentations.cpu()
    weights = weights.cpu()

    # Normalize the weight field
    alpha = weights.sum(dim=0)
    weights = weights / alpha
    alpha = (25 * alpha).sigmoid()

    # Get the data for plotting
    volume = subject.volume.data.squeeze()
    n_structs = len(segmentations.unique()) - 1
    if index is None:
        index = volume.shape[dim] // 2

    # Color using each structure using the colormap
    cmap = plt.get_cmap(cmap, n_structs)
    cmap1 = ListedColormap([(0, 0, 0, 0)] + [cmap(idx) for idx in range(n_structs)])
    x = segmentations.select(dim=dim, index=index)
    y = volume.select(dim=dim, index=index)

    # Plot the weights with a blend
    cmap2 = torch.tensor([cmap(idx) for idx in range(n_structs)], dtype=torch.float32)
    a = alpha.select(dim=dim, index=index)
    z = weights.select(dim=dim + 1, index=index).permute(1, 2, 0)
    z = (z @ cmap2).clamp(0, 1)
    z[..., -1] = a

    # Set the aspect and extent for imshow so the slices are isotropic
    W, H = x.shape
    X, Y = remove_at_index(subject.spacing, dim)
    kwargs = dict(aspect=Y / X, extent=[0, W, 0, H])

    # Make the plot
    plt.figure()

    plt.subplot(121)
    plt.imshow(y.clamp(-750, 750).rot90(k=rot90), cmap="gray", **kwargs)
    plt.imshow(x.rot90(k=rot90), cmap=cmap1, interpolation="nearest", alpha=0.75, vmax=n_structs, **kwargs)
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(z.rot90(k=rot90), **kwargs)
    plt.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    plt.show()


def remove_at_index(tup: tuple, index: int) -> tuple:
    return tup[:index] + tup[index + 1 :]
