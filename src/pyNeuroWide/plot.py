"""
plot.py

Functions to plot.

author: Bradley Rauscher (June, 2026)
"""
# %%
from matplotlib import pyplot as plt
from importlib.resources import files
import numpy as np

# %%

def allenMap(ax, val, side="both", shrink=True, clabel="", clim=None, cmap='viridis', title=""):
    data_path = files("pyNeuroWide.data") / "refAllen.npy"
    data = np.load(data_path, allow_pickle=True).item()
    
    (H,W) = data['refParcellation']['Masks'].shape[0:2]

    if side == "both":
        masks = data['refParcellation']['Masks'].sum(axis=3)
    elif side == "left":
        masks = data['refParcellation']['Masks'][:,:,:,0]
    elif side == "right":
        masks = data['refParcellation']['Masks'][:,:,:,1]
    
    img = np.zeros((H,W))

    for i in range(12):
        img[masks[:,:,i].astype(bool)] = val[i]

    refBM = data['refBM'].astype(bool)
    img[~refBM] = 0

    refBM[~masks.sum(axis=2).astype(bool)] = 0
    if shrink:
        r, c = np.where(refBM != 0)
        img = img[r.min():r.max()+1,c.min():c.max()+1]
        refBM = refBM[r.min():r.max()+1,c.min():c.max()+1]

    alpha = np.zeros_like(img)
    alpha[refBM] = 1.0

    im = ax.imshow(img, alpha=alpha)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(title)

    c = plt.colorbar(im, ax=ax)
    c.set_label(clabel)
    if clim is not None:
        im.set_clim(clim[0], clim[1])
    im.set_cmap(cmap)
    
    return img

def lineError(ax, x, y, error, color=None, ylabel="", xlabel="", xlim=None, ylim=None, yscale='linear', xscale='linear', alpha=0.3, linewidth=1):
    ax.fill_between(x, y - error, y + error, color=color, alpha=alpha)
    ax.plot(x, y, color=color, linewidth=1)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_xlim(ylim[0], ylim[1])