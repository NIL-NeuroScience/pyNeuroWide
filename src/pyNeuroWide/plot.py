"""
plot.py

Functions to plot.

author: Bradley Rauscher (June, 2026)
"""
# %%
from matplotlib import pyplot as plt
from importlib.resources import files
import numpy as np
from pyNeuroWide import colors, IRF
from scipy.stats import linregress
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib as mpl

# %%

def allenMap(ax, val, side="both", shrink=True, clabel="", clim=None, cmap='viridis', title="", ylabel="", outline=None, linewidth=2, color="red", padding=5, cbar=True):
    data_path = files("pyNeuroWide.data") / "refAllen.npy"
    data = np.load(data_path, allow_pickle=True).item()
    
    (H,W) = data['refParcellation']['Masks'].shape[0:2]

    refBM = data['refBM'].astype(bool)
    N = len(val)
    if N == 24:
        masks = np.concat([data['refParcellation']['Masks'][:,:,:,0],data['refParcellation']['Masks'][:,:,:,1]], axis=2)
    else:
        if side == "both":
            masks = data['refParcellation']['Masks'].sum(axis=3)
        elif side == "left":
            masks = data['refParcellation']['Masks'][:,:,:,0]
            refBM[:,300::] = 0
        elif side == "right":
            masks = data['refParcellation']['Masks'][:,:,:,1]
            refBM[:,0:300] = 0

    img = np.zeros((H,W))
    
    for i in range(N):
        img[masks[:,:,i].astype(bool)] = val[i]
    
    img[~refBM] = 0

    refBM[~masks.sum(axis=2).astype(bool)] = 0
    if shrink:
        r, c = np.where(refBM != 0)
        img = img[r.min()-padding:r.max()+1+padding,c.min()-padding:c.max()+1+padding]
        refBM = refBM[r.min()-padding:r.max()+1+padding,c.min()-padding:c.max()+1+padding]
        masks = masks[r.min()-padding:r.max()+1+padding,c.min()-padding:c.max()+1+padding,:]

    alpha = np.zeros_like(img)
    alpha[refBM] = 1.0

    if ax == False:
        return img, alpha

    im = ax.imshow(img, alpha=alpha, interpolation="nearest")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    if cbar:
        c = plt.colorbar(im, ax=ax)
        c.solids.set_rasterized(True)
        c.set_label(clabel)
    
    if clim is not None:
        im.set_clim(clim[0], clim[1])
    im.set_cmap(cmap)

    if outline is not None:
        if isinstance(outline, list):
            N = len(outline)
        else:
            N = 1
            outline = [outline]

        if isinstance(color, str):
            color = [color] * N
        elif isinstance(color[0], list):
            N_c = len(color)
            if N_c == 1:
                color = color * N
        else:
            color = [color] * N

        outline_masks = masks * refBM[:,:,None]
        for i in range(N):
            ax.contour(outline_masks[:,:,outline[i]], levels=[0.5], colors=color[i], linewidths=linewidth)
    
    return img, alpha

def lineError(ax, x, y, error, color=None, ylabel="", xlabel="", xlim=None, ylim=None, yscale='linear', xscale='linear', alpha=0.3, linewidth=1, title=None):
    if xlim is not None:
        idx = np.flatnonzero((x > xlim[0]) & (x < xlim[1]))
        lo = max(idx[0] - 1, 0)
        hi = min(idx[-1] + 2, len(x))
        x = x[lo:hi]
        y = y[lo:hi]
        error = error[lo:hi]
    
    ax.fill_between(x, y - error, y + error, color=color, alpha=alpha, linewidth=0)
    ax.plot(x, y, color=color, linewidth=1)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.set_title(title)

def FCmap(ax, data, cmap="viridis", clim=None, clabel=None, diag=None, ylabel=None, title=None):
    data = data.copy()

    N = data.shape[0]
    # replace diagonal values with 'diag'
    if diag is not None:
        data[np.diag_indices(N)] = diag
    
    im = ax.imshow(data)

    c = plt.colorbar(im, ax=ax)
    c.solids.set_rasterized(True)
    c.set_label(clabel)
    if clim is not None:
        im.set_clim(clim[0], clim[1])
    im.set_cmap(cmap)

    if ylabel:
        remove_axes(ax,"RBT")
        ax.spines["left"].set_visible(False)
        ax.set_yticks(np.arange(N))
        ax.set_yticklabels(ylabel)
    else:
        remove_axes(ax,"LRBT")

    ax.set_title(title)
    
def prepare(fig):
    fig.patch.set_visible(False)

    for ax in fig.axes:
        ax.patch.set_visible(False)
        for line in ax.lines:
            1
            # line.set_clip_on(False)
        for collection in ax.collections:
            1
            # collection.set_clip_on(False)
        for patch in ax.patches:
            patch.set_clip_on(False)

def remove_axes(axes, directions):
    if "L" in directions.upper():
        axes.spines["left"].set_visible(False)
        axes.set_yticks([])
    if "R" in directions.upper():
        axes.spines["right"].set_visible(False)
    if "T" in directions.upper():
        axes.spines["top"].set_visible(False)
    if "B" in directions.upper():
        axes.spines["bottom"].set_visible(False)
        axes.set_xticks([])

def bar(axes, data, ylim=None, labels=None, color=None, linewidth=2, alpha=1, error="SEM", capsize=3, ylabel=None, title=None, width=0.5, size=50):
    B = len(data)

    if labels:
        bar_labels = labels
    else:
        bar_labels = range(B)

    if color:
        if isinstance(color[0], list):
            c = len(color)
        else:
            color = [color]
            c = len(color)        
    else:
        c = 0
        color = [colors.mb()]

    if error.upper() == "SEM":
        error_vals = [np.std(i) / np.sqrt(len(i)) for i in data]
    elif error.upper() == "STD":
        error_vals = [np.std(i) for i in data]
    elif error.upper() == "NONE":
        error_vals = None
    
    mean_vals = [np.mean(i) for i in data]
    
    axes.bar(bar_labels, mean_vals, yerr=error_vals, capsize=capsize, color=[0,0,0,0], edgecolor=color, linewidth=linewidth, width=width)

    for i, values in enumerate(data):
        if c > 1:
            scatter_color = color[i]
        elif c == 0:
            scatter_color = color
        else:
            scatter_color = color[0]

        x = np.random.normal(i, 0.05, size=len(values))
        axes.scatter(x, values, zorder=0.5, color=scatter_color, alpha=alpha, linewidth=0, s=size)

    if ylim:
        axes.set_ylim(ylim[0], ylim[1])
    
    axes.set_title(title)
    axes.set_ylabel(ylabel)
    remove_axes(axes, "RT")

def scatter(ax, X, Y, alpha=1, xlim=None, ylim=None, xlabel=None, ylabel=None, plotReg=True, cmp=None, size=50):
    
    N = len(Y)

    if cmp is not None:
        if isinstance(cmp, str):
            colors = mpl.colormaps[cmp](np.linspace(0,1,N))
        elif len(cmp) == 1:
            colors = colors * N
        else:
            colors = cmp
    else:
        colors = [None] * N

    df = pd.DataFrame()
    for i in range(N):
        tmp_df = pd.DataFrame({
            'HbT': Y[i],
            'Ca': X[i],
            'mouse': i
        })
        df = pd.concat([df, tmp_df], ignore_index=True)

    model = smf.mixedlm(
        "HbT ~ Ca",
        df,
        groups=df["mouse"],
        re_formula="~Ca"
    )
    results = model.fit()

    r = []
    for i in range(N):
        ax.scatter(X[i],Y[i],alpha=alpha,color=colors[i],s=size, linewidth=0)
        r.append(IRF.corr(X[i],Y[i]))

    # slope, intercept, r, p, stderr = linregress(np.concatenate(X), np.concatenate(Y))

    if xlim is None:
        x_bounds = ax.get_xlim()
        x_bounds = np.array([x_bounds[0],x_bounds[1]])
    else:
        x_bounds = np.array([xlim[0],xlim[1]])

    if plotReg:
        for i, (mouse, d) in enumerate(df.groupby('mouse')):
            coef = results.random_effects[mouse]
            intercept = results.fe_params['Intercept'] + coef['Group']
            slope = results.fe_params['Ca'] + coef['Ca']
            ax.plot(x_bounds, x_bounds * slope + intercept,color=colors[i])

    ax.set_xlim(x_bounds[0],x_bounds[1])
    if ylim:
        ax.set_xlim(ylim[0],ylim[1])

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    remove_axes(ax, "TR")

    ax.set_title(f"β1 = {results.fe_params['Ca']:.2e},\np = {results.pvalues['Ca']:.2e}")

    return results.fe_params['Ca'], results.pvalues['Ca'], np.array(r)
# %%
