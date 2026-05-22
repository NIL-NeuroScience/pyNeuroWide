# %%
from pyNeuroWide import utils, io, processing
import numpy as np
import os
from matplotlib import pyplot as plt
from cellpose import models, plot

# %% load data

data = io.read_suite2p_bin(1,np.arange(1,2001),[0])[0]
data = processing.smooth_2d_new(data, sigma=1, axis=0)
data = processing.bpf(data, fr=[0,5], fs=15, axis=0)
# data is (t, y, x) np.array of two-photon neural Ca imaging data

# %% plot mean
mean_img = data.mean(axis=0)
std_img = data.std(axis=0)

# %% compute local correlation feature
# For each pixel, compute correlation with local neighborhood
# Soma have high local correlation; neuropil is more incoherent
def compute_local_correlation(data, radius=3):
    """
    Compute correlation of each pixel with its local neighborhood.

    Optimizations:
    - Vectorized normalization (no pixel-by-pixel std calculation)
    - Efficient neighbor correlation using np.roll
    - Uses float32 to reduce memory
    """
    t, h, w = data.shape
    data = data.astype(np.float32)

    # Vectorized normalization: normalize each pixel's time series
    means = np.mean(data, axis=0, keepdims=True)
    stds = np.std(data, axis=0, keepdims=True)
    stds[stds == 0] = 1  # Avoid division by zero
    data_norm = (data - means) / stds

    # Compute local correlation using efficient vectorized operations
    # For each neighbor offset, compute correlation and accumulate
    local_corr = np.zeros((h, w), dtype=np.float32)
    neighbor_count = 0

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            neighbor_count += 1
            # Roll to shift neighbor pixels, multiply element-wise, average over time
            shifted = np.roll(np.roll(data_norm, dy, axis=1), dx, axis=2)
            local_corr += np.mean(data_norm * shifted, axis=0)

    local_corr /= neighbor_count
    return np.clip(local_corr, -1, 1)

corr_img = compute_local_correlation(data, radius=3)

# %% combine features for cellpose
# Use correlation to enhance cell body contrast
combined_img = (std_img / std_img.max()) + (corr_img / corr_img.max())
img = combined_img

model = models.Cellpose(model_type='cyto3')

masks, flows, styles, diams = model.eval(
    img,
    diameter=50,
    channels=[0,0],
    flow_threshold=0.2,
    cellprob_threshold=0.0
)

plt.figure(figsize=(15,5))
plt.subplot(131)
plt.imshow(std_img, cmap='gray')
plt.title("Std Image (Original)")
plt.axis('off')

plt.subplot(132)
plt.imshow(corr_img, cmap='hot')
plt.title("Local Correlation")
plt.axis('off')

plt.subplot(133)
plt.imshow(img, cmap='gray')
plt.contour(masks, colors='r', linewidths=0.5)
plt.title("Cellpose ROIs (Combined Features)")
plt.axis('off')

plt.tight_layout()
plt.show()

# %%
