# %%
from pyNeuroWide import process, utils, io
import numpy as np
import os
from matplotlib import pyplot as plt
from cellpose import models, plot
from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage import measure
from skimage.feature import blob_log

# %% load data

# data = io.read_suite2p_bin(1,np.arange(1,2001),[0])[0]
data = np.load("/Users/bcraus/Library/Mobile Documents/com~apple~CloudDocs/Documents/BostonU/Research/DevorLab/code/datasets/test2Pdata.npy")
data = process.smooth_2d_new(data, sigma=1, axis=0)
data = process.bpf(data, fr=[0,5], fs=15, axis=0)
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
    - Uses padding instead of roll to avoid edge artifacts
    - Uses float32 to reduce memory
    """
    t, h, w = data.shape
    data = data.astype(np.float32)

    # Vectorized normalization: normalize each pixel's time series
    means = np.mean(data, axis=0, keepdims=True)
    stds = np.std(data, axis=0, keepdims=True)
    stds[stds == 0] = 1  # Avoid division by zero
    data_norm = (data - means) / stds

    # Compute local correlation using padding to avoid edge wrapping artifacts
    # Pad time dimension with edge values
    data_padded = np.pad(data_norm, ((0,0), (radius, radius), (radius, radius)), mode='edge')
    
    local_corr = np.zeros((h, w), dtype=np.float32)
    neighbor_count = 0

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            neighbor_count += 1
            # Extract shifted neighbor patch
            y_start = radius + dy
            x_start = radius + dx
            shifted = data_padded[:, y_start:y_start+h, x_start:x_start+w]
            local_corr += np.mean(data_norm * shifted, axis=0)

    local_corr /= neighbor_count
    return np.clip(local_corr, -1, 1)

corr_img = compute_local_correlation(data, radius=3)

# %% enhance features for soma segmentation
# Apply Gaussian smoothing to std image to enhance soma-like circular structures
std_smooth = gaussian_filter(std_img, sigma=1.5)

# Percentile-based normalization for robustness to outliers
def robust_normalize(img):
    """Normalize image using percentile-based scaling"""
    p2, p98 = np.percentile(img, [2, 98])
    return np.clip((img - p2) / (p98 - p2), 0, 1)

std_norm = robust_normalize(std_smooth)
corr_norm = robust_normalize(corr_img)

# Combine features: emphasize std (soma brightness) with correlation weighting
# Using weighted combination optimized for soma detection
combined_img = (0.6 * std_norm) + (0.4 * corr_norm)
img = combined_img

# %%

def doubleGaussFilt(img, sigma1=1, sigma2=4):
    small = gaussian_filter(img, sigma=sigma1)
    large = gaussian_filter(img, sigma=sigma2)

    return small - large

# hpf_img = img - gaussian_filter(img, sigma=5)
hpf_img = doubleGaussFilt(img, sigma1=2, sigma2=6)
clip_img = np.clip(hpf_img, 0, None)

blobs = blob_log(
    clip_img,
    min_sigma=1,
    max_sigma=5,
    num_sigma=10,
    threshold=0.1
)

radii = np.sqrt(2) * blobs[:,2]

fig, ax = plt.subplots()
ax.imshow(clip_img)

for (y, x, r) in zip(blobs[:,0], blobs[:,1], radii):
    circ = plt.Circle((x, y), r, color='red', fill=False, linewidth=1)
    ax.add_patch(circ)

plt.imshow(clip_img)
# %% Cellpose segmentation
# Use cyto2 model and optimize parameters for layer 5 soma size (~30-50 um)
model = models.CellposeModel()

masks, flows, styles = model.eval(
    img,
    diameter=30,  # Layer 5 soma typically 30-50 um; assuming ~1 um/pixel, use 40
    cellprob_threshold=-1.0,  # Lower threshold to catch all potential somas, filter later
    flow_threshold=0.4,
)

# %% post-processing: filter ROIs by morphological properties
def filter_rois_by_morphology(masks, min_area=500, max_area=15000, min_circularity=0.2):
    """
    Filter detected ROIs to match soma morphology.
    
    Layer 5 somas are:
    - Medium-sized (100-1500 pixels for ~40um diameter)
    - Roughly circular (circularity > 0.4)
    """
    unique_ids = np.unique(masks)
    unique_ids = unique_ids[unique_ids > 0]
    
    filtered_masks = np.zeros_like(masks)
    valid_id_map = {}
    new_id = 1
    
    for roi_id in unique_ids:
        roi = masks == roi_id
        area = np.sum(roi)
        
        # Check area constraint
        if area < min_area or area > max_area:
            continue
        
        # Compute circularity: 4*pi*area / perimeter^2
        labeled_roi, _ = ndimage.label(roi)
        props = measure.regionprops(labeled_roi)
        
        if len(props) == 0:
            continue
        
        prop = props[0]
        perimeter = prop.perimeter
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        
        if circularity >= min_circularity:
            filtered_masks[roi] = new_id
            valid_id_map[roi_id] = new_id
            new_id += 1
    
    return filtered_masks, valid_id_map

masks_filtered, valid_ids = filter_rois_by_morphology(masks)

# %% visualization with improvements
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: Features
axes[0, 0].imshow(std_img, cmap='gray')
axes[0, 0].set_title("Std Image (Activity)", fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(corr_img, cmap='hot')
axes[0, 1].set_title("Local Correlation", fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(combined_img, cmap='gray')
axes[0, 2].set_title("Combined Feature", fontsize=12)
axes[0, 2].axis('off')

# Row 2: Segmentation results
axes[1, 0].imshow(img, cmap='gray')
axes[1, 0].contour(masks > 0, colors='r', linewidths=0.5, levels=[0.5])
axes[1, 0].set_title(f"Raw Cellpose ({np.max(masks)} ROIs)", fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(img, cmap='gray')
axes[1, 1].contour(masks_filtered > 0, colors='g', linewidths=0.5, levels=[0.5])
axes[1, 1].set_title(f"After Morphological Filter ({np.max(masks_filtered)} ROIs)", fontsize=12)
axes[1, 1].axis('off')

# Segmentation overlay with labels
axes[1, 2].imshow(img, cmap='gray')
for roi_id in np.unique(masks_filtered):
    if roi_id > 0:
        roi = masks_filtered == roi_id
        y, x = np.where(roi)
        cy, cx = np.mean(y), np.mean(x)
        axes[1, 2].text(cx, cy, str(roi_id), color='cyan', fontsize=8, ha='center')
axes[1, 2].contour(masks_filtered > 0, colors='cyan', linewidths=0.5, levels=[0.5])
axes[1, 2].set_title("Filtered ROIs (Labeled)", fontsize=12)
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

print(f"\nSegmentation Summary:")
print(f"  Raw Cellpose detections: {np.max(masks)}")
print(f"  After morphological filtering: {np.max(masks_filtered)}")
print(f"  Cells removed by filter: {np.max(masks) - np.max(masks_filtered)}")

# %% adjust tuft masks
large_cells = binary_dilation(masks, iterations=10)
height, width = large_cells.shape
trim = 10

radii = np.sqrt(2) * blobs[:,2]

fig, ax = plt.subplots()
ax.imshow(clip_img)

for (y, x, r) in zip(blobs[:,0], blobs[:,1], radii):
    if not large_cells[int(y),int(x)]:
        if y > trim and x > trim and y < height - trim and x < width - trim:
            circ = plt.Circle((x, y), r, color='red', fill=False, linewidth=1)
            ax.add_patch(circ)

ax.contour(masks > 0, colors='g', linewidths=1, levels=[0.5])

plt.imshow(clip_img)