# %%
import os
import numpy as np
import torch
import argparse
from pyNeuroWide import io, process
import numpy as np
import os
from matplotlib import pyplot as plt
from cellpose import models
from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage import measure
from skimage.feature import blob_log

torch.set_num_threads(6)

# %%
def classify(data: str, ops=None, findCells=True):
    # %% extract parameters
    if ops is None:
        ops = default_ops()

    resolution = ops["resolution"]
    local_corr_radius = ops["local_corr_radius"] / resolution
    spatial_filt_gamma1 = ops["spatial_filt_gamma1"] / resolution
    spatial_filt_gamma2 = ops["spatial_filt_gamma2"] / resolution
    min_sigma = ops["blob_log"]["min_sigma"] / resolution
    max_sigma = ops["blob_log"]["max_sigma"] / resolution
    diameter = ops["Cellpose"]["diameter"] / resolution
    dendrite_trim = ops["dendrite_trim"] / resolution

    # %% data processing
    data = process.bpf(data, fr=[0,ops["low_pass_cutoff"]], fs=ops["framerate"], axis=0)

    # %% compute reference images
    corr_img = compute_local_correlation(data, radius=local_corr_radius)
    std_img = data.std(axis=0)

    std_norm = robust_normalize(std_img)
    corr_norm = robust_normalize(corr_img)
    
    ref_img = (0.6 * std_norm) + (0.4 * corr_norm)

    # %% find dendrite trunks
    hpf_img = doubleGaussFilt(ref_img, sigma1=spatial_filt_gamma1, sigma2=spatial_filt_gamma2)
    hpf_img = np.clip(hpf_img, 0, None)

    blobs = blob_log(
        hpf_img,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=ops["blob_log"]["num_sigma"],
        threshold=ops["blob_log"]["threshold"]
    )

    # %% find cell masks
    if findCells:
        model = models.CellposeModel()

        masks, flows, styles = model.eval(
            ref_img,
            diameter=diameter,  # Layer 5 soma typically 30-50 um; assuming ~1 um/pixel, use 40
            cellprob_threshold=ops["Cellpose"]["cellprob_threshold"],  # Lower threshold to catch all potential somas, filter later
            flow_threshold=ops["Cellpose"]["flow_threshold"],
        )
    else:
        masks = np.zeros(ref_img.shape)

    # %% filter cell masks and dendritic trunks
    masks_filtered, _ = filter_rois_by_morphology(masks)
    filtered_blobs = filter_trunks(blobs, masks_filtered, trim=dendrite_trim)

    # %% plot results
    fig, ax = plt.subplots()
    ax.imshow(hpf_img)

    for (y, x, r) in filtered_blobs:
        circ = plt.Circle((x, y), r, color='red', fill=False, linewidth=1)
        ax.add_patch(circ)

    ax.contour(masks_filtered > 0, colors='g', linewidths=1, levels=[0.5])

    plt.imshow(hpf_img)

    # %%
    return masks_filtered, filtered_blobs

def default_ops():
    ops = {
        "resolution": 1, # um / pixel
        "Cellpose": {
            "cellprob_threshold": -1.0,
            "diameter": 8, # um
            "flow_threshold": 0.4
        },
        "blob_log": {
            "min_sigma": 0.3, # um
            "max_sigma": 1.5, # um
            "num_sigma": 10,
            "threshold": 0.1
        },
        "dendrite_trim": 3, # um
        "framerate": 15, # Hz
        "low_pass_cutoff": 0.5, # Hz
        "local_corr_radius": 1, # um
        "spatial_filt_gamma1": 2/3, # um
        "spatial_filt_gamma2": 2 # um
    }

    return ops

def compute_local_correlation(data, radius=3):
    """
    Compute correlation of each pixel with its local neighborhood.
    """
    t, h, w = data.shape
    data = data.astype(np.float32)
    radius = int(radius)

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

def robust_normalize(img):
    """Normalize image using percentile-based scaling"""
    p2, p98 = np.percentile(img, [2, 98])
    return np.clip((img - p2) / (p98 - p2), 0, 1)

def doubleGaussFilt(img, sigma1=1, sigma2=4):
    small = gaussian_filter(img, sigma=sigma1)
    large = gaussian_filter(img, sigma=sigma2)

    return small - large

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

def filter_trunks(rois, cell_masks, trim=10):
    trim = int(trim)
    dilated_cells = binary_dilation(cell_masks, iterations=trim)
    h, w = dilated_cells.shape
    radii = np.sqrt(2) * rois[:,2]

    filtered_rois = []
    
    for (y, x, r) in zip(rois[:,0], rois[:,1], radii):
        if not dilated_cells[int(y),int(x)]:
            if y > trim and x > trim and y < h-trim and x < w-trim:
                filtered_rois.append([y, x, r])
    
    return filtered_rois