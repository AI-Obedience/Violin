import cv2
import numpy as np
import torch
from tqdm import tqdm
from skimage.color import deltaE_ciede2000

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  IMPORTANT: 'img_gt' (or 'path_gt' or 'tensor2') MUST be the Ground Truth. 
  The code uses the Ground Truth to automatically detect color split directions.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

You can use:
    - Color_metrics_from_img_bgr
    - Color_metrics_from_img_path
    - Color_metrics_from_img_list
    - Color_metrics_from_tensor

KEY PARAMETER - 'is_multi_block':
    - Set to False (default): Evaluates the entire image as a single color field.
    - Set to True: Used for 'Variation 2' (Dual-color blocks). 
      The function will:
        1. Analyze the Ground Truth to detect if the split is Vertical or Horizontal.
        2. Automatically divide both images into two corresponding blocks.
        3. Calculate metrics for each block and return the average.

METRICS INCLUDED:
    - Precision: d_rgb_ed (RGB Distance), d_lab_00 (CIEDE2000).
    - Purity: d_sd (Std Dev), d_hf (FFT High-Freq), d_ced (Canny Edge Density).
    - d_mean: The arithmetic mean of all 5 normalized metrics.
"""

# ==========================================
# 1. Utility Functions
# ==========================================

def load_image(img_path):
    """Load image from path."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Cannot load image: {img_path}')
    return img

def convert_BGR_to_LAB(img_bgr):
    """Convert BGR image to standard LAB space [0-100, -128-127, -128-127]."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV LAB ranges: L[0,255], a[0,255], b[0,255]
    # Scaling to standard CIE LAB ranges
    lab[:, :, 0] = lab[:, :, 0] * 100.0 / 255.0
    lab[:, :, 1] = lab[:, :, 1] - 128.0
    lab[:, :, 2] = lab[:, :, 2] - 128.0
    return lab

def tensor2npBGR(tensor):
    """Convert [C, H, W] Tensor to [H, W, C] BGR Numpy array."""
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()
    img = tensor.detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    if img.dtype == np.float32 and np.max(img) <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    # Assuming input tensor is RGB, convert to BGR for OpenCV
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def change_list2dict(dicts):
    """Convert list of dicts to dict of lists."""
    new_dict = dict()
    for key in dicts[0].keys():
        new_dict[key] = [d[key] for d in dicts]
    return new_dict

def dict_mean(dicts):
    """Calculate mean for each element in the dict."""
    new_dict = dict()
    for key in dicts.keys():
        new_dict[key] = sum(dicts[key]) / len(dicts[key])
    return new_dict

# ==========================================
# 2. Core Metric Components
# ==========================================
def calc_rgb_dist(img1_bgr, img2_bgr):
    """
    (1) RGB distance (increased fault tolerance):
    Calculate the distance of the average color of the region and ignore small numerical fluctuations.
    """
    
    m1 = np.mean(img1_bgr, axis=(0, 1))
    m2 = np.mean(img2_bgr, axis=(0, 1))
    
    dist = np.linalg.norm(m1 - m2)
    
    # --- Fault Tolerance Logic ---
    # If the average color difference is within 5 pixel values ​​(0-255 range),
    #  it is considered to be without deviation.
    tolerance = 5.0 
    dist = max(0, dist - tolerance)
    
    return float(min(dist / 441.67, 1.0))

def calc_lab_ciede(img1_bgr, img2_bgr):
    """
    (2) CIEDE2000 (increased fault tolerance):
    Calculate the perceptual difference of the average colors and introduce JND (Just Noticeable Difference).
    """
    # Convert to LAB and take the mean
    lab1 = convert_BGR_to_LAB(img1_bgr)
    lab2 = convert_BGR_to_LAB(img2_bgr)
    m1 = np.mean(lab1, axis=(0, 1))
    m2 = np.mean(lab2, axis=(0, 1))
    
    # Calculate the perceptual difference between the two average color points
    res = deltaE_ciede2000(m1[None, None, :], m2[None, None, :]).item()
    
    # --- Fault Tolerance Logic ---
    # JND (Just Noticeable Difference): Usually considered that Delta E < 3.0 is imperceptible to the human eye
    jnd_threshold = 3.0 
    res = max(0, res - jnd_threshold)
    
    return float(min(res / 100.0, 1.0))

# def calc_rgb_dist(img1_bgr, img2_bgr):
#     """(1) RGB Euclidean Distance: Measures digital signal alignment."""
#     diff = img1_bgr.astype(np.float32) - img2_bgr.astype(np.float32)
#     dist = np.sqrt(np.sum(diff**2, axis=2)).mean().item()
#     return min(dist / 441.67, 1.0) # Normalized by sqrt(255^2 * 3)

# def calc_lab_ciede(img1_bgr, img2_bgr):
#     """(2) CIEDE2000: Measures perceptual color accuracy."""
#     lab1 = convert_BGR_to_LAB(img1_bgr)
#     lab2 = convert_BGR_to_LAB(img2_bgr)
#     res = deltaE_ciede2000(lab1, lab2).mean().item()
#     return min(res / 100.0, 1.0) # Normalized by max CIE error

def calc_std(img_bgr):
    """(3) Standard Deviation: Measures global pixel consistency."""
    stds = [np.std(img_bgr[:, :, i]) for i in range(3)]
    avg_std = np.mean(stds).item()
    return min(avg_std / 127.5, 1.0) # Normalized by half of 255

def calc_hf_ratio(img_bgr, threshold=0.02):
    """(4) High-Frequency Ratio: Measures texture artifacts in frequency domain."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(f_shift) ** 2
    
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol)**2 + (y - crow)**2)
    mask = (dist / np.sqrt(crow**2 + ccol**2)) > threshold
    
    total_energy = np.sum(magnitude)
    if total_energy == 0: return 0.0
    return (np.sum(magnitude[mask]) / total_energy).item()

def calc_ced(img_bgr, low=20, high=60):
    """(5) Canny Edge Density: Measures structural artifacts/edges."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    edge_pixels = np.sum(edges > 0)
    return (edge_pixels / edges.size).item()

# ==========================================
# 3. Main Interface Functions
# ==========================================

def auto_infer_split(img_gt_bgr):
    """
    Infer if the GT image is split vertically (left-right) or horizontally (top-bottom).
    """
    # Use the middle row and middle column to detect the transition
    h, w = img_gt_bgr.shape[:2]
    mid_row = img_gt_bgr[h // 2, :, :].astype(np.float32)
    mid_col = img_gt_bgr[:, w // 2, :].astype(np.float32)

    # Calculate the max gradient (difference) along x and y axes
    diff_x = np.max(np.linalg.norm(np.diff(mid_row, axis=0), axis=1))
    diff_y = np.max(np.linalg.norm(np.diff(mid_col, axis=0), axis=1))

    return 'v' if diff_x > diff_y else 'h'

def Color_metrics_from_img_bgr(img_gen, img_gt, is_multi_block=False, split='auto'):
    """
    Primary evaluator for a BGR image pair.
    is_multi_block: Set True for Variation 2 (Dual-color split).
    split: 'v' for vertical split, 'h' for horizontal split.
    """

    h_gt, w_gt = img_gt.shape[:2]
    if img_gen.shape[0] != h_gt or img_gen.shape[1] != w_gt:
        img_gen = cv2.resize(img_gen, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC)


    if not is_multi_block:
        res = {
            'd_rgb_ed': calc_rgb_dist(img_gen, img_gt),
            'd_lab_00': calc_lab_ciede(img_gen, img_gt),
            'd_sd':     calc_std(img_gen),
            'd_hf':     calc_hf_ratio(img_gen),
            'd_ced':    calc_ced(img_gen)
        }
    else:
        # Step 1: Automatically detect split direction from Ground Truth
        actual_split = auto_infer_split(img_gt) if split == 'auto' else split
        # print(actual_split)
        
        h, w = img_gen.shape[:2]
        # Step 2: Perform the split
        if actual_split == 'v': # Left-Right
            m = w // 2
            blocks = [(img_gen[:, :m], img_gt[:, :m]), (img_gen[:, m:], img_gt[:, m:])]
        else: # Top-Bottom
            m = h // 2
            blocks = [(img_gen[:m, :], img_gt[:m, :]), (img_gen[m:, :], img_gt[m:, :])]
        
        # Step 3: Recursive call for each block and average results
        sub_res = [Color_metrics_from_img_bgr(b_gen, b_gt, is_multi_block=False) for b_gen, b_gt in blocks]
        res = dict_mean(change_list2dict(sub_res))

    # Calculate unified precision and purity scores
    # res['d_pre_mean'] = (res['d_rgb_ed'] + res['d_lab_00']) / 2
    # res['d_pur_mean'] = (res['d_sd'] + res['d_hf']) / 2
    res['d_mean'] = (res['d_rgb_ed'] + res['d_lab_00'] + res['d_sd'] + res['d_hf'] + res['d_ced']) / 5
    return res

def Color_metrics_from_img_path(path_gen, path_gt, **kwargs):
    """Calculate color metrics from image paths."""
    img_gen = load_image(path_gen)
    img_gt = load_image(path_gt)
    return Color_metrics_from_img_bgr(img_gen, img_gt, **kwargs)

def Color_metrics_from_img_list(list_gen, list_gt, return_each_sample=False, **kwargs):
    """Calculate color metrics for a list of image paths."""
    if len(list_gen) != len(list_gt):
        raise ValueError("Length of image lists must match.")
    
    results = []
    for p_gen, p_gt in tqdm(zip(sorted(list_gen), sorted(list_gt)), desc="Evaluating Color"):
        results.append(Color_metrics_from_img_path(p_gen, p_gt, **kwargs))
    
    res_dict = change_list2dict(results)
    return res_dict if return_each_sample else dict_mean(res_dict)


def Color_metrics_from_img_list_no_equal(list_gen, list_gt, return_each_sample=False, **kwargs):
    """
    List processing function for calculating color metrics.
    Supports cases where list_gen is shorter than list_gt by automatically aligning filenames.
    """

    gt_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in list_gt}

    matched_pairs = []
    for p_gen in sorted(list_gen):
        file_name = os.path.splitext(os.path.basename(p_gen))[0]
        if file_name in gt_dict:
            matched_pairs.append((p_gen, gt_dict[file_name]))
    
    if not matched_pairs:
        return {} 

    results = []

    for p_gen, p_gt in tqdm(matched_pairs, desc="Evaluating Color Metrics"):
        res = Color_metrics_from_img_path(p_gen, p_gt, **kwargs)
        
        if res is not None:
            results.append(res)
    
    if not results:
        return {}

    # 4. 格式转换与返回逻辑
    res_dict = change_list2dict(results)
    return res_dict if return_each_sample else dict_mean(res_dict)



def Color_metrics_from_tensor(tensor_gen, tensor_gt, return_each_sample=False, **kwargs):
    """Calculate color metrics for BxCxHxW tensors."""
    B = tensor_gen.shape[0]
    results = []
    for i in range(B):
        img_gen = tensor2npBGR(tensor_gen[i])
        img_gt = tensor2npBGR(tensor_gt[i])
        results.append(Color_metrics_from_img_bgr(img_gen, img_gt, **kwargs))
    
    res_dict = change_list2dict(results)
    return res_dict if return_each_sample else dict_mean(res_dict)

