import cv2
import numpy as np
import torch
from tqdm import tqdm
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

if current_dir not in sys.path:
    sys.path.append(current_dir)
from shape_metric import load_image, change_list2dict, dict_mean, dict2tensor, Metric_IoU, Metric_Dist_Normalized



"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'img_gt' (or 'path_gt' or 'tensor2') MUST be the Ground Truth. 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

You can use:
    - Mask_metrics_from_img_bgr
    - Mask_metrics_from_img_path
    - Mask_metrics_from_img_list
    - Mask_metrics_from_tensor
"""


def get_mask_binary(img, threshold=127):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary

# --- Mask Specific Metrics ---
def Metric_Boundary_IoU(bin_gen, bin_gt, d=5):
    """
    A more standard implementation: compare only the overlap within a d-pixel distance of the edge.
    """
    def get_boundary(mask, d):
        # This method is more robust than contour finding; it extracts a band-shaped area 5 pixels inward from the shape edge.
        kernel = np.ones((3, 3), np.uint8)
        mask_eroded = cv2.erode(mask, kernel, iterations=d)
        return cv2.bitwise_xor(mask, mask_eroded)

    
    bound_gt = get_boundary(bin_gt, d)
    bound_gen = get_boundary(bin_gen, d)

    # Calculate IoU in these two edge bands
    inter = np.logical_and(bound_gen, bound_gt).sum()
    union = np.logical_or(bound_gen, bound_gt).sum()
    
    return 1.0 - (inter / union) if union > 0 else 0.0



def Metric_Leak(img_gen_bgr, bin_gt_filled):
    # Erode 3 pixels, exclude the edge anti-aliasing area, and only look at the inside of the mask core.
    kernel = np.ones((3, 3), np.uint8)
    inner_mask = cv2.erode(bin_gt_filled, kernel, iterations=3)
    
    mask = inner_mask > 0
    if not np.any(mask): 
        mask = bin_gt_filled > 0
        if not np.any(mask): return 0.0
    
    gray_gen = cv2.cvtColor(img_gen_bgr, cv2.COLOR_BGR2GRAY)
    actual_brightness = gray_gen[mask].mean()
    return actual_brightness / 255.0



def Metric_Mask_Edge(img_gen_bgr, img_gt_bgr, bin_gt_filled):
    gray_gen = cv2.cvtColor(img_gen_bgr, cv2.COLOR_BGR2GRAY)
    gray_gt = cv2.cvtColor(img_gt_bgr, cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((3,3), np.uint8)
    edge_mask = cv2.Canny(bin_gt_filled, 100, 200)
    edge_zone = cv2.dilate(edge_mask, kernel, iterations=2)
    
    if np.count_nonzero(edge_zone) == 0: return 0.0

    # gradient computation function
    def get_avg_grad(img_gray, mask):
        sx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(sx**2 + sy**2)
        return grad[mask > 0].mean()

    # Calculate the average gradient magnitude in the edge zone for both GT and Gen
    gt_grad = get_avg_grad(gray_gt, edge_zone)
    gen_grad = get_avg_grad(gray_gen, edge_zone)

    # If GT edge is very weak, we can't really evaluate edge quality, so we default to 0.
    if gt_grad < 1.0: return 0.0

    score = 1.0 - (gen_grad / gt_grad)
    
    return max(0.0, min(1.0, score))


# --- Wrapper Functions ---
def Mask_metrics_from_img_bgr(img_gen, img_gt, threshold=30, return_mean=True):
    # 1. Extract the mask from the generated image (since the mask is black, use THRESH_BINARY_INV)
    gray_gen = cv2.cvtColor(img_gen, cv2.COLOR_BGR2GRAY)
    # THRESH_BINARY_INV will make the black mask white (255) and the rest black (0)
    _, bin_gen_raw = cv2.threshold(gray_gen, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Fill any noise holes inside the mask to ensure a complete geometric shape when calculating IoU
    contours_gen, _ = cv2.findContours(bin_gen_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bin_gen_filled = np.zeros_like(bin_gen_raw)
    if contours_gen:
        max_cnt_gen = max(contours_gen, key=cv2.contourArea)
        cv2.drawContours(bin_gen_filled, [max_cnt_gen], -1, 255, -1)
    
    # 2. Extract the GT mask (same needs to be inverted)
    gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
    _, bin_gt_raw = cv2.threshold(gray_gt, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # GT also needs filling to be safe (although GT is usually clean)
    contours_gt, _ = cv2.findContours(bin_gt_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bin_gt_filled = np.zeros_like(bin_gt_raw)
    if contours_gt:
        max_cnt_gt = max(contours_gt, key=cv2.contourArea)
        cv2.drawContours(bin_gt_filled, [max_cnt_gt], -1, 255, -1)
    
    # 3. Calculate metrics
    res = {
        'd_iou': Metric_IoU(bin_gen_filled, bin_gt_filled),
        'd_dist': Metric_Dist(bin_gen_filled, bin_gt_filled),
        'd_biou': Metric_Boundary_IoU(bin_gen_filled, bin_gt_filled),
        
        'd_leak': Metric_Leak(img_gen, bin_gt_filled), 
        
        'd_edge': Metric_Mask_Edge(img_gen, img_gt, bin_gt_filled)
    }
    
    if return_mean:
        res['mean'] = sum(res.values()) / len(res)
    return res


def Mask_metrics_from_img_path(path_gen, path_gt, **kwargs):
    img_gen = load_image(path_gen)
    img_gt = load_image(path_gt)
    return Mask_metrics_from_img_bgr(img_gen, img_gt, **kwargs)


def Mask_metrics_from_img_path_scale(path_gen, path_gt, **kwargs):
    img_gen = load_image(path_gen)
    img_gt = load_image(path_gt)

    h,w = 512,512

    if img_gen.shape[0] != h or img_gen.shape[1] != w:
        img_gen = cv2.resize(img_gen, (w, h), interpolation=cv2.INTER_CUBIC)
    
    return Mask_metrics_from_img_bgr(img_gen, img_gt, **kwargs)



def Mask_metrics_from_img_list(list_gen, list_gt, rescale_generated_image=False, return_each_sample=False, **kwargs):
    if len(list_gen) != len(list_gt): raise ValueError("List length mismatch")
    res = []
    for p_gen, p_gt in tqdm(zip(sorted(list_gen), sorted(list_gt)), total=len(list_gen)):
        if rescale_generated_image:
            res.append(Mask_metrics_from_img_path_scale(p_gen, p_gt, **kwargs))
        else:
            res.append(Mask_metrics_from_img_path(p_gen, p_gt, **kwargs))
    res = change_list2dict(res)
    return res if return_each_sample else dict_mean(res)


def Mask_metrics_from_img_list_non_equal(list_gen, list_gt, rescale_generated_image=False, return_each_sample=False, **kwargs):
    gt_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in list_gt}
    
    matched_pairs = []
    for p_gen in sorted(list_gen):
        file_name = os.path.splitext(os.path.basename(p_gen))[0]
        if file_name in gt_dict:
            matched_pairs.append((p_gen, gt_dict[file_name]))
    
    if not matched_pairs:
        return {} 

    res = []

    for p_gen, p_gt in tqdm(matched_pairs, desc="Calculating Mask Metrics"):
        if rescale_generated_image:
            res.append(Mask_metrics_from_img_path_scale(p_gen, p_gt, **kwargs))
        else:
            res.append(Mask_metrics_from_img_path(p_gen, p_gt, **kwargs))
    

    res = change_list2dict(res)
    return res if return_each_sample else dict_mean(res)



def Mask_metrics_from_tensor(tensor1, tensor2, return_tensor=True, return_each_sample=False, **kwargs):
    if tensor1.shape != tensor2.shape: raise ValueError("Shape mismatch")
    B = tensor1.shape[0]
    res = []
    for idx in tqdm(range(B)):
        t1 = tensor2npBGR(tensor1[idx])
        t2 = tensor2npBGR(tensor2[idx])
        res.append(Mask_metrics_from_img_bgr(t1, t2, **kwargs))
    res = change_list2dict(res)
    if return_each_sample:
        return dict2tensor(res) if return_tensor else res
    else:
        return dict2tensor(dict_mean(res)) if return_tensor else dict_mean(res)
    