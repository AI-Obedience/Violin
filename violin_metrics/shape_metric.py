import cv2
import numpy as np
import torch
from tqdm import tqdm
import os


def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Cannot load image: {img_path}')
    return img

def get_binary(img, threshold=127):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary



def Metric_IoU(bin_gen, bin_gt):
    inter = np.logical_and(bin_gen, bin_gt).sum()
    union = np.logical_or(bin_gen, bin_gt).sum()
    return 1.0 - (inter / union) if union > 0 else 1.0

def Metric_Purity(img_bgr, bin_gt):
    mask = bin_gt > 0
    if not np.any(mask): return 0.0
    std = np.std(img_bgr[mask], axis=0).mean()
    return min(std / 128.0, 1.0)

def Metric_Dist_Normalized(cnt_gen, cnt_gt, shape_gen, shape_gt):
    M_gen = cv2.moments(cnt_gen)
    M_gt = cv2.moments(cnt_gt)
    
    h_gen, w_gen = shape_gen[:2]
    h_gt, w_gt = shape_gt[:2]

    if M_gen["m00"] == 0 or M_gt["m00"] == 0: 
        return 1.0
    
    # normalize
    nx_gen, ny_gen = (M_gen["m10"]/M_gen["m00"]) / w_gen, (M_gen["m01"]/M_gen["m00"]) / h_gen
    nx_gt, ny_gt = (M_gt["m10"]/M_gt["m00"]) / w_gt, (M_gt["m01"]/M_gt["m00"]) / h_gt
    

    dist = np.sqrt((nx_gen - nx_gt)**2 + (ny_gen - ny_gt)**2)
    return min(dist / np.sqrt(2), 1.0)

def Metric_Size_Normalized(cnt_gen, cnt_gt, shape_gen, shape_gt):
    area_gen_ratio = cv2.contourArea(cnt_gen) / (shape_gen[0] * shape_gen[1])
    area_gt_ratio = cv2.contourArea(cnt_gt) / (shape_gt[0] * shape_gt[1])
    
    if area_gt_ratio == 0: 
        return 1.0 if area_gen_ratio > 0 else 0.0
    return min(abs(area_gen_ratio / area_gt_ratio - 1.0), 1.0)


def Shape_metrics_from_img_bgr(img_gen, img_gt, threshold=127, return_mean=True):
    bin_gen_raw = get_binary(img_gen, threshold)
    contours_gen, _ = cv2.findContours(bin_gen_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bin_gt_raw = get_binary(img_gt, threshold)
    contours_gt, _ = cv2.findContours(bin_gt_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours_gen or not contours_gt:
        res = {k: 1.0 for k in ['d_iou', 'd_dist', 'd_size', 'd_shape']}
        res['d_purity'] = 0.0
    else:
        cnt_gen = max(contours_gen, key=cv2.contourArea)
        cnt_gt = max(contours_gt, key=cv2.contourArea)

        d_shape = min(cv2.matchShapes(cnt_gen, cnt_gt, cv2.CONTOURS_MATCH_I1, 0.0), 1.0)

        d_dist = Metric_Dist_Normalized(cnt_gen, cnt_gt, img_gen.shape, img_gt.shape)

        d_size = Metric_Size_Normalized(cnt_gen, cnt_gt, img_gen.shape, img_gt.shape)

        h_gt, w_gt = img_gt.shape[:2]

        mask_gen = np.zeros(img_gen.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_gen, [cnt_gen], -1, 255, -1)
        mask_gt = np.zeros(img_gt.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_gt, [cnt_gt], -1, 255, -1)
        
        mask_gen_resized = cv2.resize(mask_gen, (w_gt, h_gt), interpolation=cv2.INTER_NEAREST)
        d_iou = Metric_IoU(mask_gen_resized, mask_gt)

        mask_gen_final = np.zeros(img_gen.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_gen_final, [cnt_gen], -1, 255, -1)
        d_purity = Metric_Purity(img_gen, mask_gen_final)

        res = {
            'd_iou': d_iou,
            'd_dist': d_dist,
            'd_size': d_size,
            'd_shape': d_shape,
            'd_purity': d_purity
        }

    if return_mean:
        res['mean'] = sum(res.values()) / len(res)
    return res

def Shape_metrics_from_img_path(path_gen, path_gt, **kwargs):
    img_gen = load_image(path_gen)
    img_gt = load_image(path_gt)
    return Shape_metrics_from_img_bgr(img_gen, img_gt, **kwargs)

def change_list2dict(dicts):
    new_dict = dict()
    for key in dicts[0].keys():
        new_dict[key] = [d[key] for d in dicts]
    return new_dict

def dict_mean(dicts):
    new_dict = dict()
    for key in dicts.keys():
        new_dict[key] = sum(dicts[key])/len(dicts[key])
    return new_dict

def dict2tensor(dicts):
    new_dict = dict()
    for key in dicts.keys():
        new_dict[key] = torch.tensor(dicts[key])
    return new_dict


def Shape_metrics_from_img_list(list_gen, list_gt, return_each_sample=False, **kwargs):
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




def Shape_metrics_from_tensor(tensor1, tensor2, return_tensor=True, return_each_sample=False, **kwargs):
    if tensor1.shape != tensor2.shape: raise ValueError("Shape mismatch")
    B = tensor1.shape[0]
    res = []
    for idx in tqdm(range(B)):
        t1 = tensor2npBGR(tensor1[idx])
        t2 = tensor2npBGR(tensor2[idx])
        res.append(Shape_metrics_from_img_bgr(t1, t2, **kwargs))
    res = change_list2dict(res)
    if return_each_sample:
        return dict2tensor(res) if return_tensor else res
    else:
        return dict2tensor(dict_mean(res)) if return_tensor else dict_mean(res)