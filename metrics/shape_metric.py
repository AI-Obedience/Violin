import cv2
import numpy as np
import torch
from tqdm import tqdm

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'img_gt' (or 'path_gt' or 'tensor2') MUST be the Ground Truth. 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

You can use:
    - Shape_metrics_from_img_bgr
    - Shape_metrics_from_img_path
    - Shape_metrics_from_img_list
    - Shape_metrics_from_tensor
"""

def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Cannot load image: {img_path}')
    return img

def tensor2npBGR(tensor):
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()
    np_img = tensor.detach().numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    if np_img.dtype == np.float32:
        np_img = (np_img * 255).clip(0, 255).astype(np.uint8)
    else:
        np_img = np_img.astype(np.uint8)
    # Assuming the tensor is RGB, convert it to BGR for use by OpenCV.
    np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return np_img_bgr

def get_binary(img, threshold=127):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

# --- Metric Functions ---

def Metric_IoU(bin_gen, bin_gt):
    inter = np.logical_and(bin_gen, bin_gt).sum()
    union = np.logical_or(bin_gen, bin_gt).sum()
    return 1.0 - (inter / union) if union > 0 else 0.0

def Metric_Dist(bin_gen, bin_gt):
    M_gen = cv2.moments(bin_gen)
    M_gt = cv2.moments(bin_gt)
    h, w = bin_gt.shape
    diag = np.sqrt(h**2 + w**2)
    if M_gen["m00"] == 0 or M_gt["m00"] == 0: return 1.0
    cx_gen, cy_gen = M_gen["m10"]/M_gen["m00"], M_gen["m01"]/M_gen["m00"]
    cx_gt, cy_gt = M_gt["m10"]/M_gt["m00"], M_gt["m01"]/M_gt["m00"]
    dist = np.sqrt((cx_gen - cx_gt)**2 + (cy_gen - cy_gt)**2)
    return min(dist / diag, 1.0)

def Metric_Size(bin_gen, bin_gt):
    area_gen = np.count_nonzero(bin_gen)
    area_gt = np.count_nonzero(bin_gt)
    if area_gt == 0: return 1.0 if area_gen > 0 else 0.0
    return min(abs(area_gen / area_gt - 1.0), 1.0)

def Metric_Shape(bin_gen, bin_gt):
    score = cv2.matchShapes(bin_gen, bin_gt, cv2.CONTOURS_MATCH_I1, 0.0)
    return min(score, 1.0)

def Metric_Purity(img_bgr, bin_gt):
    mask = bin_gt > 0
    if not np.any(mask): return 0.0
    std = np.std(img_bgr[mask], axis=0).mean()
    return min(std / 128.0, 1.0)

# --- Wrapper Functions ---
def Shape_metrics_from_img_bgr(img_gen, img_gt, threshold=127, return_mean=True):
    # 1. Extract the solid outline of the generated image (for geometric indices)
    gray_gen = cv2.cvtColor(img_gen, cv2.COLOR_BGR2GRAY)
    _, bin_gen = cv2.threshold(gray_gen, threshold, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(bin_gen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bin_gen_filled = np.zeros_like(bin_gen)
    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(bin_gen_filled, [max_cnt], -1, 255, -1)
    
    bin_gt = get_binary(img_gt, threshold)
    
    res = {
        'd_iou': Metric_IoU(bin_gen_filled, bin_gt),
        'd_dist': Metric_Dist(bin_gen_filled, bin_gt),
        'd_size': Metric_Size(bin_gen_filled, bin_gt),
        'd_shape': Metric_Shape(bin_gen_filled, bin_gt),
        # Modification: Calculate purity within the area of ​​objects drawn in the generated image
        # If bin_gen_filled is completely black, then Purity defaults to 0 (nothing is present, so there's no question of impurity).
        'd_purity': Metric_Purity(img_gen, bin_gen_filled) 
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
    if len(list_gen) != len(list_gt): raise ValueError("List length mismatch")
    res = []
    for p_gen, p_gt in tqdm(zip(sorted(list_gen), sorted(list_gt)), total=len(list_gen)):
        res.append(Shape_metrics_from_img_path(p_gen, p_gt, **kwargs))
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


if __name__ == '__main__':
    # img1 = 'VIOLIN_v2\data\Variation_3\id_1.png'
    # # img2 = '/data1/lhy/pure_color/my_code/pure_red.png'
    # img2 = 'VIOLIN_v2\data\Variation_3\id_2.png'


    # img_list1 = [img1]*10
    # img_list2 = [img2]*10

    # img_tensor1 = torch.randn(2,3,56,56)
    # img_tensor2 = torch.randn(2,3,56,56)

    # res = Shape_metrics_from_img_path(img1, img2)
    # print(res)

    gt = 'metrics_v2/test_cases/gt.png'
    cases = ['case_dist.png', 'case_size.png', 'case_shape.png', 'case_purity.png']

    for c in cases:
        path = f'metrics_v2/test_cases/{c}'
        res = Shape_metrics_from_img_path(path, gt)
        print(f"Results for {c}:")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}")
        print("-" * 20)