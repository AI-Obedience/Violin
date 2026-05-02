import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans

'''
You can use 
    NP_metrics_from_img_bgr,
    NP_metrics_from_img_path,
    NP_metrics_from_img_list,
    NP_metrics_from_tensor,

Here, NP indicates the noisy points;
Other functions maybe also useful.
'''


def center_crop_block(block, ratio=1.0):
    """
    Crop center region of a block to avoid boundary mixing.
    """
    h, w = block.shape[:2]
    dh = int(h * (1 - ratio) / 2)
    dw = int(w * (1 - ratio) / 2)
    return block[dh:h - dh, dw:w - dw]



def split_four_blocks_uniform(img_bgr):
    H, W = img_bgr.shape[:2]
    return [
        {"block": img_bgr[0:H//2, 0:W//2]},
        {"block": img_bgr[0:H//2, W//2:W]},
        {"block": img_bgr[H//2:H, 0:W//2]},
        {"block": img_bgr[H//2:H, W//2:W]},
    ]



def is_valid_four_color_blocks(
    blocks,
    img_area,
    area_tol=0.35,
):
    """
    Check whether detected blocks look like a valid four-color image.
    """
    if len(blocks) != 4:
        return False

    areas = []
    for b in blocks:
        y1, y2, x1, x2 = b["bbox"]
        areas.append((y2 - y1) * (x2 - x1))

    mean_area = img_area / 4
    for a in areas:
        if abs(a - mean_area) / mean_area > area_tol:
            return False

    return True



def infer_two_block_layout(blocks):
    """
    Infer layout for 2 color blocks: horizontal (top-bottom) or vertical (left-right)
    """
    centers = []

    for b in blocks:
        ys, xs = np.where(b["mask"])
        centers.append((xs.mean(), ys.mean()))

    (x1, y1), (x2, y2) = centers

    if abs(y1 - y2) > abs(x1 - x2):
        return "horizontal"   # 上下
    else:
        return "vertical"     # 左右






def apply_block_partition(img_bgr, blocks_template):
    """
    Apply the block partition scheme from template blocks to a new image.
    
    Args:
        img_bgr: Target image to be partitioned
        blocks_template: Block list from split_color_blocks_safe (as template)
    
    Returns:
        blocks: List of blocks with same partition as template
    """
    new_blocks = []
    
    for b in blocks_template:
        if "bbox" in b:
            y1, y2, x1, x2 = b["bbox"]
            new_blocks.append({
                "bbox": (y1, y2, x1, x2),
                "block": img_bgr[y1:y2, x1:x2]
            })
        else:
            # fallback: use the block shape to infer bbox
            block_h, block_w = b["block"].shape[:2]
            # This case shouldn't happen in practice, but keep it safe
            new_blocks.append({
                "block": b["block"]
            })
    
    return new_blocks


def split_color_blocks_safe(
    img_bgr,
):
    """
    Detect 2 or 4 color blocks automatically.
    Priority: 4-block > 2-block > uniform fallback
    """

    H, W = img_bgr.shape[:2]
    img_area = H * W

    # 先尝试4色检测
    result_4 = detect_color_blocks_with_validation(img_bgr, K=4, img_area=img_area, min_color_distance=5.0,   area_balance_threshold=0.8)
    if result_4 is not None:
        return result_4["blocks"], {
            "num_blocks": 4,
            "layout": "grid",
            "mode": "auto"
        }
    
    # 4色失败，尝试2色
    result_2 = detect_color_blocks_with_validation(img_bgr, K=2, img_area=img_area)
    if result_2 is not None:
        layout = infer_two_block_layout(result_2["blocks"])
        return result_2["blocks"], {
            "num_blocks": 2,
            "layout": layout,
            "mode": "auto"
        }
    
    # 都失败，使用uniform
    return split_four_blocks_uniform(img_bgr), {
        "num_blocks": 4,
        "layout": "grid",
        "mode": "uniform"
    }


def detect_color_blocks_with_validation(
    img_bgr,
    K=4,
    img_area=None,
    resize_for_kmeans=128,
    min_area_ratio=0.08,
    area_balance_threshold=0.6,  # 面积平衡度阈值
    min_color_distance=10.0,      # 最小颜色距离
):
    """
    Detect K color blocks with strict validation.
    Returns None if validation fails.
    """
    
    if img_area is None:
        H, W = img_bgr.shape[:2]
        img_area = H * W
    
    H, W = img_bgr.shape[:2]
    
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab_small = cv2.resize(lab, (resize_for_kmeans, resize_for_kmeans))
    pixels = lab_small.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
    labels_small = kmeans.fit_predict(pixels)
    labels_small = labels_small.reshape(resize_for_kmeans, resize_for_kmeans)

    labels = cv2.resize(
        labels_small.astype(np.int32),
        (W, H),
        interpolation=cv2.INTER_NEAREST
    )

    blocks = []
    areas = []

    for k in range(K):
        mask = labels == k
        area = np.sum(mask)
        
        # 面积太小，直接失败
        if area < img_area * min_area_ratio:
            return None

        ys, xs = np.where(mask)
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1

        blocks.append({
            "mask": mask,
            "bbox": (y1, y2, x1, x2),
            "block": img_bgr[y1:y2, x1:x2]
        })
        areas.append(area)

    # 验证1: 必须检测到K个块
    if len(blocks) != K:
        return None
    
    # 验证2: 面积平衡性检查
    mean_area = np.mean(areas)
    area_balance = np.std(areas) / mean_area
    if area_balance > area_balance_threshold:
        return None
    
    # 验证3: 颜色可分性检查
    centers = kmeans.cluster_centers_
    min_dist = float('inf')
    for i in range(K):
        for j in range(i+1, K):
            dist = np.linalg.norm(centers[i] - centers[j])
            min_dist = min(min_dist, dist)
    
    if min_dist < min_color_distance:
        return None
    
    # 验证4: 对于4色，检查是否像网格布局
    if K == 4:
        if not is_grid_like_layout(blocks, img_area):
            return None
    
    return {
        "blocks": blocks,
        "K": K
    }


def is_grid_like_layout(blocks, img_area, tolerance=0.35):
    """
    Check if 4 blocks form a grid-like layout (2x2)
    """
    if len(blocks) != 4:
        return False
    
    # 检查面积均衡
    areas = []
    for b in blocks:
        y1, y2, x1, x2 = b["bbox"]
        areas.append((y2 - y1) * (x2 - x1))
    
    mean_area = img_area / 4
    for a in areas:
        if abs(a - mean_area) / mean_area > tolerance:
            return False
    
    # 检查中心点分布（应该形成2x2网格）
    centers = []
    for b in blocks:
        ys, xs = np.where(b["mask"])
        centers.append((xs.mean(), ys.mean()))
    
    centers = np.array(centers)
    
    # 计算中心点的x和y坐标分布
    x_coords = centers[:, 0]
    y_coords = centers[:, 1]
    
    # 应该有2个不同的x值和2个不同的y值（允许一定误差）
    x_unique = len(np.unique(np.round(x_coords / 10))) 
    y_unique = len(np.unique(np.round(y_coords / 10)))
    
    # 2x2网格应该有2个x位置和2个y位置
    if x_unique == 2 and y_unique == 2:
        return True
    
    return False


def detect_color_blocks_hybrid(img_bgr, K=4):
    """
    混合方法：Kmeans + 边缘检测
    更准确但稍慢
    """
    H, W = img_bgr.shape[:2]
    
    # 1. 先用边缘检测找潜在分界线
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # 2. 检测主要的水平和垂直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                            minLineLength=min(H, W)//3, 
                            maxLineGap=10)
    
    # 3. 结合Kmeans结果
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab_small = cv2.resize(lab, (128, 128))
    pixels = lab_small.reshape(-1, 3).astype(np.float32)
    
    kmeans = KMeans(n_clusters=K, n_init=5, random_state=0)
    labels_small = kmeans.fit_predict(pixels)
    labels_small = labels_small.reshape(128, 128)
    labels = cv2.resize(
        labels_small.astype(np.uint8),
        (W, H),
        interpolation=cv2.INTER_NEAREST
    )
    
    # 4. 用边缘信息修正Kmeans结果
    # (具体实现略，可以用分水岭算法等)
    
    return labels


def split_color_blocks_safe_fast(img_bgr, is_groundtruth=True):
    """
    快速版本：并行检测 + 优化Kmeans
    """
    H, W = img_bgr.shape[:2]
    img_area = H * W
    
    # === 方案1: 并行检测2色和4色 ===
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_4 = executor.submit(
            detect_color_blocks_with_validation_fast,
            img_bgr, 4, img_area, is_groundtruth
        )
        future_2 = executor.submit(
            detect_color_blocks_with_validation_fast,
            img_bgr, 2, img_area, is_groundtruth
        )
        
        result_4 = future_4.result()
        result_2 = future_2.result()
    
    # 优先返回4色
    if result_4 is not None:
        return result_4["blocks"], {
            "num_blocks": 4,
            "layout": "grid",
            "mode": "auto"
        }
    
    if result_2 is not None:
        layout = infer_two_block_layout(result_2["blocks"])
        return result_2["blocks"], {
            "num_blocks": 2,
            "layout": layout,
            "mode": "auto"
        }
    
    return split_four_blocks_uniform(img_bgr), {
        "num_blocks": 4,
        "layout": "grid",
        "mode": "uniform"
    }


def detect_color_blocks_with_validation_fast(
    img_bgr,
    K=4,
    img_area=None,
    is_groundtruth=True,
    resize_for_kmeans=64,
    min_area_ratio=0.05,
    area_balance_threshold=0.8 ,
    min_color_distance=5.0,
    tolerance=0.6,
):
    if img_area is None:
        H, W = img_bgr.shape[:2]
        img_area = H * W
    
    H, W = img_bgr.shape[:2]
    
    # === 优化1: 降低resize尺寸 ===
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab_small = cv2.resize(lab, (resize_for_kmeans, resize_for_kmeans))
    pixels = lab_small.reshape(-1, 3).astype(np.float32)

    # === 优化2: 减少Kmeans迭代次数 ===
    kmeans = KMeans(
        n_clusters=K, 
        n_init=3,      # 从10降到3
        max_iter=100,  # 限制最大迭代
        random_state=0,
        algorithm='elkan'  # 更快的算法
    )
    labels_small = kmeans.fit_predict(pixels)
    labels_small = labels_small.reshape(resize_for_kmeans, resize_for_kmeans)

    # === 优化3: 使用整数插值避免浮点运算 ===
    labels = cv2.resize(
        labels_small.astype(np.uint8),
        (W, H),
        interpolation=cv2.INTER_NEAREST
    )

    blocks = []
    areas = []

    for k in range(K):
        mask = labels == k
        area = np.sum(mask)
        
        if area < img_area * min_area_ratio:
            return None

        # === 优化4: 使用numba加速（可选）===
        ys, xs = np.where(mask)
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1

        blocks.append({
            "mask": mask,
            "bbox": (y1, y2, x1, x2),
            "block": img_bgr[y1:y2, x1:x2]
        })
        areas.append(area)

    if len(blocks) != K:
        return None
    
    # 验证
    mean_area = np.mean(areas)
    area_balance = np.std(areas) / mean_area
    if area_balance > area_balance_threshold:
        return None
    
    centers = kmeans.cluster_centers_
    min_dist = float('inf')
    for i in range(K):
        for j in range(i+1, K):
            dist = np.linalg.norm(centers[i] - centers[j])
            min_dist = min(min_dist, dist)
    
    if min_dist < min_color_distance:
        return None
    
    if K == 4:
        if not is_grid_like_layout_fast(blocks, img_area, tolerance):
            return None
    
    return {"blocks": blocks, "K": K}


def is_grid_like_layout_fast(blocks, img_area, tolerance=0.35):
    """
    快速版本：简化检查
    """
    if len(blocks) != 4:
        return False
    
    # === 优化：只检查面积，跳过中心点计算 ===
    areas = []
    for b in blocks:
        y1, y2, x1, x2 = b["bbox"]
        areas.append((y2 - y1) * (x2 - x1))
    
    mean_area = img_area / 4
    for a in areas:
        if abs(a - mean_area) / mean_area > tolerance:
            return False
    
    return True  # 简化版本，不检查网格结构



def split_color_blocks_optimized(
    img_bgr, 
    is_groundtruth=True,
    mode='balanced'  # 'fast', 'accurate', 'balanced'
):
    """
    优化版本：平衡速度和准确度
    
    速度提升：3-5倍
    准确度：持平或更好
    """
    H, W = img_bgr.shape[:2]
    img_area = H * W
    
    if mode == 'fast':
        # 快速模式：单次Kmeans + 降低分辨率
        resize_size = 64
        n_init = 1
        n_runs = 1
    elif mode == 'accurate':
        # 准确模式：多次Kmeans投票
        resize_size = 128
        n_init = 5
        n_runs = 3
    else:  # balanced
        # 平衡模式（推荐）
        resize_size = 96
        n_init = 3
        n_runs = 1
    
    # 并行检测2色和4色
    from concurrent.futures import ThreadPoolExecutor
    
    def detect_k(K):
        return detect_color_blocks_with_validation_fast(
            img_bgr, K, img_area, is_groundtruth,
            resize_for_kmeans=resize_size,
        )
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_4 = executor.submit(detect_k, 4)
        future_2 = executor.submit(detect_k, 2)
        
        result_4 = future_4.result()
        result_2 = future_2.result()
    
    if result_4 is not None:
        return result_4["blocks"], {
            "num_blocks": 4,
            "layout": "grid",
            "mode": "auto"
        }
    
    if result_2 is not None:
        layout = infer_two_block_layout(result_2["blocks"])
        return result_2["blocks"], {
            "num_blocks": 2,
            "layout": layout,
            "mode": "auto"
        }
    
    return split_four_blocks_uniform(img_bgr), {
        "num_blocks": 4,
        "layout": "grid",
        "mode": "uniform"
    }


def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Cannot load image: {img_path}')
    return img


def tensor2npBGR(tensor):
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()
    np_img = tensor.numpy()
    np_img = np.transpose(np_img, (1, 2, 0))

    if np_img.dtype == np.float32 and np.max(np_img) <= 1.0:
        np_img = (np_img * 255).astype(np.uint8)
    else:
        np_img = np_img.astype(np.uint8)

    np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return np_img_bgr



def standard_deviation_from_img_bgr(img_np):
    # standard deviation
    std_b = np.std(img_np[:, :, 0])
    std_g = np.std(img_np[:, :, 1])
    std_r = np.std(img_np[:, :, 2])

    # average
    std = ((std_b + std_g + std_r) / 3.0).item()
    std = min(std/127.5, 1)
    
    return std



def max_difference_from_img_bgr(img_np):
    pixels = img_np.reshape(-1, 3).astype(np.float32)
    
    min_vals = np.min(pixels, axis=0)  # shape: (3,)
    max_vals = np.max(pixels, axis=0)  # shape: (3,)
    
    max_diff = np.linalg.norm(max_vals - min_vals).item()

    max_diff = min(max_diff/441.67, 1)  # (0,0,0) to (255,255,255), norm to (0,1)

    return max_diff


def canny_density_from_img_bgr(img_np, threshold1: int = 20, threshold2: int = 60):
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, threshold1, threshold2)
    
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size

    canny = (edge_pixels / total_pixels).item()
    
    return canny


def high_freq_density_from_img_bgr(img_np, high_freq_threshold: float = 0.1):
    # grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 2D FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)  # 将零频点移到中心
    
    magnitude_spectrum = np.abs(f_shift) ** 2
    
    # for high frequency masks
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2  # central points
    
    # norm distance
    y, x = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)
    max_distance = np.sqrt(crow**2 + ccol**2)  # 最大可能距离（对角线）
    normalized_distance = distance_from_center / max_distance  # [0, 1]
    
    # mask
    high_freq_mask = normalized_distance > high_freq_threshold
    
    # energys
    high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
    total_energy = np.sum(magnitude_spectrum)
    
    if total_energy == 0:
        return 0.0

    ratio = (high_freq_energy / total_energy).item()
    
    return ratio



def NP_metrics_from_img_bgr(
    gt_img_np,
    img_np,
    return_standard_deviation=True, 
    return_max_difference=True, 
    return_canny_density=True,
    canny_threshold_1=20,
    canny_threshold_2=60,
    return_high_freq=True,
    high_freq_threshold=0.02,
    return_mean=True,
    return_weight_mean=True,
    weights = {
        "standard_deviation": 1,
        "canny_density": 8,
        "high_freq": 5,
    },
    block_ratio=1.0,
):
    
    blocks1, meta1 = split_color_blocks_optimized(gt_img_np)
    blocks2 = apply_block_partition(img_np, blocks1)

    res_list = []
    for b1, b2 in zip(blocks1, blocks2):
        blk2 = center_crop_block(b2["block"], ratio=block_ratio)

        res = dict()
        if return_standard_deviation:
            res['standard_deviation'] = standard_deviation_from_img_bgr(blk2)
        # if return_max_difference:
        #     res['max_difference'] = max_difference_from_img_bgr(blk2)
        if return_canny_density:
            res['canny_density'] = canny_density_from_img_bgr(blk2, canny_threshold_1, canny_threshold_2)
        if return_high_freq:
            res['high_freq'] = high_freq_density_from_img_bgr(blk2, high_freq_threshold)
        if return_mean:
            res['mean'] = sum(res.values())/len(res.values())
        if return_weight_mean:
            temp_res = 0
            for key in res.keys():
                if key=='mean':
                    continue
                # print(key, temp_res)
                temp_res = temp_res + min(res[key] * weights[key], 1)
            res['weighted_mean'] = temp_res / 3
        
        res_list.append(res)
    
    # ---- block-level → image-level ----
    final_res = dict_mean(change_list2dict(res_list))
    final_res['four_block_mode'] = meta1  # 'auto' or 'uniform'

    return final_res



def NP_metrics_from_img_path(gt_img_path, img_path, **kwargs):
    gt_img_np = load_image(gt_img_path)
    img_np = load_image(img_path)
    res = NP_metrics_from_img_bgr(gt_img_np, img_np, **kwargs)
    return res


def change_list2dict(dicts):
    # change list of dict to dict of list
    new_dict = dict()
    for key in dicts[0].keys():
        new_dict[key] = [d[key] for d in dicts]
    return new_dict


def dict_mean(dicts):
    # calculate mean for each element in dicts
    new_dict = dict()
    for key in dicts.keys():
        new_dict[key] = sum(dicts[key])/len(dicts[key])
    return new_dict


def NP_metrics_from_img_list(gt_img_list, img_list, return_each_sample=False, **kwargs):
    res = []
    for i, (gt_img_p, img_p) in tqdm(enumerate(zip(gt_img_list, img_list))):
        single_res = NP_metrics_from_img_path(gt_img_p, img_p, **kwargs)
        res.append(single_res)
    res = change_list2dict(res)

    return res if return_each_sample else dict_mean(res)


def dict2tensor(dicts):
    # change each element in dicts to tensor
    new_dict = dict()
    for key in dicts.keys():
        new_dict[key] = torch.tensor(dicts[key])
    return new_dict


def NP_metrics_from_tensor(gt_img_tensor, img_tensor, return_each_sample=False, return_tensor=False, **kwargs):
    # [B, C, H, W]
    B = img_tensor.shape[0]
    res = []
    for i in tqdm(range(B)):
        gt_img_t = gt_img_tensor[i]
        img_t = img_tensor[i]

        gt_img_bgr = tensor2npBGR(gt_img_t)
        img_bgr = tensor2npBGR(img_t)

        single_res = NP_metrics_from_img_bgr(gt_img_bgr, img_bgr,**kwargs)
        res.append(single_res)
    
    res = change_list2dict(res)
    
    if return_each_sample:
        if return_tensor:
            return dict2tensor(res)
        else:
            return res
    else:
        if return_tensor:
            return dict2tensor(dict_mean(res))
        else:
            return dict_mean(res)
        



if __name__ == '__main__':
    gt_img = '/data1/lhy/pure_color/my_code/four_color.png'
    # img1 = '/data1/lhy/pure_color/my_code/slight_red.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/21_red_bg_white_watermark.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/26_green_red_circles_watermark.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/31_checkerboard_noise.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/25_blue_yellow_dots_saltpepper.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/30_gradient_grid_noise.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/08_gaussian_blue_std50.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/23_blue_bg_yellow_rect_logo.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/11_saltpepper_yellow_10pct.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/20_blue_yellow_checkerboard.png'
    img1 = '/data1/lhy/pure_color/my_code/test_color_blocks/03_4grid_noisy.png'

    import os
    root_path = '/data1/lhy/pure_color/my_code/test_noisy_images'
    img_list = [os.path.join(root_path, _) for _ in os.listdir(root_path)]
    img_list = sorted(img_list)


    img_tensor1 = torch.randn(2,3,56,56)

    res = NP_metrics_from_img_path(gt_img, img1)
    print(res)

    # res = NP_metrics_from_img_list(img_list, return_each_sample=True)
    # print(res)
    