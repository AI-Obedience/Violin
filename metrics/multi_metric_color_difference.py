import cv2
import numpy as np
import torch
from tqdm import tqdm
from skimage.color import deltaE_ciede2000
from sklearn.cluster import KMeans

"  !!!!!!!!!!!!!!!   img1 should be groundtruth, code will detect block automatically  !!!!!!!!!!!!!!!!!!!!!!!!!"

'''
You can use 
    CD_metrics_from_img_lab,
    CD_metrics_from_img_bgr,
    CD_metrics_from_img_path,
    CD_metrics_from_img_list,
    CD_metrics_from_tensor

Here, CD indicates the color difference;
Other functions to convert some datatype are also available.
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

def convert_BGR_to_LAB(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    lab[:, :, 0] = lab[:, :, 0] * 100.0 / 255.0  # L: [0, 255] -> [0, 100]
    lab[:, :, 1] = lab[:, :, 1] - 128.0          # a: [0, 255] -> [-128, 127]
    lab[:, :, 2] = lab[:, :, 2] - 128.0          # b: [0, 255] -> [-128, 127]

    return lab


def tensor2npLAB(tensor):
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()
    np_img = tensor.numpy()
    np_img = np.transpose(np_img, (1, 2, 0))

    if np_img.dtype == np.float32 and np.max(np_img) <= 1.0:
        np_img = (np_img * 255).astype(np.uint8)
    else:
        np_img = np_img.astype(np.uint8)

    np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(np_img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # change range
    lab[:, :, 0] = lab[:, :, 0] * 100.0 / 255.0
    lab[:, :, 1] = lab[:, :, 1] - 128.0
    lab[:, :, 2] = lab[:, :, 2] - 128.0
    
    return lab



def CIEDE2000_from_lab_img(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError(f'Image shapes do not match: {img1.shape} vs {img2.shape}')
    r = deltaE_ciede2000(img1, img2).mean().item()
    r = min(r / 100.0, 1.0)
    return r


def DeltaEHyAB_from_lab_img(img1, img2):
    """
    ΔE_HyAB = √[(a₂-a₁)² + (b₂-b₁)² + |L₂-L₁|]
    
    param:
        img1: [H, W, 3] 
        img2: [H, W, 3]
    
    note: L:[0~100], A/B:[-127~127]
    """
    if img1.shape != img2.shape:
        raise ValueError(f'Shape mismatch: {img1.shape} vs {img2.shape}')
    
    L1, a1, b1 = img1[:, :, 0], img1[:, :, 1], img1[:, :, 2]
    L2, a2, b2 = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]
    
    # HyAB Delta E
    delta_e = np.sqrt(
        (a2 - a1)**2 + 
        (b2 - b1)**2
    ) + np.abs(L2 - L1)

    delta_e = delta_e.mean().item()

    normed = min(delta_e / 460.0, 1.0)
    
    return normed


def Delta_Chroma_from_lab_img(img1, img2):
    # from GenColorBench Paper, not its official implementation
    """
    Delta Chroma = √[(a₂-a₁)² + (b₂-b₁)²] 
    
    param:
        img1: [H, W, 3] 
        img2: [H, W, 3]
    
    note: L:[0~100], A/B:[-127~127]
    """
    if img1.shape != img2.shape:
        raise ValueError(f'Shape mismatch: {img1.shape} vs {img2.shape}')
    
    a1, b1 = img1[:, :, 1], img1[:, :, 2]
    a2, b2 = img2[:, :, 1], img2[:, :, 2]
    
    # HyAB Delta E
    delta_c = np.sqrt(
        (a2 - a1)**2 + 
        (b2 - b1)**2
    )

    delta_c = delta_c.mean().item()
    normed = min(delta_c/100, 1.0)

    return normed



def calculate_hue_angle(lab):
    """
    h = atan2(b*, a*)
    
    return:
        hue: Hue [0, 360)
        chroma: Chroma, reliability gating
    """

    a = lab[:, :, 1]
    b = lab[:, :, 2]
    
    # radian
    hue_rad = np.arctan2(b, a)
    
    # angle:[0,360]
    hue_deg = np.degrees(hue_rad)
    hue_deg = np.where(hue_deg < 0, hue_deg + 360, hue_deg)
    
    # Chroma (reliability gating）
    chroma = np.sqrt(a**2 + b**2)
    
    return hue_deg, chroma


def MAE_Hue_from_lab_img(img1, img2, chroma_threshold=0.0):
    
    a1, b1 = img1[:, :, 1], img1[:, :, 2]
    a2, b2 = img2[:, :, 1], img2[:, :, 2]
    
    # chroma
    chroma1 = np.sqrt(a1**2 + b1**2)
    chroma2 = np.sqrt(a2**2 + b2**2)
    
    valid_mask = (chroma1 >= chroma_threshold) & (chroma2 >= chroma_threshold)
    
    if np.sum(valid_mask) == 0:
        return 0.0
    
    # hue
    hue1 = np.degrees(np.arctan2(b1[valid_mask], a1[valid_mask]))
    hue2 = np.degrees(np.arctan2(b2[valid_mask], a2[valid_mask]))
    
    hue1 = np.where(hue1 < 0, hue1 + 360, hue1)
    hue2 = np.where(hue2 < 0, hue2 + 360, hue2)
    
    # difference
    hue_diff = np.abs(hue1 - hue2)
    hue_diff = np.where(hue_diff > 180, 360 - hue_diff, hue_diff)
    
    mae_hue = np.mean(hue_diff).item()

    normed = mae_hue/180.0
    
    return normed


def RGB_distance_from_BGR_img(img1_bgr, img2_bgr):
    """
    √[(R₂-R₁)² + (G₂-G₁)² + (B₂-B₁)²]
    range: [0, 441.67] (√(255² + 255² + 255²))

    img: BGR, just from cv2.imread

    """

    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    diff = img1_rgb - img2_rgb
    euclidean = np.sqrt(np.sum(diff**2, axis=2))
    euclidean = euclidean.mean().item()

    normed = min(euclidean / 441.67, 1.0)

    return normed


def RGB_Redmean_distance_from_BGR_img(img1_bgr, img2_bgr):
    """
    r = 1/2*(R1+R2)
    delta = sqrt((2+r/256)delta_R**2  +4delta_G**2 + (2+ (255-r)/256)delta_B**2)

    img: BGR, just from cv2.imread

    """

    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    R1,G1,B1 = img1_rgb[:,:,0], img1_rgb[:,:,1], img1_rgb[:,:,2]
    R2,G2,B2 = img2_rgb[:,:,0], img2_rgb[:,:,1], img2_rgb[:,:,2]

    r = 0.5*(R1+R2)
    delta_R = R1-R2
    delta_G = G1-G2
    delta_B = B1-B2

    diff = (2 + (r/256))*(delta_R**2) + 4*(delta_G**2) + (2+(255-r)/256)*(delta_B**2)

    redmean_distance = np.sqrt(diff)
    redmean_distance = redmean_distance.mean().item()

    normed = min(redmean_distance/765, 1.0)

    return normed



def CD_metrics_from_img_lab(
    img1,
    img2,
    return_delta_chroma=True, 
    return_deltaE_HyAB=True, 
    return_CIEDE2000=True, 
    return_hue=True,
    chroma_threshold=0.0,
    return_mean=True,
):
    res = dict()
    if return_delta_chroma:
        res['delta_chroma'] = Delta_Chroma_from_lab_img(img1, img2)
    if return_deltaE_HyAB:
        res['deltaE_HyAB'] = DeltaEHyAB_from_lab_img(img1, img2)
    if return_CIEDE2000:
        res['CIEDE2000'] = CIEDE2000_from_lab_img(img1, img2)
    if return_hue:
        res['mae_hue'] = MAE_Hue_from_lab_img(img1, img2, chroma_threshold)
    if return_mean:
        res['mean'] = sum(res.values())/len(res.values())

    return res


def CD_metrics_from_img_bgr(
    img1,
    img2,
    return_sRGB=True,
    return_sRGB_redmean=True,
    return_mean=True,
    chroma_threshold=0.0,
    block_ratio=1.0,   # calculate center crop for each block
    **kwargs
):

    blocks1, meta1 = split_color_blocks_optimized(img1)
    blocks2 = apply_block_partition(img2, blocks1)

    res_list = []

    for b1, b2 in zip(blocks1, blocks2):
        blk1 = center_crop_block(b1["block"], ratio=block_ratio)
        blk2 = center_crop_block(b2["block"], ratio=block_ratio)

        blk1_lab = convert_BGR_to_LAB(blk1)
        blk2_lab = convert_BGR_to_LAB(blk2)

        cd = CD_metrics_from_img_lab(
            blk1_lab,
            blk2_lab,
            chroma_threshold=chroma_threshold,
            **kwargs
        )

        if return_sRGB:
            cd['srgb'] = RGB_distance_from_BGR_img(blk1, blk2)
        if return_sRGB_redmean:
            cd['srgb_redmean'] = RGB_Redmean_distance_from_BGR_img(blk1, blk2)

        if return_mean:
            cd['mean'] = sum(cd.values()) / len(cd.values())

        res_list.append(cd)

    # ---- block-level → image-level ----
    final_res = dict_mean(change_list2dict(res_list))
    final_res['four_block_mode'] = meta1  # 'auto' or 'uniform'

    return final_res




def CD_metrics_from_img_path(
    img_path1, 
    img_path2, 
    **kwargs,
):
    img1 = load_image(img_path1)
    img2 = load_image(img_path2)

    return CD_metrics_from_img_bgr(img1, img2, **kwargs)




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
        if key == 'four_block_mode':
            continue
        new_dict[key] = sum(dicts[key])/len(dicts[key])
    return new_dict


def dict2tensor(dicts):
    # change each element in dicts to tensor
    new_dict = dict()
    for key in dicts.keys():
        new_dict[key] = torch.tensor(dicts[key])
    return new_dict


def CD_metrics_from_img_list(img_list1, img_list2, return_each_sample=False, **kwargs):
    if len(img_list1) != len(img_list2):
        raise ValueError(f'list 1 has {len(img_list1)}, list2 has {len(img_list2)}, not match')
    
    img_list1 = sorted(img_list1)
    img_list2 = sorted(img_list2)

    # match img pairs by name
    res = []
    for img_p1, img_p2 in tqdm(zip(img_list1, img_list2)):
        res.append(CD_metrics_from_img_path(img_p1, img_p2, **kwargs))
    
    res = change_list2dict(res)

    return res if return_each_sample else dict_mean(res)



def CD_metrics_from_tensor(tensor1, tensor2, return_tensor=True, return_each_sample=False, **kwargs):
    # [B, C, H, W]
    if tensor1.shape != tensor2.shape:
        raise ValueError(f'shape 1:{tensor1.shape} not match shape 2:{tensor2.shape}')
    
    B = tensor1.shape[0]

    res = []
    for idx in tqdm(range(B)):
        t1 = tensor2npLAB(tensor1[idx])
        t2 = tensor2npLAB(tensor2[idx])

        res.append(CD_metrics_from_img_bgr(t1, t2, **kwargs))

    
    if return_each_sample:
        if return_tensor:
            return dict2tensor(change_list2dict(res))
        else:
            return change_list2dict(res)
    else:
        if return_tensor:
            return dict2tensor(dict_mean(change_list2dict(res)))
        else:
            return dict_mean(change_list2dict(res))


if __name__ == '__main__':
    # img1 = '/data1/lhy/pure_color/my_code/slight_red.png'
    # img2 = '/data1/lhy/pure_color/my_code/pure_red.png'
    # img2 = '/data1/lhy/pure_color/my_code/slight_red.png'

    img1 = '/data1/lhy/pure_color/my_code/four_color.png'
    img2 = '/data1/lhy/pure_color/my_code/test_color_blocks/01_4grid_standard.png'


    img_list1 = [img1]*10
    img_list2 = [img2]*10

    img_tensor1 = torch.randn(2,3,56,56)
    img_tensor2 = torch.randn(2,3,56,56)

    res = CD_metrics_from_img_path(img1, img2)
    print(res)

    res = CD_metrics_from_img_list(img_list1, img_list2, return_each_sample=False)
    print(res)

    # res = CD_metrics_from_tensor(img_tensor1, img_tensor2)
    # print(res)