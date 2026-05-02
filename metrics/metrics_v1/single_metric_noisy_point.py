import cv2
import numpy as np
import torch
from tqdm import tqdm

'''
You can use 
    NP_metrics_from_img_bgr,
    NP_metrics_from_img_path,
    NP_metrics_from_img_list,
    NP_metrics_from_tensor,

Here, NP indicates the noisy points;
Other functions maybe also useful.
'''


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
    }
):
    res = dict()
    if return_standard_deviation:
        res['standard_deviation'] = standard_deviation_from_img_bgr(img_np)
    # if return_max_difference:
    #     res['max_difference'] = max_difference_from_img_bgr(img_np)
    if return_canny_density:
        res['canny_density'] = canny_density_from_img_bgr(img_np, canny_threshold_1, canny_threshold_2)
    if return_high_freq:
        res['high_freq'] = high_freq_density_from_img_bgr(img_np, high_freq_threshold)
    if return_mean:
        res['mean'] = sum(res.values())/len(res.values())
    if return_weight_mean:
        temp_res = 0
        for key in res.keys():
            if key=='mean':
                continue
            print(key, temp_res)
            temp_res = temp_res + min(res[key] * weights[key], 1)
        res['weighted_mean'] = temp_res / 3
    
    return res



def NP_metrics_from_img_path(img_path, **kwargs):
    img_np = load_image(img_path)
    res = NP_metrics_from_img_bgr(img_np, **kwargs)
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


def NP_metrics_from_img_list(img_list, return_each_sample=False, **kwargs):
    res = []
    for i, img_p in tqdm(enumerate(img_list)):
        single_res = NP_metrics_from_img_path(img_p, **kwargs)
        res.append(single_res)
    res = change_list2dict(res)

    return res if return_each_sample else dict_mean(res)


def dict2tensor(dicts):
    # change each element in dicts to tensor
    new_dict = dict()
    for key in dicts.keys():
        new_dict[key] = torch.tensor(dicts[key])
    return new_dict


def NP_metrics_from_tensor(img_tensor, return_each_sample=False, return_tensor=False, **kwargs):
    # [B, C, H, W]
    B = img_tensor.shape[0]
    res = []
    for i in tqdm(range(B)):
        img_t = img_tensor[i]
        img_bgr = tensor2npBGR(img_t)
        single_res = NP_metrics_from_img_bgr(img_bgr,**kwargs)
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
    # img1 = '/data1/lhy/pure_color/my_code/slight_red.png'
    img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/21_red_bg_white_watermark.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/26_green_red_circles_watermark.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/31_checkerboard_noise.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/25_blue_yellow_dots_saltpepper.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/30_gradient_grid_noise.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/08_gaussian_blue_std50.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/23_blue_bg_yellow_rect_logo.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/11_saltpepper_yellow_10pct.png'
    # img1 = '/data1/lhy/pure_color/my_code/test_noisy_images/20_blue_yellow_checkerboard.png'

    import os
    root_path = '/data1/lhy/pure_color/my_code/test_noisy_images'
    img_list = [os.path.join(root_path, _) for _ in os.listdir(root_path)]
    img_list = sorted(img_list)


    img_tensor1 = torch.randn(2,3,56,56)

    res = NP_metrics_from_img_path(img1)
    print(res)

    # res = NP_metrics_from_img_list(img_list, return_each_sample=True)
    # print(res)
    