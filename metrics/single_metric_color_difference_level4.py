import cv2
import numpy as np
import torch
from tqdm import tqdm
from skimage.color import deltaE_ciede2000

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
    **kwargs
):

    img1_lab = convert_BGR_to_LAB(img1)
    img2_lab = convert_BGR_to_LAB(img2)

    res = CD_metrics_from_img_lab(img1_lab, img2_lab, **kwargs)

    if return_sRGB:
        res['srgb'] = RGB_distance_from_BGR_img(img1, img2)
    if return_sRGB_redmean:
        res['srgb_redmean'] = RGB_Redmean_distance_from_BGR_img(img1, img2)

    if return_mean:
        res['mean'] = sum(res.values())/len(res.values())

    return res




def CD_metrics_from_img_path(
    img_path1, 
    img_path2, 
    prompt,
    **kwargs,
):
    # img1 = load_image(img_path1)
    img2 = load_image(img_path2)

    lower, upper, lower_hls, upper_hls = extract_range_hsl(prompt)

    # Suppose img2 is loaded as numpy array [H,W,3] in RGB
    mean_rgb = get_image_mean_color(img2)
    closest_hex = project_to_range_hsl(mean_rgb, lower_hls, upper_hls)

    # print("Closest point on range:", closest_hex)
    
    img1 = solid_image_from_hex(closest_hex, img2.shape)
    
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
        new_dict[key] = sum(dicts[key])/len(dicts[key])
    return new_dict


def dict2tensor(dicts):
    # change each element in dicts to tensor
    new_dict = dict()
    for key in dicts.keys():
        new_dict[key] = torch.tensor(dicts[key])
    return new_dict



def CD_metrics_from_img_list(img_list1, img_list2, prompt_list, return_each_sample=False, **kwargs):
    if not (len(img_list1) == len(img_list2) == len(prompt_list)):
        raise ValueError(
            f'list1={len(img_list1)}, list2={len(img_list2)}, prompt_list={len(prompt_list)}, not match'
        )
    
    # zip three lists together
    triplets = list(zip(img_list1, img_list2, prompt_list))
    
    # sort by img_list1 (first element of triplet)
    triplets = sorted(triplets, key=lambda x: x[0])
    
    res = []
    for img_p1, img_p2, prompt in tqdm(triplets):
        res.append(CD_metrics_from_img_path(img_p1, img_p2, prompt, **kwargs))
    
    res = change_list2dict(res)
    return res if return_each_sample else dict_mean(res)




# def CD_metrics_from_tensor(tensor1, tensor2,prompt, return_tensor=True, return_each_sample=False, **kwargs):
#     # [B, C, H, W]
#     if tensor1.shape != tensor2.shape:
#         raise ValueError(f'shape 1:{tensor1.shape} not match shape 2:{tensor2.shape}')
    
#     B = tensor1.shape[0]

#     res = []
#     for idx in tqdm(range(B)):
#         t1 = tensor2npLAB(tensor1[idx])
#         t2 = tensor2npLAB(tensor2[idx])

#         res.append(CD_metrics_from_img_bgr(t1, t2, **kwargs))

    
#     if return_each_sample:
#         if return_tensor:
#             return dict2tensor(change_list2dict(res))
#         else:
#             return change_list2dict(res)
#     else:
#         if return_tensor:
#             return dict2tensor(dict_mean(change_list2dict(res)))
#         else:
#             return dict_mean(change_list2dict(res))






import numpy as np
import colorsys
import re
from collections import Counter

def hex_to_rgb(hex_val):
    hex_val = hex_val.lstrip('#')
    return tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def rgb_to_hls(rgb):
    r, g, b = [c/255.0 for c in rgb]
    return colorsys.rgb_to_hls(r, g, b)  # returns (h, l, s)

def hls_to_rgb(hls):
    r, g, b = colorsys.hls_to_rgb(*hls)
    return (int(round(r*255)), int(round(g*255)), int(round(b*255)))

def extract_range_hsl(prompt):
    """Extract lower and upper hex colors from prompt and convert to HLS"""
    hex_codes = re.findall(r'#[0-9A-Fa-f]{6}', prompt)
    if len(hex_codes) < 2:
        raise ValueError("Prompt does not contain two hex color codes.")
    lower, upper = hex_codes[0], hex_codes[1]
    lower_hls = rgb_to_hls(hex_to_rgb(lower))
    upper_hls = rgb_to_hls(hex_to_rgb(upper))
    return lower, upper, lower_hls, upper_hls

def get_image_mean_color(img):
    """Compute mean RGB color of image"""
    mean_rgb = img.mean(axis=(0,1)).astype(int)
    return tuple(mean_rgb.tolist())

def project_to_range_hsl(rgb_color, lower_hls, upper_hls):
    """
    Project a color (in RGB) onto the line segment defined by lower_hls and upper_hls.
    Returns the closest point on the line in HLS, converted back to hex.
    """
    hls_color = rgb_to_hls(rgb_color)
    v = np.array(hls_color)
    a = np.array(lower_hls)
    b = np.array(upper_hls)

    # Projection of v onto line segment ab
    ab = b - a
    t = np.dot(v - a, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)  # clamp to segment
    proj = a + t * ab

    proj_rgb = hls_to_rgb(tuple(proj))
    return rgb_to_hex(proj_rgb)


def solid_image_from_hex(hex_color, shape):
    """
    Generate a solid color image array with the same shape as img2.
    
    param:
        hex_color: str, e.g. "#7F4829"
        shape: tuple, (H, W, C) from img2.shape
    return:
        numpy array [H, W, 3] filled with the given color
    """
    rgb = hex_to_rgb(hex_color)
    img = np.zeros(shape, dtype=np.uint8)
    img[:] = rgb
    return img





if __name__ == '__main__':
    prompt = 'Specified in Hex code, generate an image in a single solid color within the range of #58321D and #A65E35.'
    img1 = '.\examples\id_4.png' # ground truth
    img2 = '.\examples\id_5.png'



    img_list1 = [img1]*10
    img_list2 = [img2]*10
    prompt_list = [prompt]*10

    # img_tensor1 = torch.randn(2,3,56,56)
    # img_tensor2 = torch.randn(2,3,56,56)

    res = CD_metrics_from_img_path(img1, img2, prompt=prompt)
    print(res)

    res = CD_metrics_from_img_list(img_list1, img_list2, prompt_list=prompt_list, return_each_sample=False)
    print(res)

    # res = CD_metrics_from_tensor(img_tensor1, img_tensor2)
    # print(res)