#!/usr/bin/env python3
"""
Simple evaluation script for VIOLIN metrics.
Location: eval_open_source/evaluate/evaluate_open_source_models.py

Usage:
    python evaluate_open_source_models.py <gen_image> <gt_image> --type shape
    python evaluate_open_source_models.py <gen_image> <gt_image> --type color
    python evaluate_open_source_models.py <gen_image> <gt_image> --type color --multi
    python evaluate_open_source_models.py <gen_image> <gt_image> --type mask
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

# Add violin_metrics to path
project_root = Path(__file__).parent.parent.parent
violin_metrics_path = str(project_root / "violin_metrics")
if violin_metrics_path not in sys.path:
    sys.path.insert(0, violin_metrics_path)

# Import shape_metric module
from shape_metric import Shape_metrics_from_img_bgr, Metric_Dist_Normalized, Metric_IoU

# Create a wrapper function that converts binary masks to contours
def Metric_Dist(bin_gen, bin_gt):
    """
    Wrapper for Metric_Dist_Normalized that works with binary masks.
    Converts binary masks to contours and image shapes, then calls the original function.
    """
    # Find contours from binary masks
    contours_gen, _ = cv2.findContours(bin_gen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_gt, _ = cv2.findContours(bin_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return maximum distance
    if not contours_gen or not contours_gt:
        return 1.0
    
    # Get the largest contour from each
    cnt_gen = max(contours_gen, key=cv2.contourArea)
    cnt_gt = max(contours_gt, key=cv2.contourArea)
    
    # Use binary mask shapes as image shapes
    shape_gen = bin_gen.shape
    shape_gt = bin_gt.shape
    
    # Call the original function with correct arguments
    return Metric_Dist_Normalized(cnt_gen, cnt_gt, shape_gen, shape_gt)

# Inject the wrapper function into mask_metric module BEFORE importing from it
import mask_metric
mask_metric.Metric_Dist = Metric_Dist

# Now import the mask metric function
from mask_metric import Mask_metrics_from_img_bgr
from color_metric import Color_metrics_from_img_bgr


def resize_to_match(img1, img2):
    """Resize img1 to match img2's size."""
    h2, w2 = img2.shape[:2]
    h1, w1 = img1.shape[:2]
    
    if (h1, w1) != (h2, w2):
        print(f"Warning: Resizing image from {h1}x{w1} to {h2}x{w2}")
        img1 = cv2.resize(img1, (w2, h2), interpolation=cv2.INTER_LINEAR)
    
    return img1


def evaluate(gen_path, gt_path, metric_type='shape', is_multi_block=False):
    """
    Evaluate a single image.
    
    Args:
        gen_path: Generated image path
        gt_path: Ground truth image path
        metric_type: 'shape', 'mask', or 'color'
        is_multi_block: True for dual-color blocks (Variation 2)
    """
    # Load images
    img_gen = cv2.imread(gen_path)
    img_gt = cv2.imread(gt_path)
    
    if img_gen is None:
        raise FileNotFoundError(f'Cannot load image: {gen_path}')
    if img_gt is None:
        raise FileNotFoundError(f'Cannot load image: {gt_path}')
    
    # Resize generated image to match ground truth
    img_gen = resize_to_match(img_gen, img_gt)
    
    # Evaluate
    if metric_type == 'shape':
        return Shape_metrics_from_img_bgr(img_gen, img_gt)
    elif metric_type == 'mask':
        return Mask_metrics_from_img_bgr(img_gen, img_gt)
    elif metric_type == 'color':
        return Color_metrics_from_img_bgr(img_gen, img_gt, is_multi_block=is_multi_block)
    else:
        raise ValueError("metric_type must be 'shape', 'mask', or 'color'")


def main():
    parser = argparse.ArgumentParser(description='Evaluate images using VIOLIN metrics')
    parser.add_argument('gen_image', type=str, help='Path to generated image')
    parser.add_argument('gt_image', type=str, help='Path to ground truth image')
    parser.add_argument('--type', type=str, choices=['shape', 'color', 'mask'], 
                       default='shape', help='Metric type (default: shape)')
    parser.add_argument('--multi', action='store_true', 
                       help='Use multi-block mode for dual-color images (Variation 2)')
    
    args = parser.parse_args()
    
    # Evaluate
    results = evaluate(args.gen_image, args.gt_image, 
                      metric_type=args.type, 
                      is_multi_block=args.multi)
    
    # Print results
    print(f"\nResults ({args.type} metrics):")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key:15s}: {value:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()