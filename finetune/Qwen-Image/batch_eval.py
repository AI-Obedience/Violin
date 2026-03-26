#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量测试 ColorBench metrics（多进程版）
"""

import os
import sys
import glob
import re
import cv2
import json
import warnings
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')


# ==================== 配置 ====================

GT_BASE = 'ColorBench-v1/Test_Sets'
RESULT_DIR = 'finetune/Qwen-Image/results'
CHECKPOINT_DIR = 'finetune/Qwen-Image/checkpoints'
EXPERIMENTS_CONFIG = 'finetune/Qwen-Image/experiments.json'

TARGET_SIZE = 512
NUM_WORKERS = 32
SAVE_INTERVAL = 500

CD_KEYS = ['srgb', 'srgb_redmean', 'delta_chroma', 'deltaE_HyAB', 'CIEDE2000', 'mae_hue']
NP_KEYS = ['standard_deviation', 'high_freq']

DISPLAY_HEADERS = [
    'srgb', 'srgb_redmean', 'delta_chroma', 'delta_HyAB', 'CIEDE2000',
    'mae_hue', 'color difference mean', 'standard_deviation', 'high_freq', 'color purity mean'
]

INTERNAL_KEYS = [
    'srgb', 'srgb_redmean', 'delta_chroma', 'deltaE_HyAB', 'CIEDE2000',
    'mae_hue', 'cd_mean', 'standard_deviation', 'high_freq', 'np_mean'
]

LEVEL5_SPLIT_FIELD = 'lang'
LEVEL5_SPLIT_MAP = {'zh': 'Chinese', 'fr': 'French'}

LEVEL6_SPLIT_FIELD = 'color_format'
LEVEL6_SPLIT_MAP = {'RGB': 'RGB', 'HSL': 'HSL'}


# ==================== 工具函数 ====================

def load_experiment_dirs():
    with open(EXPERIMENTS_CONFIG, 'r') as f:
        return json.load(f)


def load_and_resize(img_path, target_size=TARGET_SIZE):
    img = cv2.imread(img_path)
    if img is None:
        return None
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size))
    return img


def parse_filename(filename):
    pattern = r'Prompt_Level_(\d+)_Color_Level_(\d+)_test_id_(\d+)_gen\.png'
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def get_gt_path(prompt_level, color_level, test_id):
    return os.path.join(
        GT_BASE,
        f'Prompt_Level_{prompt_level}',
        f'Color_Level_{color_level}',
        'test',
        f'id_{test_id}.png'
    )


def get_prompt_path(prompt_base, prompt_level, color_level, test_id):
    return os.path.join(
        prompt_base,
        f'Prompt_Level_{prompt_level}',
        f'Color_Level_{color_level}',
        'test',
        f'id_{test_id}.txt'
    )


def load_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def get_checkpoint_path(exp_name):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f'{exp_name}_checkpoint.json')


def load_checkpoint(exp_name, retry_skipped=False):
    ckpt_path = get_checkpoint_path(exp_name)
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'r') as f:
            data = json.load(f)
        
        processed = set(data.get('processed', []))
        results = data.get('results', {})
        skipped = data.get('skipped', [])
        
        if retry_skipped and skipped:
            skipped_files = {item['file'] for item in skipped}
            processed -= skipped_files
            for f in skipped_files:
                results.pop(f, None)
            print(f"  -> Retrying {len(skipped_files)} previously skipped files")
            skipped = []
        
        return {'processed': processed, 'results': results, 'skipped': skipped}
    return {'processed': set(), 'results': {}, 'skipped': []}


def save_checkpoint(exp_name, processed, results, skipped):
    ckpt_path = get_checkpoint_path(exp_name)
    tmp_path = ckpt_path + '.tmp'
    data = {
        'processed': list(processed),
        'results': results,
        'skipped': skipped
    }
    with open(tmp_path, 'w') as f:
        json.dump(data, f)
    os.replace(tmp_path, ckpt_path)


# ==================== 单图处理（Worker 函数）====================

def process_single_image(args):
    gen_path, prompt_base = args
    filename = os.path.basename(gen_path)
    
    prompt_level, color_level, test_id = parse_filename(filename)
    if prompt_level is None:
        return (filename, None, 'invalid filename')
    
    gt_path = get_gt_path(prompt_level, color_level, test_id)
    
    try:
        sys.path.append('metrics')
        from single_metric_color_difference import CD_metrics_from_img_bgr as single_CD
        from single_metric_color_difference_level4 import CD_metrics_from_img_path as single_CD_lv4
        from single_metric_noisy_point import NP_metrics_from_img_bgr as single_NP
        from multi_metric_color_difference import CD_metrics_from_img_bgr as multi_CD
        from multi_metric_noisy_point import NP_metrics_from_img_bgr as multi_NP
    except Exception as e:
        return (filename, None, f'import error: {str(e)}')
    
    try:
        gen_img = load_and_resize(gen_path)
        if gen_img is None:
            return (filename, None, 'failed to load gen image')
        
        prompt = None
        if prompt_level == 4:
            prompt_path = get_prompt_path(prompt_base, prompt_level, color_level, test_id)
            try:
                prompt = load_prompt(prompt_path)
            except Exception as e:
                return (filename, None, f'failed to load prompt: {str(e)}')
            
            cd_result = single_CD_lv4(None, gen_path, prompt)
            np_result = single_NP(gen_img)
        elif prompt_level in [1, 5, 6]:
            gt_img = load_and_resize(gt_path)
            if gt_img is None:
                return (filename, None, 'failed to load gt image')
            cd_result = single_CD(gt_img, gen_img)
            np_result = single_NP(gen_img)
        else:
            gt_img = load_and_resize(gt_path)
            if gt_img is None:
                return (filename, None, 'failed to load gt image')
            cd_result = multi_CD(gt_img, gen_img)
            np_result = multi_NP(gt_img, gen_img)
        
        if cd_result is None:
            return (filename, None, 'metric computation failed')
        
        return (filename, {'cd': cd_result, 'np': np_result}, None)
    
    except Exception as e:
        return (filename, None, str(e))


def init_worker():
    import builtins
    builtins.print = lambda *args, **kwargs: None
    warnings.filterwarnings('ignore')


# ==================== 拆分相关 ====================

def load_split_csv(csv_base, prompt_level, color_level):
    csv_path = os.path.join(
        csv_base,
        f'Prompt_Level_{prompt_level}',
        f'Color_Level_{color_level}',
        f'Prompt_{prompt_level}_Color_{color_level}_normal_split.csv'
    )
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def build_split_map(csv_base):
    split_maps = {5: {}, 6: {}}
    
    for prompt_level in [5, 6]:
        split_field = LEVEL5_SPLIT_FIELD if prompt_level == 5 else LEVEL6_SPLIT_FIELD
        
        for color_level in range(1, 9):
            df = load_split_csv(csv_base, prompt_level, color_level)
            if df is None:
                continue
            
            df_test = df[df['split'] == 'test']
            for _, row in df_test.iterrows():
                if split_field in row:
                    split_maps[prompt_level][(color_level, row['id'])] = row[split_field]
    
    return split_maps


def get_level_key(prompt_level, color_level, test_id, split_maps):
    if split_maps and prompt_level in split_maps:
        split_value = split_maps[prompt_level].get((color_level, test_id))
        if split_value:
            split_name_map = LEVEL5_SPLIT_MAP if prompt_level == 5 else LEVEL6_SPLIT_MAP
            split_name = split_name_map.get(split_value, split_value)
            return f"Level-{prompt_level}-{split_name}"
    
    return f"Level-{prompt_level}"


# ==================== 计算统计 ====================

def compute_averages_for_group(cd_list, np_list):
    if not cd_list:
        return None
    
    avg = {'count': len(cd_list)}
    
    for key in CD_KEYS:
        values = [d[key] for d in cd_list if key in d]
        avg[key] = sum(values) / len(values) if values else 0
    
    cd_mean_values = [d['mean'] for d in cd_list if 'mean' in d]
    avg['cd_mean'] = sum(cd_mean_values) / len(cd_mean_values) if cd_mean_values else 0
    
    for key in NP_KEYS:
        values = [d[key] for d in np_list if key in d]
        avg[key] = sum(values) / len(values) if values else 0
    
    np_mean_values = [d['mean'] for d in np_list if 'mean' in d]
    avg['np_mean'] = sum(np_mean_values) / len(np_mean_values) if np_mean_values else 0
    
    return avg


def compute_summary(results, split_maps=None):
    grouped = {}
    
    for filename, data in results.items():
        p_level, c_level, t_id = parse_filename(filename)
        if p_level is None:
            continue
        
        level_key = get_level_key(p_level, c_level, t_id, split_maps)
        
        if level_key not in grouped:
            grouped[level_key] = {'cd': [], 'np': []}
        
        grouped[level_key]['cd'].append(data['cd'])
        grouped[level_key]['np'].append(data['np'])
    
    summary = {}
    for level_key, data in grouped.items():
        avg = compute_averages_for_group(data['cd'], data['np'])
        if avg:
            summary[level_key] = avg
    
    return summary


# ==================== 主测试逻辑 ====================

def test_experiment(exp_name, exp_config, retry_skipped=False, num_workers=None):
    print(f"\n{'='*60}")
    print(f"Testing: {exp_name}")
    print(f"{'='*60}")
    
    exp_dir = exp_config['output_dir']
    prompt_base = exp_config['prompt_base']
    split_csv_base = exp_config.get('split_csv_base')
    
    gen_files = glob.glob(os.path.join(exp_dir, '*_gen.png'))
    print(f"Found {len(gen_files)} images")
    print(f"Prompt base: {prompt_base}")
    if split_csv_base:
        print(f"Split CSV base: {split_csv_base}")
    
    if num_workers is None:
        num_workers = min(cpu_count(), 16)
    print(f"Using {num_workers} workers")
    
    ckpt = load_checkpoint(exp_name, retry_skipped=retry_skipped)
    processed = set(ckpt['processed'])
    results = ckpt['results']
    skipped = ckpt['skipped']
    
    if processed:
        print(f"Resuming: {len(processed)} done, {len(skipped)} skipped")
    
    todo_files = [f for f in gen_files if os.path.basename(f) not in processed]
    print(f"To process: {len(todo_files)} images")
    
    if not todo_files:
        print("Nothing to process")
    else:
        task_args = [(f, prompt_base) for f in todo_files]
        
        new_count = 0
        
        try:
            with Pool(num_workers, initializer=init_worker) as pool:
                for result in tqdm(
                    pool.imap_unordered(process_single_image, task_args),
                    total=len(task_args),
                    desc="Processing",
                    ncols=80
                ):
                    filename, data, error = result
                    
                    if data is not None:
                        results[filename] = data
                    elif error != 'invalid filename':
                        skipped.append({'file': filename, 'reason': error})
                    
                    processed.add(filename)
                    new_count += 1
                    
                    if new_count % SAVE_INTERVAL == 0:
                        save_checkpoint(exp_name, processed, results, skipped)
        
        except KeyboardInterrupt:
            print(f"\n\nInterrupted! Saving checkpoint...")
            save_checkpoint(exp_name, processed, results, skipped)
            raise
    
    save_checkpoint(exp_name, processed, results, skipped)
    
    print(f"Total processed: {len(results)}, Skipped: {len(skipped)}")
    
    split_maps = None
    if split_csv_base:
        print("Building split maps...")
        split_maps = build_split_map(split_csv_base)
        print(f"  Level-5: {len(split_maps.get(5, {}))}, Level-6: {len(split_maps.get(6, {}))}")
    
    summary = compute_summary(results, split_maps)
    
    return {
        'experiment': exp_name,
        'summary': summary,
        'skipped': skipped,
        'has_split': split_csv_base is not None
    }


# ==================== 输出结果 ====================

def get_level_order(has_split):
    if has_split:
        return [
            'Level-1', 'Level-2', 'Level-3', 'Level-4',
            'Level-5-Chinese', 'Level-5-French',
            'Level-6-RGB', 'Level-6-HSL'
        ]
    return ['Level-1', 'Level-2', 'Level-3', 'Level-4', 'Level-5', 'Level-6']


def print_summary(all_results):
    has_any_split = any(r.get('has_split', False) for r in all_results)
    
    all_level_keys = set()
    for result in all_results:
        all_level_keys.update(result['summary'].keys())
    
    level_order = get_level_order(has_any_split)
    for key in sorted(all_level_keys):
        if key not in level_order:
            level_order.append(key)
    
    col_width = 20
    level_col_width = 18 if has_any_split else 12
    
    print("\n" + "=" * 240)
    header = f"{'Level':<{level_col_width}} {'Model':<30} "
    header += " ".join([f"{h:<{col_width}}" for h in DISPLAY_HEADERS])
    print(header)
    print("-" * 240)
    
    for level in level_order:
        if level not in all_level_keys:
            continue
        first = True
        for result in all_results:
            if level in result['summary']:
                data = result['summary'][level]
                level_col = level if first else ""
                row = f"{level_col:<{level_col_width}} {result['experiment']:<30} "
                row += " ".join([f"{data.get(k, 0):<{col_width}.6f}" for k in INTERNAL_KEYS])
                print(row)
                first = False
        print("-" * 240)


def save_results(all_results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    has_any_split = any(r.get('has_split', False) for r in all_results)
    
    all_level_keys = set()
    for result in all_results:
        all_level_keys.update(result['summary'].keys())
    
    level_order = get_level_order(has_any_split)
    for key in sorted(all_level_keys):
        if key not in level_order:
            level_order.append(key)
    
    col_width = 20
    level_col_width = 18 if has_any_split else 12
    
    with open(output_path, 'w') as f:
        f.write("ColorBench Metrics Results\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Image Size: {TARGET_SIZE}x{TARGET_SIZE}\n")
        if has_any_split:
            f.write("Note: Level-5 split by language, Level-6 split by format\n")
        f.write("=" * 240 + "\n\n")
        
        header = f"{'Level':<{level_col_width}} {'Model':<30} "
        header += " ".join([f"{h:<{col_width}}" for h in DISPLAY_HEADERS])
        f.write(header + "\n")
        f.write("-" * 240 + "\n")
        
        for level in level_order:
            if level not in all_level_keys:
                continue
            first = True
            for result in all_results:
                if level in result['summary']:
                    data = result['summary'][level]
                    level_col = level if first else ""
                    row = f"{level_col:<{level_col_width}} {result['experiment']:<30} "
                    row += " ".join([f"{data.get(k, 0):<{col_width}.6f}" for k in INTERNAL_KEYS])
                    f.write(row + "\n")
                    first = False
            f.write("-" * 240 + "\n")
        
        f.write("\n\nSAMPLE COUNTS\n" + "=" * 60 + "\n")
        for result in all_results:
            f.write(f"\n{result['experiment']}:\n")
            for level in level_order:
                if level in result['summary']:
                    f.write(f"  {level}: {result['summary'][level].get('count', 0)}\n")
        
        f.write("\n\nSKIPPED FILES\n" + "=" * 60 + "\n")
        for result in all_results:
            if result['skipped']:
                f.write(f"\n{result['experiment']}:\n")
                for item in result['skipped']:
                    f.write(f"  - {item['file']}: {item['reason']}\n")
    
    print(f"\nResults saved to: {output_path}")


def list_experiments(experiment_dirs):
    print("Configured experiments:")
    for name, config in experiment_dirs.items():
        exists = "✓" if os.path.exists(config['output_dir']) else "✗"
        print(f"  [{exists}] {name}")
        print(f"        output: {config['output_dir']}")
        print(f"        prompt: {config['prompt_base']}")
        if config.get('split_csv_base'):
            print(f"        split:  {config['split_csv_base']}")


def main():
    parser = argparse.ArgumentParser(description='ColorBench Metrics (Parallel)')
    parser.add_argument('--retry-skipped', action='store_true')
    parser.add_argument('--experiments', nargs='+')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()
    
    experiment_dirs = load_experiment_dirs()
    
    if args.list:
        list_experiments(experiment_dirs)
        return
    
    num_workers = args.workers or NUM_WORKERS
    
    print("ColorBench Metrics Batch Evaluation (Parallel)")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Save interval: {SAVE_INTERVAL}")
    
    if args.experiments:
        for exp in args.experiments:
            if exp not in experiment_dirs:
                print(f"Error: Unknown experiment '{exp}'")
                return
        experiments = args.experiments
    else:
        experiments = list(experiment_dirs.keys())
    
    all_results = []
    for exp_name in experiments:
        exp_config = experiment_dirs[exp_name]
        if not os.path.exists(exp_config['output_dir']):
            print(f"Warning: {exp_config['output_dir']} not found, skipping")
            continue
        result = test_experiment(exp_name, exp_config, args.retry_skipped, num_workers)
        all_results.append(result)
    
    if all_results:
        print_summary(all_results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_results(all_results, os.path.join(RESULT_DIR, f'metrics_{timestamp}.txt'))
    
    print("\nDone!")


if __name__ == '__main__':
    main()