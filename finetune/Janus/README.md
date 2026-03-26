# JanusColor

Janus-Pro-7B LoRA fine-tuning for solid color image generation.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data

Select dataset in `prepare_data.py`:
```python
# Select dataset: "color_region" or "prompt_level"
dataset_name = "color_region"
# dataset_name = "prompt_level"
```

Then run:
```bash
CUDA_VISIBLE_DEVICES=2 python prepare_data.py
```

This generates:
- `./processed_data_color_region/train.json`
- `./processed_data_color_region/test.json`

### 2. Train

Select dataset in `train.py`:
```python
# Select dataset: "color_region" or "prompt_level"
dataset_name = "color_region"
# dataset_name = "prompt_level"
```

**Single GPU:**
```bash
CUDA_VISIBLE_DEVICES=2 python train.py
```

**Multi-GPU:**
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu --num_processes=4 train.py
```

Model saved to `./lora_output_{dataset_name}/`

### 3. Generate

**Base model:**
```bash
CUDA_VISIBLE_DEVICES=4 python inference.py --dataset color_region
```

**Base model (Multi-GPU):**
```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 python inference.py --dataset color_region
```

**LoRA model:**
```bash
CUDA_VISIBLE_DEVICES=3 python inference.py --dataset color_region --use_lora
```

**LoRA model (Multi-GPU):**
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python inference.py --dataset color_region --use_lora
```

**Custom LoRA path:**
```bash
python inference.py --dataset color_region --use_lora --lora_path ./custom_lora_path
```

Output saved to `./test_outputs_{dataset}_{lora|base}/`

## Configuration

### prepare_data.py
- `dataset_name` - Dataset to use: `"color_region"` or `"prompt_level"`
- `use_prompt_levels` - Prompt levels to include
- `use_color_levels` - Color levels to include

### train.py
- `dataset_name` - Dataset to use: `"color_region"` or `"prompt_level"`
- `batch_size = 8` - Per-device batch size
- `epochs = 5` - Training epochs  
- `lr = 3e-4` - Learning rate
- `lora_r = 16` - LoRA rank
- `lora_alpha = 32` - LoRA alpha
- `save_interval = 500` - Save checkpoint every N steps

### inference.py
- `--dataset` - Dataset to test: `color_region` or `prompt_level`
- `--use_lora` - Use LoRA model
- `--lora_path` - LoRA checkpoint path (default: auto-detect from dataset)
- `parallel_size = 1` - Batch size
- `cfg_weight = 5.0` - Guidance strength
- `temperature = 1.0` - Sampling temperature

### Model Path
```python
model_path = "deepseek-ai/Janus-Pro-7B"  # Change in all scripts
```