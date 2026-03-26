import os
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor, VLMImageProcessor
from tqdm import tqdm

model_path = "deepseek-ai/Janus-Pro-7B"

# ========== Configuration ==========
# Select dataset: "color_region", "prompt_level", "color_region_half_split" or "dataset_v2"
# dataset_name = "color_region"
# dataset_name = "prompt_level"
# dataset_name = "dataset_v2"
# dataset_name = "color_region_half_split"
dataset_name = "finetune_test_sets"

data_root = "/home/kuan/workspace/IconBench/ColorBench-v1/Test_Sets"
# if dataset_name == "dataset_v2":
#     data_root = "./IconBench_images_v2"
# else:
#     data_root = f"./IconBench_images_v3_{dataset_name}"

output_dir = f"./processed_data_{dataset_name}"

# Configure which directories to use
use_prompt_levels = ["Prompt_Level_1", "Prompt_Level_2", "Prompt_Level_3", "Prompt_Level_4", "Prompt_Level_5", "Prompt_Level_6"]
use_color_levels = ["Color_Level_1", "Color_Level_2", "Color_Level_3"]
# use_prompt_levels = ["Prompt_Level_1"]
# use_color_levels = ["Color_Level_1"]

# Maximum samples per folder for test split (set to None for no limit)
max_samples_per_folder = 1000  # 设置为 None 则不限制
# ===================================

os.makedirs(output_dir, exist_ok=True)

print("Loading model...")
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).cuda().eval()

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
vl_image_processor: VLMImageProcessor = VLMImageProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

def tokenize_text(prompt):
    conversation = [
        {'role': '<|User|>', 'content': prompt},
        {'role': '<|Assistant|>', 'content': ''},
    ]
    
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt='',
    )
    
    prompt_text = sft_format + vl_chat_processor.image_start_tag
    text_ids = tokenizer.encode(prompt_text)
    
    return text_ids

def encode_image(image_path):
    image = Image.open(image_path).convert('RGB')
    
    pixel_values = vl_image_processor(
        [image], 
        return_tensors='pt'
    )['pixel_values'].cuda().to(torch.bfloat16)
    
    with torch.no_grad():
        (
            quant,
            (vq_loss, commit_loss, entropy_loss),
            (perplexity, min_encodings, min_encoding_indices),
        ) = vl_gpt.gen_vision_model.encode(pixel_values)
    
    return min_encoding_indices.cpu().squeeze(0).tolist()

data_by_split = {'train': [], 'val': [], 'test': []}
skipped_count = 0

# Loop through all levels
for prompt_level in use_prompt_levels:
    for color_level in use_color_levels:
        folder = os.path.join(data_root, prompt_level, color_level)
        
        if not os.path.exists(folder):
            print(f"Skipping {prompt_level}/{color_level}: folder not found")
            continue
        
        # Iterate through train/val/test folders
        for split in ['train', 'val', 'test']:
            split_folder = os.path.join(folder, split)
            
            if not os.path.exists(split_folder):
                continue
            
            # Find all .png files
            png_files = [f for f in os.listdir(split_folder) if f.endswith('.png')]
            
            if not png_files:
                continue
            
            # Apply max_samples_per_folder limit (only for test split)
            if max_samples_per_folder is not None and split == 'test' and len(png_files) > max_samples_per_folder:
                png_files = png_files[:max_samples_per_folder]
                print(f"\nProcessing {prompt_level}/{color_level}/{split} (limited to {len(png_files)} samples)...")
            else:
                print(f"\nProcessing {prompt_level}/{color_level}/{split} ({len(png_files)} samples)...")
            
            for png_file in tqdm(png_files, desc=f"{prompt_level}/{color_level}/{split}"):
                # Get corresponding .txt file
                txt_file = png_file.replace('.png', '.txt')
                txt_path = os.path.join(split_folder, txt_file)
                image_path = os.path.join(split_folder, png_file)
                
                if not os.path.exists(txt_path):
                    skipped_count += 1
                    continue
                
                # Read prompt from txt file
                with open(txt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                
                text_ids = tokenize_text(prompt)
                image_tokens = encode_image(image_path)
                
                sample = {
                    "prompt": prompt,
                    "text_ids": text_ids,
                    "image_tokens": image_tokens,
                    "source": {
                        "dataset": dataset_name,
                        "prompt_level": prompt_level,
                        "color_level": color_level,
                        "filename": png_file
                    }
                }
                
                data_by_split[split].append(sample)

print(f"\n{'='*50}")
print(f"Processing complete!")
print(f"Skipped {skipped_count} samples (missing txt files)")
print(f"{'='*50}\n")

for split, data in data_by_split.items():
    if not data:
        continue
    output_path = os.path.join(output_dir, f"{split}.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} samples to {output_path}")

print("\nDone!")