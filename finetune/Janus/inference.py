import os
import json
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM
from peft import PeftModel
from janus.models import MultiModalityCausalLM, VLChatProcessor
from tqdm import tqdm
import subprocess
import sys

# Parse arguments before multiprocessing to avoid issues
parser = argparse.ArgumentParser()
parser.add_argument('--use_lora', action='store_true', help='Use LoRA finetuned model')
parser.add_argument('--lora_path', type=str, default=None, help='Path to LoRA checkpoint (default: auto-detect from dataset)')
parser.add_argument('--output_dir', type=str, default='./test_outputs', help='Output directory')
parser.add_argument('--dataset', type=str, default='color_region', choices=['color_region', 'prompt_level', 'color_region_half_split', 'dataset_v2', 'finetune_test_sets', 'finetune_level1'], 
                    help='Which dataset to use for testing')
# Internal arguments for multi-GPU worker
parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
parser.add_argument('--_gpu_id', type=int, default=0, help=argparse.SUPPRESS)
parser.add_argument('--_num_gpus', type=int, default=1, help=argparse.SUPPRESS)
parser.add_argument('--_output_dir_full', type=str, default='', help=argparse.SUPPRESS)
args = parser.parse_args()

# Auto-detect lora_path if not specified
if args.lora_path is None:
    args.lora_path = f'./lora_output_{args.dataset}'

model_path = "deepseek-ai/Janus-Pro-7B"
test_json_path = f"./processed_data_{args.dataset}/test.json"

# Generation parameters (same as original)
temperature = 1.0
cfg_weight = 5.0
parallel_size = 1


def run_as_worker():
    """Run as a single GPU worker"""
    gpu_id = args._gpu_id
    num_gpus = args._num_gpus
    output_dir = args._output_dir_full
    
    # Set CUDA device
    torch.cuda.set_device(0)  # Use device 0 since CUDA_VISIBLE_DEVICES is already set
    
    print(f"GPU {gpu_id}: Loading model...")
    config = AutoConfig.from_pretrained(model_path)
    config.language_config._attn_implementation = 'eager'

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        language_config=config.language_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda().eval()

    if args.use_lora:
        vl_gpt.language_model = PeftModel.from_pretrained(vl_gpt.language_model, args.lora_path)

    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    
    print(f"GPU {gpu_id}: Model loaded")

    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    
    # Filter out completed tasks
    pending_indices = []
    for idx, item in enumerate(test_data):
        source = item['source']
        output_filename = f"{source['prompt_level']}_{source['color_level']}_{source['filename']}"
        output_path = os.path.join(output_dir, output_filename)
        if not os.path.exists(output_path):
            pending_indices.append(idx)
    
    # Get this worker's subset of pending data
    indices = [pending_indices[i] for i in range(gpu_id, len(pending_indices), num_gpus)]
    print(f"GPU {gpu_id}: Processing {len(indices)} samples (total pending: {len(pending_indices)})")

    @torch.inference_mode()
    def generate_image(prompt):
        messages = [
            {'role': '<|User|>', 'content': prompt},
            {'role': '<|Assistant|>', 'content': ''}
        ]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=vl_chat_processor.sft_format,
            system_prompt=''
        )
        text = text + vl_chat_processor.image_start_tag
        input_ids = torch.LongTensor(tokenizer.encode(text))
        
        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id

        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((parallel_size, 576), dtype=torch.int).cuda()

        pkv = None
        for i in range(576):
            outputs = vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=pkv,
                output_hidden_states=True
            )
            pkv = outputs.past_key_values
            hidden_states = outputs.hidden_states[-1]
            
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        patches = vl_gpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, 384 // 16, 384 // 16]
        )

        dec = patches.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(dec[0])

    for idx in tqdm(indices, desc=f"GPU {gpu_id}"):
        item = test_data[idx]
        prompt = item['prompt']
        source = item['source']
        
        # Use source info to create output filename
        output_filename = f"{source['prompt_level']}_{source['color_level']}_{source['filename']}"
        output_path = os.path.join(output_dir, output_filename)
        
        img = generate_image(prompt)
        img.save(output_path)
    
    print(f"GPU {gpu_id}: Done")


def run_main():
    """Main process that spawns workers"""
    # Setup output directory in main process first
    output_dir = f"{args.output_dir}_{args.dataset}"
    if args.use_lora:
        output_dir = output_dir + "_lora"
    else:
        output_dir = output_dir + "_base"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get available GPUs from CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        gpu_list = cuda_visible.split(',')
    else:
        gpu_list = [str(i) for i in range(torch.cuda.device_count())]
    
    num_gpus = len(gpu_list)
    print(f"Found {num_gpus} GPUs: {gpu_list}")
    
    if num_gpus <= 1:
        # Single GPU - just run normally
        run_single_gpu(output_dir)
    else:
        # Multi-GPU - spawn separate processes
        processes = []
        for i, gpu in enumerate(gpu_list):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            
            cmd = [
                sys.executable, __file__,
                '--_worker',
                '--_gpu_id', str(i),
                '--_num_gpus', str(num_gpus),
                '--_output_dir_full', output_dir,
                '--output_dir', args.output_dir,
                '--dataset', args.dataset,
            ]
            if args.use_lora:
                cmd.append('--use_lora')
                cmd.extend(['--lora_path', args.lora_path])
            
            print(f"Starting worker on GPU {gpu}")
            p = subprocess.Popen(cmd, env=env)
            processes.append(p)
        
        # Wait for all workers to finish
        for p in processes:
            p.wait()
        
        print(f"All images saved to {output_dir}")


def run_single_gpu(output_dir):
    """Run on single GPU - original code"""
    print("Loading model...")
    config = AutoConfig.from_pretrained(model_path)
    config.language_config._attn_implementation = 'eager'

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        language_config=config.language_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda().eval()

    if args.use_lora:
        print(f"Loading LoRA from {args.lora_path}")
        vl_gpt.language_model = PeftModel.from_pretrained(vl_gpt.language_model, args.lora_path)
    else:
        print("Using base model (no LoRA)")

    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")
    
    # Filter out completed tasks
    pending_data = []
    for item in test_data:
        source = item['source']
        output_filename = f"{source['prompt_level']}_{source['color_level']}_{source['filename']}"
        output_path = os.path.join(output_dir, output_filename)
        if not os.path.exists(output_path):
            pending_data.append(item)
    
    print(f"Found {len(pending_data)} pending samples (skipping {len(test_data) - len(pending_data)} completed)")

    @torch.inference_mode()
    def generate_image(prompt):
        messages = [
            {'role': '<|User|>', 'content': prompt},
            {'role': '<|Assistant|>', 'content': ''}
        ]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=vl_chat_processor.sft_format,
            system_prompt=''
        )
        text = text + vl_chat_processor.image_start_tag
        input_ids = torch.LongTensor(tokenizer.encode(text))
        
        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id

        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((parallel_size, 576), dtype=torch.int).cuda()

        pkv = None
        for i in range(576):
            outputs = vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=pkv,
                output_hidden_states=True
            )
            pkv = outputs.past_key_values
            hidden_states = outputs.hidden_states[-1]
            
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        patches = vl_gpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, 384 // 16, 384 // 16]
        )

        dec = patches.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(dec[0])

    print(f"Generating images with {'LoRA' if args.use_lora else 'base'} model...")

    for item in tqdm(pending_data):
        prompt = item['prompt']
        source = item['source']
        
        # Use source info to create output filename
        output_filename = f"{source['prompt_level']}_{source['color_level']}_{source['filename']}"
        output_path = os.path.join(output_dir, output_filename)
        
        img = generate_image(prompt)
        img.save(output_path)

    print(f"All images saved to {output_dir}")


if __name__ == '__main__':
    if args._worker:
        run_as_worker()
    else:
        run_main()