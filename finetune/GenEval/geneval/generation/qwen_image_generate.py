"""Adapted from diffusers_generate.py with multi-GPU support and batch prompt optimization"""

import argparse
import json
import os

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from accelerate import PartialState
from optimum.quanto import quantize, qfloat8, freeze


torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata_file",
        type=str,
        nargs="?",
        default="prompts/evaluation_metadata.jsonl",
        help="JSONL file containing lines of metadata for each prompt"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Huggingface model name"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="number of samples per prompt",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        nargs="?",
        const="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        default=None,
        help="negative prompt for guidance"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--prompt_batch_size",
        type=int,
        default=16,
        help="how many prompts to process simultaneously",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="skip saving grid",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Path to LoRA weights file"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable qfloat8 quantization for transformer models"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing checkpoint folders (checkpoint-100, checkpoint-200, ...)"
    )
    parser.add_argument(
        "--skip_base",
        action="store_true",
        help="Skip running base model without LoRA"
    )
    opt = parser.parse_args()
    return opt


def check_already_generated(outpath, n_samples, skip_grid):
    """Check if this prompt has already been fully generated"""
    sample_path = os.path.join(outpath, "samples")
    if not os.path.exists(sample_path):
        return False
    
    for i in range(n_samples):
        if not os.path.exists(os.path.join(sample_path, f"{i:05}.png")):
            return False
    
    if not skip_grid:
        if not os.path.exists(os.path.join(outpath, "grid.png")):
            return False
    
    return True


def load_base_model(opt, distributed_state):
    """Load base model without LoRA"""
    torch_dtype = torch.bfloat16
    
    if opt.model == "stabilityai/stable-diffusion-xl-base-1.0":
        model = DiffusionPipeline.from_pretrained(
            opt.model, torch_dtype=torch_dtype, 
            use_safetensors=True, variant="fp16"
        )
    else:
        try:
            model = DiffusionPipeline.from_pretrained(opt.model, torch_dtype=torch_dtype)
        except:
            model = StableDiffusionPipeline.from_pretrained(opt.model, torch_dtype=torch_dtype)
    
    if opt.quantize and hasattr(model, 'transformer'):
        if distributed_state.is_main_process:
            print("Applying qfloat8 quantization...")
        
        all_blocks = list(model.transformer.transformer_blocks)
        for block in tqdm(all_blocks, disable=not distributed_state.is_main_process, desc="Quantizing"):
            block.to("cuda", dtype=torch_dtype)
            quantize(block, weights=qfloat8)
            freeze(block)
            block.to('cpu')
        model.transformer.to("cuda", dtype=torch_dtype)
        quantize(model.transformer, weights=qfloat8)
        freeze(model.transformer)
        
        if distributed_state.is_main_process:
            print("Quantization complete.")
        
        model.enable_model_cpu_offload(gpu_id=distributed_state.process_index)
    else:
        device = torch.device(f"cuda:{distributed_state.process_index}") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        model.enable_attention_slicing()
        
        if opt.model == "stabilityai/stable-diffusion-xl-base-1.0":
            try:
                model.enable_xformers_memory_efficient_attention()
            except:
                pass
    
    return model


def switch_lora(model, lora_path, distributed_state):
    """Switch LoRA weights on existing model"""
    # Unload existing LoRA if any
    try:
        model.unload_lora_weights()
    except:
        pass
    
    # Load new LoRA if specified
    if lora_path:
        if distributed_state.is_main_process:
            print(f"Loading LoRA: {lora_path}")
        model.load_lora_weights(lora_path, adapter_name="lora")
    else:
        if distributed_state.is_main_process:
            print("Using base model (no LoRA)")


def save_results(images, batch_indices, metadatas, outdir, opt):
    """Save generated images and metadata"""
    n_samples = opt.n_samples
    
    for j, idx in enumerate(batch_indices):
        outpath = os.path.join(outdir, f"{idx:0>5}")
        os.makedirs(outpath, exist_ok=True)
        
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        
        # Save metadata
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadatas[idx], fp)
        
        # Save individual samples
        start_idx = j * n_samples
        end_idx = start_idx + n_samples
        prompt_images = images[start_idx:end_idx]
        
        for k, img in enumerate(prompt_images):
            img.save(os.path.join(sample_path, f"{k:05}.png"))
        
        # Save grid if needed
        if not opt.skip_grid:
            n_rows = int(np.ceil(np.sqrt(n_samples)))
            grid_tensors = torch.stack([ToTensor()(img) for img in prompt_images], 0)
            grid = make_grid(grid_tensors, nrow=n_rows)
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid = Image.fromarray(grid.astype(np.uint8))
            grid.save(os.path.join(outpath, 'grid.png'))


def run_generation(opt, model, metadatas, outdir, distributed_state):
    """Run generation for a single model configuration with batch prompt processing"""
    pending_indices = []
    for index in range(len(metadatas)):
        outpath = os.path.join(outdir, f"{index:0>5}")
        if not check_already_generated(outpath, opt.n_samples, opt.skip_grid):
            pending_indices.append(index)
    
    if distributed_state.is_main_process:
        print(f"Total: {len(metadatas)}, Completed: {len(metadatas) - len(pending_indices)}, Pending: {len(pending_indices)}")
    
    if len(pending_indices) == 0:
        return

    with distributed_state.split_between_processes(pending_indices) as local_indices:
        local_indices = list(local_indices)
        
        prompt_batch_size = opt.prompt_batch_size
        n_samples = opt.n_samples
        
        total_batches = (len(local_indices) + prompt_batch_size - 1) // prompt_batch_size
        
        for i in tqdm(range(0, len(local_indices), prompt_batch_size), 
                      desc=f"GPU {distributed_state.process_index}",
                      total=total_batches,
                      disable=not distributed_state.is_local_main_process):
            
            batch_indices = local_indices[i:i + prompt_batch_size]
            batch_prompts = [metadatas[idx]['prompt'] for idx in batch_indices]
            batch_neg_prompts = [opt.negative_prompt if opt.negative_prompt else " "] * len(batch_prompts)
            
            seed = opt.seed + batch_indices[0]
            generator = torch.Generator(device="cpu").manual_seed(seed)
            
            with torch.no_grad():
                images = model(
                    prompt=batch_prompts,
                    negative_prompt=batch_neg_prompts,
                    width=opt.W,
                    height=opt.H,
                    num_inference_steps=opt.steps,
                    true_cfg_scale=opt.scale,
                    num_images_per_prompt=n_samples,
                    generator=generator,
                ).images
                
                save_results(images, batch_indices, metadatas, outdir, opt)
                
                del images
                torch.cuda.empty_cache()

    distributed_state.wait_for_everyone()


def get_checkpoint_paths(checkpoint_dir):
    """Scan checkpoint_dir and return sorted list of (step, name, path) tuples"""
    checkpoints = []
    if not os.path.exists(checkpoint_dir):
        return checkpoints
    
    for name in os.listdir(checkpoint_dir):
        if name.startswith("checkpoint-"):
            path = os.path.join(checkpoint_dir, name)
            if os.path.isdir(path):
                try:
                    step = int(name.split("-")[1])
                    checkpoints.append((step, name, path))
                except ValueError:
                    pass
    
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def build_configs(opt, distributed_state):
    """Build configs list: base + last checkpoint first, then rest"""
    configs = []
    
    all_checkpoints = []
    if opt.checkpoint_dir:
        all_checkpoints = get_checkpoint_paths(opt.checkpoint_dir)
        if distributed_state.is_main_process:
            print(f"Found {len(all_checkpoints)} checkpoints in {opt.checkpoint_dir}")
    
    # Priority: base, last checkpoint
    if not opt.skip_base:
        configs.append(("base", None))
    
    if all_checkpoints:
        last_step, last_name, last_path = all_checkpoints[-1]
        configs.append((last_name, last_path))
    
    # Then the rest (excluding last)
    for step, name, path in all_checkpoints[:-1]:
        configs.append((name, path))
    
    return configs


def main(opt):
    distributed_state = PartialState()
    
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    configs = build_configs(opt, distributed_state)
    
    if distributed_state.is_main_process:
        print(f"Execution order ({len(configs)} configs): {[c[0] for c in configs]}")
        print(f"Prompt batch size: {opt.prompt_batch_size}, Samples per prompt: {opt.n_samples}")
        print(f"Total images per batch: {opt.prompt_batch_size * opt.n_samples}")
    
    # Load base model once
    if distributed_state.is_main_process:
        print("Loading base model...")
    model = load_base_model(opt, distributed_state)
    
    for config_name, lora_path in configs:
        if distributed_state.is_main_process:
            print(f"\n{'='*50}")
            print(f"Running: {config_name}")
            print(f"{'='*50}")
        
        # Switch LoRA instead of reloading entire model
        switch_lora(model, lora_path, distributed_state)
        
        outdir = os.path.join(opt.outdir, config_name)
        run_generation(opt, model, metadatas, outdir, distributed_state)
    
    del model
    torch.cuda.empty_cache()
    
    if distributed_state.is_main_process:
        print("\nDone.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)