"""
Janus model image generation for GenEval benchmark
Improved version with multi-GPU support and better LoRA loading
"""

import argparse
import json
import os
import sys
import subprocess

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from peft import PeftModel

from janus.models import MultiModalityCausalLM, VLChatProcessor


torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using Janus for GenEval")
    parser.add_argument(
        "metadata_file",
        type=str,
        help="JSONL file containing evaluation metadata"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/Janus-Pro-7B",
        help="Model path (default:  Janus-Pro-7B)"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA weights"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of samples per prompt"
    )
    parser.add_argument(
        "--cfg_weight",
        type=float,
        default=5.0,
        help="CFG weight"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="Skip grid generation"
    )
    
    # Multi-GPU support (internal args)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_gpu_id', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--_num_gpus', type=int, default=1, help=argparse.SUPPRESS)
    
    return parser.parse_args()


def load_model(args):
    """Load Janus model with improved configuration"""
    print(f"Loading model: {args.model}")
    
    # Load config and set attention implementation
    config = AutoConfig.from_pretrained(args.model)
    config.language_config._attn_implementation = 'eager'
    
    # Load processor
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model)
    tokenizer = vl_chat_processor.tokenizer
    
    # Load base model
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        args.model,
        language_config=config.language_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda().eval()
    
    # Load LoRA weights if specified
    if args.lora_path:
        print(f"Loading LoRA from: {args.lora_path}")
        vl_gpt. language_model = PeftModel. from_pretrained(
            vl_gpt.language_model,
            args.lora_path
        )
        print("LoRA loaded successfully!")
    
    return vl_gpt, vl_chat_processor, tokenizer


@torch.inference_mode()
def generate_images(
    vl_gpt,
    vl_chat_processor,
    prompt:  str,
    n_samples:  int = 4,
    temperature: float = 1.0,
    cfg_weight: float = 5.0,
):
    """Generate images using Janus model"""
    tokenizer = vl_chat_processor.tokenizer
    
    # Prepare prompt
    messages = [
        {'role': '<|User|>', 'content': prompt},
        {'role': '<|Assistant|>', 'content': ''}
    ]
    text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=messages,
        sft_format=vl_chat_processor. sft_format,
        system_prompt=''
    )
    text = text + vl_chat_processor.image_start_tag
    input_ids = torch.LongTensor(tokenizer.encode(text))
    
    # Prepare tokens for CFG
    tokens = torch.zeros((n_samples * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(n_samples * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((n_samples, 576), dtype=torch.int).cuda()
    
    # Generate tokens
    pkv = None
    for i in range(576):
        outputs = vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=pkv,
            output_hidden_states=True
        )
        pkv = outputs.past_key_values
        hidden_states = outputs. hidden_states[-1]
        
        logits = vl_gpt. gen_head(hidden_states[: , -1, :])
        logit_cond = logits[0:: 2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[: , i] = next_token. squeeze(dim=-1)
        
        next_token = torch.cat([next_token. unsqueeze(dim=1), next_token. unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = vl_gpt. prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds. unsqueeze(dim=1)
    
    # Decode to images
    patches = vl_gpt.gen_vision_model.decode_code(
        generated_tokens. to(dtype=torch.int),
        shape=[n_samples, 8, 384 // 16, 384 // 16]
    )
    
    dec = patches.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    
    images = [Image.fromarray(dec[i]) for i in range(n_samples)]
    return images


def create_grid(images, n_cols=2):
    """Create image grid"""
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    img_width, img_height = images[0].size
    grid = Image.new('RGB', (img_width * n_cols, img_height * n_rows))
    
    for i, img in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        grid.paste(img, (col * img_width, row * img_height))
    
    return grid


def run_worker(args):
    """Run as worker process"""
    gpu_id = args._gpu_id
    num_gpus = args._num_gpus
    
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"GPU {gpu_id}:  Loading model...")
    vl_gpt, vl_chat_processor, tokenizer = load_model(args)
    
    # Load metadata
    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    
    # Get worker's subset (interleaved)
    indices = list(range(gpu_id, len(metadatas), num_gpus))
    print(f"GPU {gpu_id}: Processing {len(indices)} samples")
    
    for idx in tqdm(indices, desc=f"GPU {gpu_id}"):
        metadata = metadatas[idx]
        prompt = metadata['prompt']
        
        outpath = os.path.join(args.outdir, f"{idx:0>5}")
        os.makedirs(outpath, exist_ok=True)
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        
        # Save metadata
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)
        
        try:
            images = generate_images(
                vl_gpt, vl_chat_processor, prompt,
                n_samples=args. n_samples,
                temperature=args.temperature,
                cfg_weight=args.cfg_weight
            )
            
            for i, img in enumerate(images):
                img.save(os.path.join(sample_path, f"{i:05d}.png"))
            
            if not args.skip_grid:
                grid = create_grid(images)
                grid.save(os. path.join(outpath, "grid.png"))
                
        except Exception as e:
            print(f"GPU {gpu_id}: Error on sample {idx}: {e}")
            continue
    
    print(f"GPU {gpu_id}: Done")


def run_single_gpu(args):
    """Run on single GPU"""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    vl_gpt, vl_chat_processor, tokenizer = load_model(args)
    
    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    print(f"Loaded {len(metadatas)} prompts")
    
    for index, metadata in enumerate(tqdm(metadatas, desc="Generating")):
        prompt = metadata['prompt']
        
        outpath = os.path.join(args. outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)
        
        try: 
            images = generate_images(
                vl_gpt, vl_chat_processor, prompt,
                n_samples=args. n_samples,
                temperature=args.temperature,
                cfg_weight=args.cfg_weight
            )
            
            for i, img in enumerate(images):
                img.save(os.path.join(sample_path, f"{i:05d}.png"))
            
            if not args. skip_grid:
                grid = create_grid(images)
                grid.save(os.path. join(outpath, "grid.png"))
                
        except Exception as e:
            print(f"Error on sample {index}: {e}")
            continue
    
    print(f"Done!  Results in {args.outdir}")


def main():
    args = parse_args()
    
    if args._worker:
        run_worker(args)
        return
    
    # Setup output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Check GPUs
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        gpu_list = cuda_visible.split(',')
    else:
        gpu_list = [str(i) for i in range(torch.cuda.device_count())]
    
    num_gpus = len(gpu_list)
    print(f"Found {num_gpus} GPUs:  {gpu_list}")
    
    if num_gpus <= 1:
        run_single_gpu(args)
    else:
        # Launch workers
        processes = []
        for i, gpu in enumerate(gpu_list):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            
            cmd = [
                sys.executable, __file__,
                args.metadata_file,
                '--_worker',
                '--_gpu_id', str(i),
                '--_num_gpus', str(num_gpus),
                '--model', args.model,
                '--outdir', args.outdir,
                '--n_samples', str(args.n_samples),
                '--cfg_weight', str(args.cfg_weight),
                '--temperature', str(args.temperature),
                '--seed', str(args.seed),
            ]
            if args.lora_path:
                cmd.extend(['--lora-path', args.lora_path])
            if args.skip_grid:
                cmd.append('--skip_grid')
            
            print(f"Starting worker on GPU {gpu}")
            p = subprocess.Popen(cmd, env=env)
            processes.append(p)
        
        for p in processes:
            p. wait()
        
        print(f"All done! Results in {args.outdir}")


if __name__ == "__main__":
    main()