import argparse
import torch
from PIL import Image
from diffusers import DiffusionPipeline
from optimum.quanto import quantize, qfloat8, freeze
from tqdm.auto import tqdm

def main(args):
    """
    Main function to load the model, apply quantization, and generate an image.
    """
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    print(f"Using device: {device} with dtype: {torch_dtype}")

    # Load the diffusion pipeline with auto-detection
    print(f"Loading model from: {args.model_name}")
    pipe = DiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch_dtype)

    # Load LoRA weights
    if args.lora_weights:
        print(f"Loading LoRA weights from: {args.lora_weights}")
        pipe.load_lora_weights(args.lora_weights, adapter_name="lora")

    # Set up quantization
    quantization_map = {
        "qfloat8": qfloat8,
    }
    quantization_type = quantization_map.get(args.quantization)
    if quantization_type is None:
        raise ValueError(f"Invalid quantization type: {args.quantization}. Choose from 'qfloat8'")

    print(f"Applying {args.quantization} quantization to the transformer...")

    # Check if model has transformer_blocks (Qwen-Image-Edit style)
    if hasattr(pipe.transformer, 'transformer_blocks'):
        print("Detected transformer_blocks, applying block-level quantization...")
        all_blocks = list(pipe.transformer.transformer_blocks)
        for block in tqdm(all_blocks):
            block.to(device, dtype=torch_dtype)
            quantize(block, weights=quantization_type)
            freeze(block)
            block.to('cpu')
    else:
        print("No transformer_blocks found, using whole-model quantization (Z-Image style)...")
    
    # Quantize the entire transformer (works for both models)
    pipe.transformer.to(device, dtype=torch_dtype)
    quantize(pipe.transformer, weights=quantization_type)
    freeze(pipe.transformer)
    print("Transformer quantization complete.")

    # Offload model to CPU to save VRAM, parts will be moved to GPU as needed
    pipe.enable_model_cpu_offload()

    # Set up the generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Load input images if provided
    input_images = []
    if args.input_images:
        for idx, img_path in enumerate(args.input_images):
            print(f"Loading input image {idx + 1}: {img_path}")
            image = Image.open(img_path)
            
            # Convert the second image (mask) to RGB if there are 2 images
            if idx == 1 and len(args.input_images) == 2:
                print(f"Converting image {idx + 1} (mask) to RGB mode...")
                image = image.convert('RGB')
                print(f"Image {idx + 1} mode after conversion: {image.mode}")
            
            input_images.append(image)
        print(f"Total input images loaded: {len(input_images)}")

    print("Generating image...")
    
    # Prepare generation kwargs
    gen_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "generator": generator,
        "width": args.width,
        "height": args.height,
    }
    
    # Add model-specific parameters
    if hasattr(pipe, 'true_cfg_scale'):
        gen_kwargs["true_cfg_scale"] = args.true_cfg_scale
    elif hasattr(pipe, 'guidance_scale'):
        gen_kwargs["guidance_scale"] = args.true_cfg_scale
    
    # Add input images if provided
    if len(input_images) > 0:
        gen_kwargs["image"] = input_images
    
    image = pipe(**gen_kwargs).images[0]

    # Save the output image
    image.save(args.output_image)
    print(f"Image successfully saved to {args.output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with a quantized diffusion model using LoRA.")

    # Model and Weights Arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-Image", help="Path or name of the base model.")
    parser.add_argument("--lora_weights", type=str, default="", help="Path to the LoRA weights.")
    parser.add_argument("--output_image", type=str, default="generated_image.png", help="Filename for the output image.")

    # Input Images Arguments
    parser.add_argument("--input_images", type=str, nargs='*', default=[], help="Paths to input images (supports multiple images).")

    # Generation Arguments
    parser.add_argument("--prompt", type=str, default='''man in the city''', help="The prompt for image generation.")
    parser.add_argument("--negative_prompt", type=str, default=" ", help="The negative prompt for image generation.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--true_cfg_scale", type=float, default=5.0, help="Classifier-Free Guidance scale.")
    parser.add_argument("--seed", type=int, default=655, help="Random seed for the generator.")

    # Quantization Arguments
    parser.add_argument("--quantization", type=str, default="qfloat8", choices=["qfloat8"], help="The quantization type to apply.")

    args = parser.parse_args()
    main(args)