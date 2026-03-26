import argparse
import random
from pathlib import Path

import torch
from diffusers import SanaPipeline
from tqdm import tqdm


def build_pipeline() -> SanaPipeline:
    pipe = SanaPipeline.from_pretrained(
        "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda:1")
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.load_lora_weights("trained-sana1-5-lora-colorbench/checkpoint-3000")
    return pipe


def iter_prompt_files(test_sets_root: Path, max_per_test: int, rng: random.Random):
    test_dirs = [p for p in test_sets_root.rglob("test") if p.is_dir()]
    for test_dir in test_dirs:
        txt_files = [p for p in test_dir.glob("*.txt") if p.is_file()]
        if len(txt_files) > max_per_test:
            txt_files = rng.sample(txt_files, max_per_test)
        for txt_path in txt_files:
            yield txt_path


def output_path_for_prompt(txt_path: Path, test_sets_root: Path, output_root: Path) -> Path:
    rel_parent = txt_path.parent.parent.relative_to(test_sets_root)
    return output_root / rel_parent / f"{txt_path.stem}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ColorBench images with SANA1.5.")
    parser.add_argument(
        "--test-sets-root",
        type=Path,
        default=Path("ColorBench-v1/Finetune_Level1_Sets"),
        help="Root of ColorBench-v1/Test_Sets",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ColorBench-gen-fine"),
        help="Output root directory",
    )
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_sets_root = args.test_sets_root
    output_root = args.output_root

    pipe = build_pipeline()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    rng = random.Random(args.seed)
    prompt_files = list(iter_prompt_files(test_sets_root, 1000, rng))
    for txt_path in tqdm(prompt_files, desc="Generating", unit="prompt"):
        prompt = txt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            continue
        out_path = output_path_for_prompt(txt_path, test_sets_root, output_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            generator=generator,
        )[0][0]
        image.save(out_path)


if __name__ == "__main__":
    main()
