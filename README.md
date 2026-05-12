## Violin: Exploring the AI Obedience: Why is Generating a Pure Color Image Harder than CyberPunk?

<p align="center">
    <a href="https://arxiv.org/abs/2603.00166"><img src="https://img.shields.io/badge/arXiv-2603.00166-b31b1b.svg"></a>
    <a href="https://ai-obedience.github.io/"><img src="https://img.shields.io/badge/Project-Page-df7b33.svg"></a>
    <a href="https://huggingface.co/datasets/Perkzi/VIOLIN"><img src="https://img.shields.io/badge/Violin-Dataset-success.svg"></a>
</p>

## 🔥 News
* **2026.5.11:** The paper has been released.
* **2026.5.12:** The Violin dataset and Project Page bas been released.

![Introduction Diagram](assets/introduction.png)


## 🎨 Introduction

  Recent advances in generative AI have shown human-level performance in complex content creation. 
  However, we identify a "Paradox of Simplicity": models that can render complex scenes often fail at trivial, low-entropy tasks, such as generating a uniform pure color image. 
  We argue this is a systemic failure related to uncontrollable emergent abilities. 
  As models scale, strong priors for aesthetics and complexity override deterministic simplicity, creating an "aesthetic bias" that hinders the model's transition from data simulation to true intellectual abstraction.
  To better investigate this problem, we formalize the concept of AI Obedience, a hierarchical framework that grades a model's ability to transition from probabilistic approximation to pixel-level determinism (Levels 1 to 5).
  We introduce Violin, the first systematic benchmark designed to evaluate Level 4 Obedience through three deterministic tasks: color purity, image masking, and geometric shape generation. 
  Using Violin, we evaluate several state-of-the-art models and reveal that closed-source models generally outperform open-source ones in deterministic precision. Interestingly, performance on our benchmark correlates with the benchmark in natural image generation. 
  Our work provides a foundational framework and tools for achieving better alignment between human instructions and model outputs.





## 🤗 Dataset Download

Please run the following commands to download the dataset:

```python
# prepare
pip install -r requirements/requirement_general.txt

# download from anonymous huggingface
git clone https://huggingface.co/datasets/Perkzi/VIOLIN

# change parquet data to real images, this takes a few minutes.
python parquet_to_violin_data.py

# rename
mv Violin benchmark
```

## 🚀 Generate and Evaluate with Open-Source Models
### Installation
```bash
conda create -n violin_opensource python=3.10
conda activate violin_opensource
pip install -r requirements/requirement_opensource.txt
```
### Running the Benchmark
**Basic Usage:**
```bash
# Text-to-Image Generation
python eval_open_source/generate/generate_opensource_models.py \
    --model_name "Qwen/Qwen-Image" \
    --prompt "a pure red image" \
    --output_image "output.png"

# Image Editing (for Qwen-Image-Edit models)
python eval_open_source/generate/generate_opensource_models.py \
    --model_name "Qwen/Qwen-Image-Edit-2511" \
    --input_images "image1.png" "image2.png" \
    --prompt "replace the object with a tree" \
    --output_image "edited.png"
```

**Supported Models:**
- FLUX.1 (`black-forest-labs/FLUX.1-schnell`, `black-forest-labs/FLUX.1-dev`)
- FLUX.2 (`black-forest-labs/FLUX.2-klein-4B`, `black-forest-labs/FLUX.2-dev`)
- Z-Image (`Tongyi-MAI/Z-Image`, `Tongyi-MAI/Z-Image-Turbo`)
- Qwen-Image (`Qwen/Qwen-Image`, `Qwen/Qwen-Image-Edit-2511`)

**Key Arguments:**
- `--model_name`: Model path on HuggingFace (required)
- `--input_images`: Input images for editing (optional, space-separated)
- `--width`, `--height`: Image size (default: 1024×1024)
- `--num_inference_steps`: Steps (default: 20)
- `--seed`: Random seed (default: 655)

### Running the Benchmark
Once images are generated, evaluate them using our VIOLIN metrics:
```bash
# Evaluate Color Purity (Variation 1 - Single Block)
python eval_open_source/evaluate/evaluate_open_source_models.py \
    image1.png image2.png \
    --type color

# Evaluate Color Purity (Variation 2 - Dual Block)
python eval_open_source/evaluate/evaluate_open_source_models.py \
    gen_image.png gt_image.png \
    --type color --multi

# Evaluate Image Mask Task
python eval_open_source/evaluate/evaluate_open_source_models.py \
    generated_mask.png ground_truth_mask.png \
    --type mask

# Evaluate Geometric Shape Task
python eval_open_source/evaluate/evaluate_open_source_models.py \
    generated_shape.png ground_truth_shape.png \
    --type shape
```





## 🤝 Acknowledgements

The implementation of our open-source model evaluation suite is built upon the following repositories. We express our sincere gratitude to the authors and contributors for their pioneering work:

*   **[FLUX.1](https://github.com/black-forest-labs/flux)**
*   **[FLUX.2](https://github.com/black-forest-labs/flux2)**
*   **[Z-Image-Model](https://github.com/Tongyi-MAI/Z-Image)**
*   **[Qwen-Image](https://github.com/QwenLM/Qwen-Image)**


These repositories have significantly facilitated our research on visual obedience.


## 📑 Citation
If Violin is helpful, please help to ⭐ the repo.

```bibtex
@article{li2026exploring,
  title={Exploring the AI Obedience: Why is Generating a Pure Color Image Harder than CyberPunk?},
  author={Li, Hongyu and Liu, Kuan and Chen, Yuan and Hu, Juntao and Lu, Huimin and Chen, Guanjie and Liu, Xue and Lu, Guangming and Huang, Hong},
  journal={arXiv preprint arXiv:2603.00166},
  year={2026}
}
```


## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
