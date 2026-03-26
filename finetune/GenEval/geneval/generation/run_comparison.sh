#!/bin/bash

METADATA_FILE="${1:-../prompts/evaluation_metadata.jsonl}"
MODEL="Qwen/Qwen-Image"
LORA_WEIGHTS="/home/kuan/code/flymyai-lora-trainer/lora_saves_qwen_finetune_result/checkpoint-3000-3e-4/pytorch_lora_weights.safetensors"
GPUS="0,1,4,7"
NUM_PROCESSES=4

OUTDIR_BASE="/data4/kuan/geneval_outputs/outputs_base"
OUTDIR_LORA="/data4/kuan/geneval_outputs/outputs_lora"

export CUDA_VISIBLE_DEVICES=$GPUS

# 带 LoRA
accelerate launch --num_processes=$NUM_PROCESSES qwen_image_generate.py "$METADATA_FILE" \
    --model "$MODEL" \
    --outdir "$OUTDIR_LORA" \
    --lora_weights "$LORA_WEIGHTS" \
    --quantize \
    --batch_size 2

# 无 LoRA
accelerate launch --num_processes=$NUM_PROCESSES qwen_image_generate.py "$METADATA_FILE" \
    --model "$MODEL" \
    --outdir "$OUTDIR_BASE" \
    --quantize \
    --batch_size 2