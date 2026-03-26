#! /bin/bash

export MODEL_NAME="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
export TRAIN_DATA_DIR="ColorBench-v1/Finetune_Level1_Sets/Prompt_Level_1/Color_Level_1,ColorBench-v1/Finetune_Level1_Sets/Prompt_Level_1/Color_Level_2,ColorBench-v1/Finetune_Level1_Sets/Prompt_Level_1/Color_Level_3"
export OUTPUT_DIR="trained-sana1-5-lora-colorbench"

accelerate launch --num_processes 4 --main_process_port 29500 --gpu_ids 1,2,3,4 \
  train_scripts/train_lora_iconbench.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_data_dir=$TRAIN_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --resolution=384 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --use_8bit_adam \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=3000 \
  --validation_prompt="Represented in Hex code, create a canvas divided horizontally into two equal halves: upper color #D9B451, lower color #AC4AC3." \
  --validation_epochs=25 \
  --rank=16 \
  --alpha=32 \
  --seed="0" \
  --checkpointing_steps=100
