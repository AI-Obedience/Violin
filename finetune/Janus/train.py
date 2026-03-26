import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from janus.models import VLChatProcessor
from accelerate import Accelerator
import wandb

model_path = "deepseek-ai/Janus-Pro-7B"

# ========== Configuration ==========
# Select dataset: "color_region", "prompt_level", "color_region_half_split" or "dataset_v2"
# dataset_name = "color_region"
# dataset_name = "prompt_level"
# dataset_name = "dataset_v2"
# dataset_name = "color_region_half_split"
dataset_name = "finetune_level1"

data_path = f"./processed_data_{dataset_name}/train.json"
output_dir = f"./lora_output_{dataset_name}"
# ===================================

batch_size = 8
max_steps = 3000  # 最大训练步数
lr = 3e-4
lora_r = 16
lora_alpha = 32

save_interval = 100  # Save checkpoint every 500 steps

wandb_project = "JanusColorLora"
wandb_run_name = f"{dataset_name}_bs{batch_size}"

accelerator = Accelerator()

if accelerator.is_main_process:
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "model_path": model_path,
            "dataset": dataset_name,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "lr": lr,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "num_gpus": accelerator.num_processes,
        }
    )

class ColorDataset(Dataset):
    def __init__(self, data_path, processor):
        with open(data_path) as f:
            self.data = json.load(f)
        self.processor = processor
        self.tokenizer = processor.tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        messages = [
            {'role': '<|User|>', 'content': item["prompt"]},
            {'role': '<|Assistant|>', 'content': ''}
        ]
        
        text = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=self.processor.sft_format,
            system_prompt=''
        )
        text = text + self.processor.image_start_tag
        
        text_ids = self.tokenizer.encode(text)
        
        return {
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "image_tokens": torch.tensor(item["image_tokens"], dtype=torch.long).squeeze(),
        }

def collate_fn(batch):
    max_text_len = max(item["text_ids"].size(0) for item in batch)
    
    text_ids = []
    image_tokens = []
    
    for item in batch:
        pad_len = max_text_len - item["text_ids"].size(0)
        padded = torch.cat([torch.zeros(pad_len, dtype=torch.long), item["text_ids"]])
        text_ids.append(padded)
        image_tokens.append(item["image_tokens"])
    
    return {
        "text_ids": torch.stack(text_ids),
        "image_tokens": torch.stack(image_tokens),
    }

if accelerator.is_main_process:
    print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)

processor = VLChatProcessor.from_pretrained(model_path)

lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)

model.language_model = get_peft_model(model.language_model, lora_config)
model.language_model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

for param in model.gen_head.parameters():
    param.requires_grad = False
    
if accelerator.is_main_process:
    model.language_model.print_trainable_parameters()
    
dataset = ColorDataset(data_path, processor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

model.train()
global_step = 0
epoch = 0

while global_step < max_steps:
    total_loss = 0
    
    for step, batch in enumerate(dataloader):
        text_ids = batch["text_ids"]
        image_tokens = batch["image_tokens"]
        
        unwrapped = accelerator.unwrap_model(model)
        text_embeds = unwrapped.language_model.get_input_embeddings()(text_ids)
        img_embeds = unwrapped.prepare_gen_img_embeds(image_tokens[:, :-1])
        inputs_embeds = torch.cat([text_embeds, img_embeds], dim=1)
        
        outputs = unwrapped.language_model.model.model(inputs_embeds=inputs_embeds)
        hidden = outputs.last_hidden_state
        
        img_hidden = hidden[:, text_ids.size(1):, :]
        logits = unwrapped.gen_head(img_hidden)
        
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            image_tokens[:, 1:].reshape(-1)
        )
        
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        
        total_loss += loss.item()
        global_step += 1
        
        if accelerator.is_main_process:
            wandb.log({
                "train/loss": loss.item(),
                "train/epoch": epoch,
                "train/step": global_step,
            })
        
        if step % 10 == 0 and accelerator.is_main_process:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        
        # 每500步保存一次checkpoint
        if global_step % save_interval == 0 and accelerator.is_main_process:
            checkpoint_dir = f"{output_dir}/checkpoint-{global_step}"
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.language_model.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_dir}")
        
        if global_step >= max_steps:
            break
    
    avg_loss = total_loss / (step + 1)
    
    if accelerator.is_main_process:
        wandb.log({
            "train/epoch_loss": avg_loss,
            "train/epoch": epoch,
        })
        print(f"Epoch {epoch} done, Avg Loss: {avg_loss:.4f}")
    
    epoch += 1

if accelerator.is_main_process:
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.language_model.save_pretrained(output_dir)
    
    wandb.save(f"{output_dir}/*")
    wandb.finish()
    
    print(f"Saved to {output_dir}")