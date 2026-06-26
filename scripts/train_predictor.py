import os
import copy
import yaml
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. Environment Setup (Preserved from original script)
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 2. Parse Arguments and Load Config
    parser = argparse.ArgumentParser(description="Train EnergyServe's Perception Model")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--dataset", type=str, required=True, choices=["lmsys", "alpaca", "dolly"])
    args = parser.parse_args()
    
    config = load_config(args.config)
    dataset_config = config['datasets'][args.dataset]
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f">>> Initializing Tokenizer ({config['model']['base_model']})...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    print(f">>> Loading base model ({config['model']['base_model']})...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", 
        device_map={"": local_rank}
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    print(f">>> Configuring LoRA (Rank {config['lora']['r']})...")
    peft_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        target_modules=config['lora']['target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    
    if local_rank == 0:
        model.print_trainable_parameters()

    print(f">>> Loading dataset: {dataset_config['train_path']}")
    dataset = load_dataset("json", data_files={"train": dataset_config['train_path']})
    
    # Core Logic: Masking for Gemma's prompt format (Preserved from original script)
    def tokenize_function(example):
        user_prompt = example["input"]
        prompt_text = f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        model_answer = str(example["output"])
        full_text = prompt_text + f"{model_answer}<end_of_turn>"
        
        tokenized_full = tokenizer(full_text, truncation=True, max_length=config['model']['max_length'])
        tokenized_prompt = tokenizer(prompt_text, truncation=True, max_length=config['model']['max_length'])
        
        input_ids = tokenized_full["input_ids"]
        labels = copy.deepcopy(input_ids)
        
        # Masking: Set labels of the prompt part to -100 to ignore in loss calculation
        prompt_len = len(tokenized_prompt["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
            
        return {"input_ids": input_ids, "attention_mask": tokenized_full["attention_mask"], "labels": labels}
    
    tokenized_dataset = dataset["train"].map(tokenize_function, remove_columns=dataset["train"].column_names)
    
    print(">>> Configuring Training Arguments...")
    train_cfg = config['training']
    training_args = TrainingArguments(
        output_dir=dataset_config['output_dir'],
        per_device_train_batch_size=train_cfg['per_device_train_batch_size'],
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
        num_train_epochs=train_cfg['epochs'],
        learning_rate=float(train_cfg['learning_rate']),
        bf16=train_cfg['bf16'],
        logging_steps=train_cfg['logging_steps'],
        save_strategy="steps",
        save_steps=train_cfg['save_steps'],
        save_total_limit=train_cfg['save_total_limit'],
        deepspeed=train_cfg['deedspeed_config'],
        gradient_checkpointing=True,
        group_by_length=True,
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, data_collator=data_collator, tokenizer=tokenizer)

    print(f">>> START TRAINING (Dataset: {args.dataset.upper()})")
    trainer.train()
    
    print(">>> Saving final LoRA adapter...")
    final_output_dir = os.path.join(dataset_config['output_dir'], "final_lora")
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"✅ Training for {args.dataset.upper()} complete. Model saved to {final_output_dir}")

if __name__ == "__main__":
    main()