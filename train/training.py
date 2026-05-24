import modal
import os, torch
import wandb
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from train.app import app, VOL, volume, DATA_DIR, OUTPUT_DIR, CKPT_DIR



@app.function(
    gpu="A100-40GB",         
    volumes={VOL: volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
    timeout=7200,
    )

def train():
    
    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
    wandb.init(
        project="llama-review-finetune",
        name="llama3.2-1b--lora",
        config={
            "model":      MODEL_ID,
            "task":       "persona to predict rating+review",
            "lora_r":     16,
            "lora_alpha": 32,
            "epochs":     3,
            "batch_size": 4,
            "grad_accum": 4,
            "lr":         2e-4,
        },
    )
    volume.reload()
    train_dataset = load_from_disk(f"{DATA_DIR}/train")
    val_dataset   = load_from_disk(f"{DATA_DIR}/val")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, token=os.environ["HF_TOKEN"]
    )
    tokenizer.pad_token = tokenizer.eos_token
 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
    )
    model.config.use_cache = False
 
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
 
    sft_config = SFTConfig(
        output_dir=CKPT_DIR,
        num_train_epochs=100,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=512,
        dataset_text_field="text",
        report_to="wandb",
        greater_is_better=False
    )
 
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        args=sft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    trainer.train()
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    volume.commit()
 
    wandb.finish()

 