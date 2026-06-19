#!/usr/bin/env python3
"""
Simple Benchmark - no_prompt table 4 by default
"""

import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_READ_TIMEOUT"] = "600"

import time
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM
import wandb
import sys
from pathlib import Path

# Add model_scripts to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import grouped_shuffle_split


def benchmark(model_name: str = "phi"):
    """Train on training_data_no_prompt4.jsonl - EXACT replica of train_model.py but 50% faster"""
    
    print("\n" + "=" * 80)
    print(f"BENCHMARK (EXACT REPLICA): {model_name.upper()} | no_prompt table 4")
    print("=" * 80)
    
    # Paths
    data_file = Path(__file__).parent.parent.parent / "data" / "2_input_model" / "no_prompt" / "training_data_no_prompt4.jsonl"
    
    if not data_file.exists():
        print(f"❌ Not found: {data_file}")
        return None
    
    # Model configs - Best 7/8B models for clinical information extraction
    model_configs = {
        # ==========================================
        # 1. LES GÉNÉRALISTES SOTA (Raisonnement & Instructions)
        # ==========================================
        "llama_3_1": {
            "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        },
        "qwen_2_5": {
            "model_name": "unsloth/Qwen2.5-7B-Instruct",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "response_template": "<|im_start|>assistant\n",
        },
        "gemma_2": {
            "model_name": "unsloth/gemma-2-9b-it",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "response_template": "<start_of_turn>model\n",
        },
        "mistral_v0_3": { # Version 0.3 supérieure à la 0.2
            "model_name": "unsloth/mistral-7b-instruct-v0.3",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "response_template": "[/INST]", 
        },
        "phi_3_5": {
            "model_name": "unsloth/Phi-3.5-mini-instruct",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "response_template": "<|assistant|>\n",
        },
    }
    
    config = model_configs[model_name]
    run_name = f"bench_{model_name}"
    
    # =========================================================================
    # 2. LOAD MODEL
    # =========================================================================
    print("\n📦 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    
    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=config["target_modules"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407
    )
    print("✓ Model loaded")
    
    # =========================================================================
    # 3. LOAD DATA
    # =========================================================================
    print("\n📂 Loading data...")
    dataset = load_dataset("json", data_files=str(data_file), split="train")
    print(f"✓ Loaded {len(dataset)} examples")
    
    # Format for chat (exactly like train_model.py)
    def format_single_patient(patient_record):
        conversation = patient_record["messages"]
        formatted_texts = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False,
            add_generation_prompt=False
        )
        return { "text" : formatted_texts }
    
    dataset = dataset.map(format_single_patient)
    
    # Split train/val/test using Grouped Shuffle Split (patient-level integrity)
    # BUT: For faster benchmark, use only 50% of train data
    split_result = grouped_shuffle_split(
        dataset, 
        train_ratio=0.35,  # 50% of 70% = 35% of total
        val_ratio=0.15, 
        test_ratio=0.15, 
        seed=42
    )
    train_dataset = split_result['train']
    eval_dataset = split_result['validation']
    
    # =========================================================================
    # 4. TRAINING
    # =========================================================================
    print("\n🚀 Training...")
    wandb.login()
    
    # Use correct response template (like train_model.py)
    if model_name == "phi_3_5":
        response_template = "<|assistant|>\n"
    else:  # All others
        response_template = config.get("response_template", "[/INST]")
    
    # Create DataCollator to mask input tokens (CRITICAL!)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, 
        tokenizer=tokenizer
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        data_collator=data_collator,  # ✅ NOW included
        dataset_num_proc=8,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,  # Exact match with train_model.py
            learning_rate=2e-4,
            fp16=False,
            bf16=True,
            optim="adamw_8bit",
            num_train_epochs=1,
            warmup_ratio=0.148,  # Exact match
            logging_steps=1,
            report_to="wandb",
            run_name=run_name,
            group_by_length=True,  # Exact match
            eval_strategy="steps",
            eval_steps=5,  # Exact match
            save_strategy="steps",
            save_total_limit=3,  # Exact match
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            output_dir="/tmp/bench",
            weight_decay=0.02,  # Exact match
            lr_scheduler_type="cosine",  # Exact match
            seed=3407,  # Exact match
            remove_unused_columns=False,  # Keep text field for data_collator
        ),
    )
    
    stats = trainer.train()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # =========================================================================
    # 5. RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    
    num_train_samples = len(train_dataset)
    num_epochs = 1
    processed_samples = num_train_samples * num_epochs
    throughput = (processed_samples / elapsed) if elapsed > 0 else 0
    
    print(f"⏱️  Time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"📊 Throughput: {throughput:.2f} samples/sec")
    print(f"📈 Train loss: {stats.training_loss:.4f}")
    print(f"✅ Benchmark completed - check wandb for full logs")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["gemma_2", "llama_3_1", "qwen_2_5", "mistral_v0_3", "phi_3_5"], # ⬅️ ON NE GARDE QUE CEUX QUI EXISTENT
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print(f"🔬 BENCHMARK SUITE: {len(args.models)} models")
    print("=" * 80)
    
    results = []
    for i, model in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] Testing {model.upper()}...")
        try:
            benchmark(model)
            results.append({"model": model, "status": "✅ Success"})
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({"model": model, "status": f"❌ Failed: {str(e)[:50]}"})
    
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    for r in results:
        print(f"  {r['model']:10s} → {r['status']}")
    print("=" * 80)
