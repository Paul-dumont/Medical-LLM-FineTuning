import json
import time
import statistics
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
import wandb
from pathlib import Path
import shutil
import sys

# Add model_scripts to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import grouped_shuffle_split


def main(table_number: int, mode: str, model_type: str = "phi"):
    """
    Main function for training the model.
    
    Args:
        table_number: Table number (1-4)
        mode: Training mode (no_prompt, with_cot, without_cot, tmj, dry_run)
        model_type: Model to use ("phi" or "llama")
    """
    print("-" * 95)
    print(f" {mode}, Table {table_number}, Model: {model_type}")
    print("-" * 95)

    # Determine max_seq_length based on mode
    if mode == "tmj":
        max_seq_length = 6144
    else:
        max_seq_length = 2048

    # Model configuration
    model_configs = {
        "phi": {
            "model_name": "unsloth/Phi-3.5-mini-instruct",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "llama": {
            "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        }
    }
    
    config = model_configs[model_type]

    # -------------------------------------------------------------------------
    # 1. Path Configuration
    # -------------------------------------------------------------------------
    script_folder = Path(__file__).resolve().parent
    project_root = script_folder.parent

    json_path = str(project_root/"data"/"2_input_model"/f"{mode}"/f"training_data_{mode}{table_number}.jsonl")
    output_dir = str(project_root / "model")
    run_name = f"{config['model_name'].split('/')[-1]}_{mode}{table_number}"

    # -------------------------------------------------------------------------
    # 2. Load Model
    # -------------------------------------------------------------------------
    print("Load Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = False,
    )

    # -------------------------------------------------------------------------
    # 3. LoRA Configuration 
    # -------------------------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = config["target_modules"],
        lora_alpha = 64,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407
    )

    # -------------------------------------------------------------------------
    # 4. Data 
    # -------------------------------------------------------------------------
    print("Load Data...")
    dataset = load_dataset("json", data_files=json_path, split="train")
    print("\n")
    print(f"Load {len(dataset)} cases")
    print("\n")

    # Prepare text for the LLM
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
    split_result = grouped_shuffle_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    dataset = {
        'train': split_result['train'],
        'validation': split_result['validation'],
        'test': split_result['test']
    }

    # -------------------------------------------------------------------------
    # 5. Training 
    # -------------------------------------------------------------------------
    # Adapt response template based on model
    if model_type == "phi":
        response_template = "<|assistant|>\n"
    else:  # llama
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, 
        tokenizer=tokenizer
    )

    print("Start Fine Tuning...")
    wandb.login()
    trainer = SFTTrainer( 
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["validation"],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = collator,
        dataset_num_proc = 12,
        packing = False,
        
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            per_device_eval_batch_size = 4,
            gradient_accumulation_steps = 2,
            learning_rate = 2e-4,
            fp16 = False,
            bf16 = True,
            optim = "adamw_8bit",

            num_train_epochs = 3,
            warmup_ratio = 0.148,
            
            logging_steps = 1,
            report_to = "wandb",
            run_name = run_name,

            group_by_length = True,

            eval_strategy = "steps",
            eval_steps = 5,
            save_strategy ="steps",
            save_total_limit = 3,

            load_best_model_at_end = True,
            metric_for_best_model = "eval_loss",

            output_dir = output_dir,
            overwrite_output_dir = True,  
            weight_decay = 0.02,
            lr_scheduler_type = "cosine",
            seed = 3407,    
        ),
    )

    # Training 
    # Warm GPU sync then time the full training run (includes all epochs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_t0 = time.perf_counter()
    trainer_stats = trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_t1 = time.perf_counter()

    total_train_time = train_t1 - train_t0
    # Calculate approximate time per note (across all epochs)
    num_train_notes = len(dataset['train'])
    num_epochs = getattr(trainer.args, 'num_train_epochs', 1)
    processed_notes = num_train_notes * num_epochs if num_train_notes > 0 else 0

    # Throughput (samples/sec)
    throughput = (processed_notes / total_train_time) if (processed_notes > 0 and total_train_time > 0) else None

    # Convergence info (epochs and global steps when available)
    global_steps = None
    try:
        global_steps = getattr(trainer.state, 'global_step', None)
    except Exception:
        global_steps = None

    # VRAM / peak GPU memory
    vram_str = "Unavailable"
    devices = None
    gpu_name = None
    try:
        if torch.cuda.is_available():
            devices = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            peak = None
            try:
                peak = torch.cuda.max_memory_reserved()
            except Exception:
                try:
                    peak = torch.cuda.max_memory_allocated()
                except Exception:
                    peak = None
            if peak:
                vram_str = f"{peak/1024**3:.2f} GB VRAM on {devices}x {gpu_name}"
            else:
                vram_str = f"GPU present ({devices}x {gpu_name}), peak memory unavailable"
        else:
            vram_str = "CPU only"
    except Exception:
        vram_str = "Unavailable"

    # Training time string (display in minutes)
    minutes = total_train_time / 60.0
    if torch.cuda.is_available() and devices is not None and gpu_name is not None:
        train_time_str = f"{minutes:.2f} minutes on {devices}x {gpu_name}"
    else:
        train_time_str = f"{minutes:.2f} minutes on CPU"

    # Print only the four requested metrics (French labels)
    if throughput is not None:
        throughput_str = f"{throughput:.3f} samples/s"
    else:
        throughput_str = "indisponible"

    if global_steps is not None:
        convergence_str = f"Epochs: {num_epochs}, Global steps: {global_steps}"
    else:
        convergence_str = f"Epochs: {num_epochs}"

    print("-" * 95)
    print(f"Training Time (Temps total) : {train_time_str}")
    print(f"Vitesse d'apprentissage (Throughput) : {throughput_str}")
    print(f"Convergence Speed : {convergence_str}")
    print(f"VRAM Usage : {vram_str}")
    print("-" * 95)

    # -------------------------------------------------------------------------
    # 6. Save 
    # -------------------------------------------------------------------------
    print("Save Merge Model (Lora + Base) ...")
    model_save_path = Path(output_dir)/run_name
    model_save_path.mkdir(parents=True, exist_ok=True)

    # Save LoRA + base model
    model.save_pretrained_merged(
        str(model_save_path),
        tokenizer,
        save_method = "merged_16bit"
    )

    # Clean Checkpoints 
    for folder in Path(output_dir).glob("checkpoint-*"):
        try:
            shutil.rmtree(folder)
        except:
            pass
    print("Success")

    print("-" * 95)
    print(f" {mode}, Table {table_number}, Model: {model_type}")
    print("-" * 95)


if __name__ == "__main__":
    main(table_number=4, mode="no_prompt", model_type="llama")
