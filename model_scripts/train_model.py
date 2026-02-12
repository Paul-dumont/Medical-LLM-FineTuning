import json
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
import wandb
from pathlib import Path
import shutil


def main(table_number: int, mode: str):
    """
    Main function for training the model.
    
    Args:
        table_number: Table number (1-4)
        mode: Training mode (no_prompt, with_cot, without_cot, tmj, dry_run)
    """
    print("-" * 95)
    print(f" {mode}, Table {table_number}")
    print("-" * 95)

    # Determine max_seq_length based on mode
    if mode == "tmj":
        max_seq_length = 6144
    else:
        max_seq_length = 2048

    # -------------------------------------------------------------------------
    # 1. Path Configuration
    # -------------------------------------------------------------------------
    script_folder = Path(__file__).resolve().parent
    project_root = script_folder.parent

    json_path = str(project_root/"data"/"2_input_model"/f"{mode}"/f"training_data_{mode}{table_number}.jsonl")
    output_dir = str(project_root / "model")
    run_name = f"Phi-3.5-mini-instruct_{mode}{table_number}"

    # -------------------------------------------------------------------------
    # 2. Load Model
    # -------------------------------------------------------------------------
    print("Load Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name= "unsloth/Phi-3.5-mini-instruct",
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = False,
    )

    # -------------------------------------------------------------------------
    # 3. LoRA Configuration 
    # -------------------------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
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

    # Split train/val/test => 70/15/15
    dataset = dataset.train_test_split(test_size=0.3, seed=42)
    val_test = dataset['test'].train_test_split(test_size=0.5, seed=42)
    dataset = {
        'train': dataset['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    }
    print(f"{len(dataset['train'])} Training samples")
    print(f"{len(dataset['validation'])} Validation samples")
    print(f"{len(dataset['test'])} Test samples")

    # -------------------------------------------------------------------------
    # 5. Training 
    # -------------------------------------------------------------------------
    response_template = "<|assistant|>\n"
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
            per_device_train_batch_size = 4,  # Reduced from 16
            per_device_eval_batch_size = 4,   # Reduced from 16
            gradient_accumulation_steps = 4,  # Increased from 1 to compensate
            learning_rate = 2e-4,
            fp16 = False,
            bf16 = True,
            optim = "adamw_8bit",

            num_train_epochs = 3,
            warmup_steps = 5,
            
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
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,    
        ),
    )

    # Training 
    trainer_stats = trainer.train()

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
    print(f" {mode}, Table {table_number}")
    print("-" * 95)


if __name__ == "__main__":
    main(table_number=1, mode="no_prompt")
