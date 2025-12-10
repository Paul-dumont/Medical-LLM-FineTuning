import json
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer 
from transformers import TrainingArguments
import wandb

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
max_seq_length = 4096 
run_name = "Qwen_test1"

# -----------------------------------------------------------------------------
# 2. Load Model 
# -----------------------------------------------------------------------------
print("Load Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name= "unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length = max_seq_length,
    dtype = None ,  #Unslot will automaticaly chose the best precision (bfloat16)
    load_in_4bit = False, # No compretion (Full 16-bit)
)

# LoRA Configuration 
model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # r = The Rank. how much new information
    # We allowed the model to change all those layers (Full LoRA)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32, # The Strength of those new information, how strongly new learning is applied
    lora_dropout = 0, # Randomly turn off neurons, to prevent memorization 
    bias = "none", # don't train the bis in order to save memory 
    use_gradient_checkpointing = "unsloth",
    random_state = 3407 # Seed 
)

# -----------------------------------------------------------------------------
# 3. Data 
# -----------------------------------------------------------------------------
print("Load Data...")
json_path = "training_data_CoT.jsonl"
dataset = load_dataset("json", data_files=json_path, split="train") # Convertion in structured format + dataset mapping
print("\n")
print(f"Load {len(dataset)} cases")
print("\n")


# Prepare text for the LLM
def formatting_prompt_func(data_batch):
    list_of_records = data_batch["messages"]
    formatted_texts = [] 
    
    # Loop through each patient record 
    for patient_record in list_of_records:
        # Apply the Qwen chat template to the record
        # transforms JSON structure (System/User/Assistant) into a string with special tags (<|im_start|>)
        formatted_text = tokenizer.apply_chat_template(
            patient_record, 
            tokenize=False, # Keep it as text for now (don't convert to numbers yet)
            add_generation_prompt=False # We are training, so we provide the full answer
        )
        formatted_texts.append(formatted_text)
    # Return the new column named "text" required by the trainer
    return { "text" : formatted_texts }


# Before (List): [{"role": "system","content": "expert Orthodontist}]
# After (String): "<|im_start|>system expert Orthodontist.<|im_end|>"" 
dataset = dataset.map(formatting_prompt_func, batched = True)

# Split train/eval => dataset become a dict : {train, Test}
dataset = dataset.train_test_split(test_size=0.1,seed=42)
print(f"{len(dataset['train'])} Training samples")
print(f"{len(dataset['test'])} Evaluation samples")

# -----------------------------------------------------------------------------
# 4. Training 
# -----------------------------------------------------------------------------
print("Start Fine Tuning...")
wandb.login()
trainer = SFTTrainer( 
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    dataset_text_field = "text", # Name of the Format column "text"
    max_seq_length = max_seq_length,
    dataset_num_proc = 12, # Number of CPU Core use for tokenization  
    packing = False, # We dont want to merge patient records to fit the max_seq_length window
    
    args = TrainingArguments(

        # Performance (RTX 6000)
        per_device_train_batch_size = 24, # Size of batch (patient case load simultaneously for evaluation) 
        per_device_eval_batch_size = 12,
        gradient_accumulation_steps = 1, # Number of batch before update ( 8(batch) * 2(accum) = 16)
        learning_rate = 2e-4, # Always 2e-4 for LoRA 1e-3=> (crazy,forgot everything) 1e-5(too long to train, learn nothing)
        fp16 = False, # Disable the older 16bit format 
        bf16 = True, # Enable modern Gold Standart format 
        optim = "adamw_8bit",

        # Time for training
        num_train_epochs = 3, # Read the whole dataset 3 time (1 for testing) 
        warmup_steps = 10, # time of slow training (To not delete previus knowledge) 
        

        # Log (WandB)
        logging_steps = 1, # Terminal information frequence
        report_to = "wandb",
        run_name = "Qwen_test_1",

        # Eval and Save
        eval_strategy = "steps",
        eval_steps = 50, 
        save_strategy = "no", # "steps" if real run, test mode: "no"
        save_steps = 50, 
        save_total_limit = 2, 
        group_by_length = True,
        load_best_model_at_end = False, # "True if real run, test mode: "False"
        metric_for_best_model = "eval_loss",

        # Other 
        output_dir = "outputs", # ? 
        weight_decay = 0.01, # Counter overfitting  
        lr_scheduler_type = "linear", #?
        seed = 3407,    
    ),
)

# Training 
trainer_stats = trainer.train()

# -----------------------------------------------------------------------------
# 5. Save 
# -----------------------------------------------------------------------------
# print("Save Model...")
# model.save_pretrained("lora_model_qwen")
# tokenizer.save_pretrained("lora_model_qwen")
print("Succes")
