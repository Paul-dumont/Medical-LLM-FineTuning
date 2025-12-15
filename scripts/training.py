import json
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
import wandb
from pathlib import Path
import shutil



# -----------------------------------------------------------------------------
# 1. Path Configuration
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent #.resolve convert Relatif path into absolut path (./../home) into (~user/inux/home), .parent to keep the parent folder of the current file 
project_root = script_folder.parent # move up one level, to get the root project folder

json_path = str(project_root/"data"/"json"/"training_data_2.jsonl")
output_dir = str(project_root / "outputs")


# -----------------------------------------------------------------------------
# 1. Global Configuration
# -----------------------------------------------------------------------------
max_seq_length = 1024 # 1024 for janson patient 4096 for max patient
run_name = "Phi-3.5-mini-instruct"

# -------------------- ---------------------------------------------------------
# 2. Load Model 
# -----------------------------------------------------------------------------
print("Load Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name= "unsloth/Phi-3.5-mini-instruct",
    max_seq_length = max_seq_length,
    dtype = None ,  #Unslot will automaticaly chose the best precision (bfloat16)
    load_in_4bit = False, # No compretion (Full 16-bit)
)

# -------------------- ---------------------------------------------------------
# 3. LoRA Configuration 
# -----------------------------------------------------------------------------
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
dataset = load_dataset("json", data_files=json_path, split="train") # Convertion in structured format + dataset mapping
print("\n")
print(f"Load {len(dataset)} cases")
print("\n")

# Prepare text for the LLM
def format_single_patient(patient_record):
    conversation = patient_record["messages"]

    # Apply the chat template to the record
    # transforms JSON structure (System/User/Assistant) into a string with special tags (<|im_start|>)
    formatted_texts = tokenizer.apply_chat_template(
        conversation, 
        tokenize=False, # Keep it as text for now (don't convert to numbers yet)
        add_generation_prompt=False # We are training, so we provide the full answer
    )
    # return result in a new column "text"
    return { "text" : formatted_texts }

# Before (List): [{"role": "system","content": "expert Orthodontist}]
# After (String): "<|im_start|>system expert Orthodontist.<|im_end|>"" 
dataset = dataset.map(format_single_patient)

# Split train/eval => dataset become a dict : {train, Test}
dataset = dataset.train_test_split(test_size=0.1,seed=42)
print(f"{len(dataset['train'])} Training samples")
print(f"{len(dataset['test'])} Evaluation samples")

# -----------------------------------------------------------------------------
# 4. Training 
# -----------------------------------------------------------------------------
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
    eval_dataset = dataset["test"],
    dataset_text_field = "text", # Name of the Format column "text"
    max_seq_length = max_seq_length,
    data_collator = collator,
    dataset_num_proc = 12, # Number of CPU Core use for tokenization  
    packing = False, # We dont want to merge patient records to fit the max_seq_length window
    
    args = TrainingArguments(
        # 1 step = 1 Batch = 24 patient

        # Performance (RTX 6000)
        per_device_train_batch_size = 4, # Size of batch (patient case load simultaneously for training) 
        per_device_eval_batch_size = 4,    # Size of batch (patient case load simultaneously for evaluation) 
        gradient_accumulation_steps = 4, # Number of batch before update ( (batch)4 * (accum)4 = 16)
        learning_rate = 2e-4, # Always 2e-4 for LoRA 1e-3=> (crazy,forgot everything) 1e-5(too long to train, learn nothing)
        fp16 = False, # Disable the older 16bit format 
        bf16 = True, # Enable modern Gold Standart format 
        optim = "adamw_8bit",

        # Time for training
        num_train_epochs = 3, # Read the whole dataset 3 time (1 for testing) 
        warmup_steps = 5, # time of slow training (To not delete previus knowledge) 
        
        # Log (WandB)
        logging_steps = 1, # Terminal information frequence
        report_to = "wandb",
        run_name = run_name, # Name for log and wanbd 

        # Eval and Save
        group_by_length = True, # group short notes together et logn notes together 

        eval_strategy = "steps", # "steps" if real run, test mode: "no"
        eval_steps = 5,
        save_strategy ="steps",
        save_total_limit = 3, # Save disk storage, keep last chekpoint

        load_best_model_at_end = True, # "True if real run, test mode: "False"
        metric_for_best_model = "eval_loss", # gold standart

        # Other 
        output_dir = output_dir,
        weight_decay = 0.01, # Counter overfitting  
        lr_scheduler_type = "linear", # gold standart
        seed = 3407,    
    ),
)

# Training 
trainer_stats = trainer.train()

# -----------------------------------------------------------------------------
# 5. Save 
# -----------------------------------------------------------------------------
print("Save Merge Model (Lora + Base) ...")
model_save_path = Path(output_dir)/run_name
model_save_path.mkdir(parents=True, exist_ok=True) # Create all folder necessary, et if all allready exist, do not create error  

# Save LoRA + base model
model.save_pretrained_merged(
    str(model_save_path),
    tokenizer,
    save_method = "json"
)

# clean Checkpoints 
for folder in Path(output_dir).glob("checkpoint-*"):
    try:
        shutil.rmtree(folder)
    except:
        pass
print("Succes")


