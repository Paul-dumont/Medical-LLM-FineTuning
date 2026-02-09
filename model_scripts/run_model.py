import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_READ_TIMEOUT"] = "600"

import json
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
# import torch
from tqdm import tqdm # Barre de progression v  


#TO RUN:
table_number = 2
mode = "unknow"
eval_only = True  # Si False, génère sur tout le dataset (train + test)

print("-" * 95)
print(f" {mode}, Table {table_number}")
print("-" * 95)

# Determine max_seq_length based on mode
if mode == "tmj":
    max_seq_length = 6144  # TMJ needs 5015+ tokens, use 6K for safety
else:
    max_seq_length = 2048 #2048

# -----------------------------------------------------------------------------
# 1. Path Configuration 
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent #.resolve convert Relatif path into absolut path (./../home) into (~user/inux/home), .parent to keep the parent folder of the current file 
project_root = script_folder.parent # move up one level, to get the root project folder

model_path = str(project_root / "model"/f"Phi-3.5-mini-instruct_{mode}{table_number}")
json_path = str(project_root / "data"/"2_input_model"/f"{mode}"/f"training_data_{mode}{table_number}.jsonl")
output_path = str(project_root / "data" / "3_output_model" / f"{mode}" / f"extraction_{mode}{table_number}.jsonl")

# -----------------------------------------------------------------------------
# 2. Load Model
# -----------------------------------------------------------------------------
print("Load Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = None ,  #Unslot will automaticaly chose the best precision (bfloat16)
    load_in_4bit = False, # No compretion (Full 16-bit)
)

FastLanguageModel.for_inference(model) # Activate inference mode from unsloth (quicker)

# -----------------------------------------------------------------------------
# 3. Load Data (Same split as training)
# -----------------------------------------------------------------------------

print("Load Data...")
dataset = load_dataset("json", data_files=json_path, split="train") # Convertion in structured format + dataset mapping

if eval_only:
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    generation_dataset = dataset["test"]
    print(f"Eval on {len(dataset['test'])} notes")
else:
    generation_dataset = dataset
    print(f"Generation on {len(dataset)} notes (entire dataset)")

# -----------------------------------------------------------------------------
# 4. Generation 
# -----------------------------------------------------------------------------
print("Generation...")
result = []
valid_json_count = 0
invalid_json_count = 0

for patient_record in tqdm(generation_dataset): #loop on patient records, tqdm = progress bar
    prompt = patient_record["messages"][:-1] # Cut to keep onyl assitant : patient_record["messages"] looks like [System, User, Assistant]
    truth = patient_record["messages"][-1]["content"] # True features 
    
    # Extract metadata for traceability
    metadata = patient_record.get("metadata", {})
    patient_id = metadata.get("patient_id", "Unknown")
    note_date = metadata.get("note_date", "Unknown")
    note_month = metadata.get("note_month", "Unknown")
    
    # Convert datetime objects to string for JSON serialization
    if hasattr(note_date, 'isoformat'):
        note_date = note_date.isoformat()
    if hasattr(note_month, 'isoformat'):
        note_month = note_month.isoformat()

    input_ids = tokenizer.apply_chat_template(
        prompt,
        tokenize = True, 
        add_generation_prompt = True, # mode generation and not training
        return_tensors = "pt" # create tensor and not a List as classic python
    ).to("cuda") # stock this tensor in the GPU and not the RAM

    outputs = model.generate(
        input_ids = input_ids,
        max_new_tokens = 647, # Based on analysis: max JSON size is ~1065 tokens (unknow mode)
        use_cache = True,
        temperature = 0.0,
        do_sample = False,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id,
    )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean, keep only value and not prompt 
    start_index = generated_text.find("{")
    if start_index != -1:
        prediction_json = generated_text[start_index:]
    else: 
        prediction_json = "" #no json found 

    # Check JSON validity (counting only, no modification)
    try:
        json.loads(prediction_json)
        valid_json_count += 1
    except (json.JSONDecodeError, ValueError):
        invalid_json_count += 1

    # Save result with metadata
    result.append({
        "metadata": {
            "patient_id": patient_id,
            "note_date": note_date,
            "note_month": note_month
        },
        "original_note": prompt[-1]["content"], # notes
        "original": truth, # what we wanted 
        "prediction": prediction_json # what we have (model prediction)
    })

# Print quality baseline
total = len(generation_dataset)
valid_pct = 100 * valid_json_count / total if total > 0 else 0
print(f"\n{'='*60}")
print(f"BASELINE JSON Quality Report:")
print(f"  Valid JSON:   {valid_json_count}/{total} ({valid_pct:.1f}%)")
print(f"  Invalid JSON: {invalid_json_count}/{total} ({100-valid_pct:.1f}%)")
print(f"{'='*60}\n")


# -----------------------------------------------------------------------------
# 5. Save 
# -----------------------------------------------------------------------------
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as json_file:
    for item in result:
        json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved as {output_path}")

print("-" * 95)
print(f" {mode}, Table {table_number}")
print("-" * 95)

 