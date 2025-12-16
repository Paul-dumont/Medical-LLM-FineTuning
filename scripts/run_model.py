import json
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
# import torch
from tqdm import tqdm # Barre de progression

# -----------------------------------------------------------------------------
# 1. Path Configuration
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent #.resolve convert Relatif path into absolut path (./../home) into (~user/inux/home), .parent to keep the parent folder of the current file 
project_root = script_folder.parent # move up one level, to get the root project folder

model_path = str(project_root / "model"/"Phi-3.5-mini-instruct")
json_path = str(project_root / "data"/"json"/"training_data_2.jsonl")
output_path = str(project_root / "output" / "extraction.jsonl")

# -----------------------------------------------------------------------------
# 2. Load Model
# -----------------------------------------------------------------------------
print("Load Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 1024, # 1024 for janson patient 4096 for max patient
    dtype = None ,  #Unslot will automaticaly chose the best precision (bfloat16)
    load_in_4bit = False, # No compretion (Full 16-bit)
)

FastLanguageModel.for_inference(model) # Activate inference mode from unsloth (quicker)

# -----------------------------------------------------------------------------
# 3. Load Data (Same split as training)
# -----------------------------------------------------------------------------

print("Load Data...")
dataset = load_dataset("json", data_files=json_path, split="train") # Convertion in structured format + dataset mapping
dataset = dataset.train_test_split(test_size=0.1,seed=42)
eval_dataset = dataset["test"]
print(f"Eval on {len(dataset['test'])} notes ")

# -----------------------------------------------------------------------------
# 4. Generation 
# -----------------------------------------------------------------------------
print("Generation...")
result = []

for patient_record in tqdm(eval_dataset): #loop on patient records, tqdm = progress bar
    prompt = patient_record["messages"][:-1] # Cut to keep onyl assitant : patient_record["messages"] looks like [System, User, Assistant]
    truth = patient_record["messages"][-1]["content"] # True features 

    input_ids = tokenizer.apply_chat_template(
        prompt,
        tokenize = True, 
        add_generation_prompt = True, # mode generation and not training
        return_tensors = "pt" # create tensor and not a List as classic python
    ).to("cuda") # stock this tensor in the GPU and not the RAM

    outputs = model.generate(
        input_ids = input_ids,
        max_new_tokens = 1024, #a verifier 
        use_cache = True,
        temperature = 0.0,
        do_sample = False
    )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean, keep only value and not prompt 
    start_index = generated_text.find("{")
    if start_index != -1:
        prediction_json = generated_text[start_index:]
    else: 
        prediction_json = "" #no json found 

    # Save result 
    result.append({
        "orignal_note": prompt[-1]["content"], # notes
        "extraction": truth, # what we wanted 
        "prediction": prediction_json # what we have (model prediction)
    })


# -----------------------------------------------------------------------------
# 5. Save 
# -----------------------------------------------------------------------------
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as json_file:
    for item in result:
        json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved as {output_path}")

 