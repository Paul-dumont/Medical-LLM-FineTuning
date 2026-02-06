import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_READ_TIMEOUT"] = "600"

import json
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm


# TO RUN: Dry Run Configuration (BASELINE - NO TRAINING)
table_number = 1
# LLaMA 3.1 70B optimized via unsloth - SOTA baseline for extraction
# RTX Ada 6000 has 48GB VRAM - perfect for 70B models in 4bit
model_name = "unsloth/llama-3.1-70b-instruct"  # LLaMA 3.1 70B via unsloth (optimized version)
# Alternative options:
# - "unsloth/Meta-Llama-3.1-8B-Instruct"  # 8B if memory is limited
# - "unsloth/Qwen2.5-32B"           # 32B middle ground
# Using unsloth versions for optimization and compatibility


# -----------------------------------------------------------------------------
# 1. Path Configuration
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent

json_path = str(project_root / "data" / "2_input_model" / "dry_run" / f"training_data_dry_run{table_number}.jsonl")
output_path = str(project_root / "data" / "3_output_model" / "dry_run" / f"extraction_dry_run{table_number}.jsonl")

# Ensure output directory exists
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DRY RUN - SOTA BASELINE MODEL (NO TRAINING)")
print("Testing on EVAL set only (same as training split: 10% test)")
print("=" * 80)
print(f"Model: LLaMA 3.1 70B (unsloth/llama-3.1-70b-instruct) - SOTA Extraction Baseline")
print(f"GPU: RTX Ada 6000 (48GB VRAM - 4bit quantization)")
print(f"Input Data: {json_path}")
print(f"Output: {output_path}")
print(f"Note: This is a STRONG baseline - compare fine-tuned models against this!")
print("=" * 80)


# -----------------------------------------------------------------------------
# 2. Load Model (Base model from Hugging Face via unsloth, no fine-tuning)
# -----------------------------------------------------------------------------
print("\n[1/4] Loading Base Model from Hugging Face...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,  # Increased: input needs up to 1,583 tokens (550 system + 1,033 user)
        dtype=None,  # Auto-detect best dtype
        load_in_4bit=True,  # 4bit quantization for 70B model efficiency
    )
    print(f"✓ Base model loaded: {model_name}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

FastLanguageModel.for_inference(model)
print("✓ Inference mode activated")


# -----------------------------------------------------------------------------
# 3. Load Data (Same split as training: 10% eval, seed=42)
# -----------------------------------------------------------------------------
print("\n[2/4] Loading Data...")
try:
    dataset = load_dataset("json", data_files=json_path, split="train")
    print(f"✓ Dataset loaded: {len(dataset)} total samples")
    
    # Split with same seed and ratio as training
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    eval_dataset = dataset["test"]
    
    print(f"✓ Train/Test split (seed=42, test_size=10%)")
    print(f"  → Evaluation set: {len(eval_dataset)} samples")
    print(f"  → Training set (not used): {len(dataset['train'])} samples")
    
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)


# -----------------------------------------------------------------------------
# 4. Inference on EVAL set (Baseline Predictions)
# -----------------------------------------------------------------------------
print(f"\n[3/4] Running Baseline Inference on {len(eval_dataset)} eval samples...")
results = []
errors = 0

for patient_record in tqdm(eval_dataset):
    try:
        # Prepare prompt (exclude assistant message)
        prompt = patient_record["messages"][:-1]
        truth = patient_record["messages"][-1]["content"]
        
        # Tokenize
        input_ids = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        # Generate
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,  # Reduced: JSON extraction doesn't need long outputs
            use_cache=True,
            temperature=0.0,
            do_sample=False
        )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON
        start_index = generated_text.find("{")
        if start_index != -1:
            prediction_json = generated_text[start_index:]
        else:
            prediction_json = ""
        
        # Store result (same format as run_model.py)
        results.append({
            "original_note": prompt[-1]["content"],
            "original": truth,
            "prediction": prediction_json
        })
        
    except Exception as e:
        errors += 1
        results.append({
            "original_note": "ERROR",
            "original": "ERROR",
            "prediction": f"ERROR: {str(e)}"
        })

print(f"✓ Inference completed: {len(results) - errors}/{len(eval_dataset)} successful")
if errors > 0:
    print(f"⚠ Errors: {errors}")


# -----------------------------------------------------------------------------
# 5. Save Results (same format as run_model.py for comparison)
# -----------------------------------------------------------------------------
print(f"\n[4/4] Saving Results...")
try:
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"✓ Results saved to: {output_path}")
except Exception as e:
    print(f"✗ Error saving results: {e}")
    exit(1)


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("DRY RUN COMPLETED - BASELINE RESULTS")
print("=" * 80)
print(f"Eval samples processed: {len(results)}")
print(f"Successful predictions: {len(results) - errors}")
print(f"Failed predictions: {errors}")
print(f"\nOutput file: {output_path}")
print(f"\nYou can now compare with trained model results:")
print(f"  → Baseline (same for both modes): {output_path}")
print(f"  → Training with_cot: extraction_with_cot{table_number}.jsonl")
print(f"  → Training without_cot: extraction_without_cot{table_number}.jsonl")
print("=" * 80)
