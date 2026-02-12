import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_READ_TIMEOUT"] = "600"

import json
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm


def main(table_number: int, mode: str, eval_set: str = "test"):
    """
    Main function for running the model and generating predictions.
    
    Args:
        table_number: Table number (1-4)
        mode: Training mode (no_prompt, with_cot, without_cot, tmj, dry_run)
        eval_set: Evaluation set to use ("validation" or "test")
    """
    print("-" * 95)
    print(f" {mode}, Table {table_number}")
    print("-" * 95)

    # Determine max_seq_length based on mode
    if mode == "tmj":
        max_seq_length = 6144
    else:
        max_seq_length = 2048

    print(f"Evaluating on {eval_set} set")

    # Path Configuration 
    # -------------------------------------------------------------------------
    script_folder = Path(__file__).resolve().parent
    project_root = script_folder.parent

    model_path = str(project_root / "model" / f"Phi-3.5-mini-instruct_{mode}{table_number}")
    json_path = str(project_root / "data" / "2_input_model" / f"{mode}" / f"training_data_{mode}{table_number}.jsonl")
    output_path = str(project_root / "data" / "3_output_model" / f"{mode}" / f"extraction_{mode}{table_number}.jsonl")

    # -------------------------------------------------------------------------
    # 1. Load Model
    # -------------------------------------------------------------------------
    print("Load Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    FastLanguageModel.for_inference(model)

    # -------------------------------------------------------------------------
    # 2. Load Data (Same split as training)
    # -------------------------------------------------------------------------
    print("Load Data...")
    dataset = load_dataset("json", data_files=json_path, split="train")

    # Apply same 70/15/15 split as training
    dataset = dataset.train_test_split(test_size=0.3, seed=42)
    val_test = dataset['test'].train_test_split(test_size=0.5, seed=42)
    splits = {
        'train': dataset['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    }

    generation_dataset = splits[eval_set]
    print(f"Evaluating on {eval_set.upper()} set: {len(generation_dataset)} notes")
    print(f"  Train: {len(splits['train'])} | Validation: {len(splits['validation'])} | Test: {len(splits['test'])}")

    # -------------------------------------------------------------------------
    # 3. Generation 
    # -------------------------------------------------------------------------
    print("Generation...")
    result = []
    valid_json_count = 0
    invalid_json_count = 0

    for patient_record in tqdm(generation_dataset):
        prompt = patient_record["messages"][:-1]
        truth = patient_record["messages"][-1]["content"]
        
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
            tokenize=True, 
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=647,
            use_cache=True,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean, keep only value and not prompt 
        start_index = generated_text.find("{")
        if start_index != -1:
            prediction_json = generated_text[start_index:]
        else: 
            prediction_json = ""

        # Check JSON validity
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
            "original_note": prompt[-1]["content"],
            "original": truth,
            "prediction": prediction_json
        })

    # Print quality baseline
    total = len(generation_dataset)
    valid_pct = 100 * valid_json_count / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"BASELINE JSON Quality Report:")
    print(f"  Valid JSON:   {valid_json_count}/{total} ({valid_pct:.1f}%)")
    print(f"  Invalid JSON: {invalid_json_count}/{total} ({100-valid_pct:.1f}%)")
    print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # 4. Save 
    # -------------------------------------------------------------------------
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as json_file:
        for item in result:
            json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved as {output_path}")

    print("-" * 95)
    print(f" {mode}, Table {table_number}")
    print("-" * 95)


if __name__ == "__main__":
    main(table_number=1, mode="no_prompt", eval_set="test")
