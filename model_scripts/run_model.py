import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_READ_TIMEOUT"] = "600"

import json
import time
import statistics
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm
import torch
import sys

# Add model_scripts to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import grouped_shuffle_split

# (No energy/power monitoring required)


def main(table_number: int, mode: str, eval_set: str = "test", model_type: str = "phi"):
    """
    Main function for running the model and generating predictions.
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
        "phi": {"model_name": "unsloth/Phi-3.5-mini-instruct",},
        "llama": {"model_name": "unsloth/Meta-Llama-3.1-8B-Instruct",}
    }
    config = model_configs[model_type]

    print(f"Evaluating on {eval_set} set")

    # Path Configuration
    script_folder = Path(__file__).resolve().parent
    project_root = script_folder.parent

    model_path = str(project_root / "model" / f"{config['model_name'].split('/')[-1]}_{mode}{table_number}")
    json_path = str(project_root / "data" / "2_input_model" / f"{mode}" / f"training_data_{mode}{table_number}.jsonl")
    if model_type == "llama":
        output_path = str(project_root / "data" / "3_output_model" / f"{mode}" / f"extraction_llama_{mode}{table_number}.jsonl")
    else:
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
    dataset = grouped_shuffle_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    generation_dataset = dataset[eval_set]
    print(f"Evaluating on {eval_set.upper()} set: {len(generation_dataset)} notes")
    print(f"  Train: {len(dataset['train'])} | Validation: {len(dataset['validation'])} | Test: {len(dataset['test'])}")

    # -------------------------------------------------------------------------
    # 3. Generation
    # -------------------------------------------------------------------------
    print("Generation...")
    result = []
    valid_json_count = 0
    invalid_json_count = 0
    gen_times = []

    # Metrics tracking
    total_generated_tokens = 0
    memory_samples = []
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_t0 = time.perf_counter()

    for patient_record in tqdm(generation_dataset):
        prompt = patient_record["messages"][:-1]
        truth = patient_record["messages"][-1]["content"]

        metadata = patient_record.get("metadata", {})
        patient_id = metadata.get("patient_id", "Unknown")
        note_date = metadata.get("note_date", "Unknown")
        note_month = metadata.get("note_month", "Unknown")
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

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=647,
            use_cache=True,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        sample_gen_time = t1 - t0
        gen_times.append(sample_gen_time)

        # sample current GPU memory allocated (GB)
        if torch.cuda.is_available():
            try:
                mem_gb = torch.cuda.memory_allocated() / 1024**3
            except Exception:
                mem_gb = 0.0
            memory_samples.append(mem_gb)
        # (no system RAM sampling)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        start_index = generated_text.find("{")
        if start_index != -1:
            prediction_json = generated_text[start_index:]
        else:
            prediction_json = ""

        try:
            json.loads(prediction_json)
            valid_json_count += 1
        except (json.JSONDecodeError, ValueError):
            invalid_json_count += 1

        result.append({
            "metadata": {"patient_id": patient_id, "note_date": note_date, "note_month": note_month},
            "original_note": prompt[-1]["content"],
            "original": truth,
            "prediction": prediction_json
        })

        try:
            out_len = outputs.shape[-1] if hasattr(outputs, 'shape') else (outputs[0].numel() if hasattr(outputs[0], 'numel') else len(outputs[0]))
        except Exception:
            out_len = 0
        try:
            if isinstance(input_ids, dict):
                first = next(iter(input_ids.values()))
            else:
                first = input_ids
            in_len = first.shape[-1]
        except Exception:
            in_len = 0
        generated_tokens = max(0, out_len - in_len)
        total_generated_tokens += generated_tokens

    # finalize wall time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_t1 = time.perf_counter()
    wall_total = wall_t1 - wall_t0

    # Compute and print only the requested inference metrics
    total_tokens = total_generated_tokens
    total_gen_time = sum(gen_times)
    latency_per_note = statistics.mean(gen_times) if gen_times else 0

    # Compute samples-based throughput and total samples
    total_samples = len(gen_times)
    inference_throughput = total_samples / total_gen_time if total_gen_time > 0 else 0

    print(f"Total Inference (all samples): {total_samples}")
    print(f"Inference Throughput (samples/s): {inference_throughput:.2f}")
    print(f"Latency per Note (avg generation time): {latency_per_note:.3f} s")
    wall_total_min = wall_total / 60.0 if wall_total is not None else None
    if wall_total_min is not None:
        print(f"Full run time (wall-clock): {wall_total:.3f} s ({wall_total_min:.3f} min)")
    else:
        print("Full run time (wall-clock): N/A")

    # VRAM stats (GB)
    peak_vram_gb = None
    avg_vram_gb = None
    if torch.cuda.is_available():
        try:
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        except Exception:
            peak_vram_gb = max(memory_samples) if memory_samples else None
        avg_vram_gb = statistics.mean(memory_samples) if memory_samples else None

    if peak_vram_gb is not None:
        print(f"VRAM usage peak (GB): {peak_vram_gb:.2f}")
    else:
        print("VRAM usage peak (GB): N/A")
    if avg_vram_gb is not None:
        print(f"VRAM usage avg (GB): {avg_vram_gb:.2f}")
    else:
        print("VRAM usage avg (GB): N/A")

    # (system RAM metrics removed)

    # -------------------------------------------------------------------------
    # 4. Save
    # -------------------------------------------------------------------------
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as json_file:
        for item in result:
            json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved as {output_path}")

    print("-" * 95)
    print(f" {mode}, Table {table_number}, Model: {model_type}")
    print("-" * 95)


if __name__ == "__main__":
    main(table_number=4, mode="no_prompt", eval_set="test", model_type="llama")
