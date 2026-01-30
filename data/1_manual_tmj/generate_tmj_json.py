"""
Script to convert TMJ medical notes into JSON training format
Reads from data_input/ (raw notes) and data_output_clean_46/ (structured summaries)
Generates JSONL files with messages in OpenAI chat format
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import json
from pathlib import Path
from datetime import datetime


# ============================================================================
# 1. Configuration
# ============================================================================
script_folder = Path(__file__).resolve().parent
data_input_folder = script_folder / "data_input"
data_output_folder = script_folder / "data_output_clean_46"
output_folder = script_folder / "json_output"

# Create output folder if it doesn't exist
output_folder.mkdir(exist_ok=True)

json_path_output = output_folder / "training_data_tmj.jsonl"

# ============================================================================
# 2. System Prompt
# ============================================================================
system_prompt = (
    "You are an expert TMJ (Temporomandibular Joint) Assistant. "
    "Extract the clinical findings and symptoms from the patient note as JSON."
)

# ============================================================================
# 3. Parse Summary Files
# ============================================================================
def parse_summary_file(summary_path):
    """
    Parse the summary file (key: value format) and return as dictionary
    """
    summary_data = {}
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Skip unknown values
                if value.lower() == 'unknown':
                    continue
                
                # Convert boolean strings
                if value.lower() == 'false':
                    summary_data[key] = False
                elif value.lower() == 'true':
                    summary_data[key] = True
                # Try to convert to float/int
                elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                    try:
                        if '.' in value:
                            summary_data[key] = float(value)
                        else:
                            summary_data[key] = int(value)
                    except ValueError:
                        summary_data[key] = value
                else:
                    summary_data[key] = value
    except Exception as e:
        print(f"Error parsing {summary_path}: {e}")
    
    return summary_data


# ============================================================================
# 4. Load Notes
# ============================================================================
def load_raw_notes():
    """
    Load all raw note files from data_input folder
    Returns dict: {patient_id: note_content}
    """
    notes = {}
    txt_files = sorted(data_input_folder.glob("*_Word_text.txt"))
    
    print(f"Found {len(txt_files)} note files")
    
    for txt_file in txt_files:
        try:
            # Extract patient ID from filename (e.g., B001_Word_text.txt -> B001)
            patient_id = txt_file.stem.replace("_Word_text", "")
            
            with open(txt_file, 'r', encoding='utf-8') as f:
                note_content = f.read().strip()
            
            notes[patient_id] = note_content
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
    
    return notes


def load_summaries():
    """
    Load all summary files from data_output_clean_46 folder
    Returns dict: {patient_id: summary_dict}
    """
    summaries = {}
    summary_files = sorted(data_output_folder.glob("*_summary.txt"))
    
    print(f"Found {len(summary_files)} summary files")
    
    for summary_file in summary_files:
        try:
            # Extract patient ID from filename (e.g., B001_summary.txt -> B001)
            patient_id = summary_file.stem.replace("_summary", "")
            
            summary_data = parse_summary_file(summary_file)
            summaries[patient_id] = summary_data
        except Exception as e:
            print(f"Error reading {summary_file}: {e}")
    
    return summaries


# ============================================================================
# 5. Generate Training Data
# ============================================================================
def generate_training_data(notes, summaries):
    """
    Generate training data in OpenAI chat format
    """
    training_data = []
    
    # Get patients that have both note and summary
    common_patients = sorted(set(notes.keys()) & set(summaries.keys()))
    print(f"\nPatients with both note and summary: {len(common_patients)}")
    
    for patient_id in common_patients:
        note = notes[patient_id]
        summary = summaries[patient_id]
        
        # Remove patient_id from extraction to avoid duplication
        extraction_data = {k: v for k, v in summary.items() if k != "patient_id"}
        
        # Create JSON response
        json_response = json.dumps({"extraction": extraction_data}, ensure_ascii=False)
        
        # Build metadata
        metadata = {
            "patient_id": patient_id,
            "note_date": summary.get("note_date", "unknown"),
            "created_at": datetime.now().isoformat()
        }
        
        # Chat format
        training_data.append({
            "metadata": metadata,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": note},
                {"role": "assistant", "content": json_response}
            ]
        })
    
    return training_data


# ============================================================================
# 6. Save JSONL File
# ============================================================================
def save_jsonl(training_data, output_path):
    """
    Save training data to JSONL file (one JSON per line)
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✓ Saved {len(training_data)} entries to {output_path}")


# ============================================================================
# 7. Main Execution
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("TMJ Notes to JSON Conversion Script")
    print("="*70)
    
    # Load data
    print("\n[1] Loading notes from data_input/")
    notes = load_raw_notes()
    
    print("\n[2] Loading summaries from data_output_clean_46/")
    summaries = load_summaries()
    
    # Generate training data
    print("\n[3] Generating training data...")
    training_data = generate_training_data(notes, summaries)
    
    # Save to file
    print("\n[4] Saving to JSONL file...\n")
    save_jsonl(training_data, json_path_output)
    
    print("\n" + "="*70)
    print("✓ Conversion completed successfully!")
    print(f"  Output folder: {output_folder}")
    print("="*70)
