import json
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. Path Configuration
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent #.resolve convert Relatif path into absolut path (./../home) into (~user/inux/home), .parent to keep the parent folder of the current file 
project_root = script_folder.parent # move up one level, to get the root project folder

json_path = str(project_root / "data" / "3_output_model" / "extraction.jsonl")
table_path = str(project_root / "data" / "4_model_table" / "extraction_patients.xlsx")

# -----------------------------------------------------------------------------
# 2. Data preprocessing
# -----------------------------------------------------------------------------
data = []

with open(json_path, "r", encoding="utf-8") as json_file:
    for i, line in enumerate(json_file):
        if not line.strip(): continue
        try: 
            record = json.loads(line)
            
            # Notes
            note = record.get('original_note')
            
            # truth 
            truth_str = record.get('original')
            truth_json = json.loads(truth_str)
            truth = truth_json.get("extraction") 

            # prediction
            prediction_str = record.get('prediction')
            prediction_json = json.loads(prediction_str)
            prediction = prediction_json.get("extraction")

            thought = prediction_json.get("thought")

            data.append({
                "Raw Note": note,
                "Manual Extraction": json.dumps(truth, indent=2, ensure_ascii=False),
                "Model Extraction" : json.dumps(prediction, indent=2, ensure_ascii=False),
                "Model Reasoning" : thought
            })

        except Exception as e:
            print(f"Error line: {e}") 

# -----------------------------------------------------------------------------
# 3. Save 
# -----------------------------------------------------------------------------
df = pd.DataFrame(data)
df.to_excel(table_path, index=False)
print(f"Save file {table_path}")
