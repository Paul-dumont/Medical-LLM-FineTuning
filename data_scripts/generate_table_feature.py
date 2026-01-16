import json
import pandas as pd
from pathlib import Path


#TO RUN:
table_number = 4


# -----------------------------------------------------------------------------
# 1. Path Configuration
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent #.resolve convert Relatif path into absolut path (./../home) into (~user/inux/home), .parent to keep the parent folder of the current file 
project_root = script_folder.parent # move up one level, to get the root project folder

json_path = project_root/"data"/"3_output_model"/f"extraction{table_number}.jsonl"
table_path =  project_root/"data"/"4_model_table"/f"extraction_features{table_number}.xlsx"

# -----------------------------------------------------------------------------
# 2. Data preprocessing 
# -----------------------------------------------------------------------------
data_row = []

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

            # Extract Patient_ID, Note_Date, Note_Month
            patient_id = truth.get("Patient_ID", "Unknown")
            note_date = truth.get("Note_Date", "Unknown")
            note_month = truth.get("Note_Month", "Unknown")

            # Line Manual
            row_manual = {
                "Patient_ID": patient_id,
                "Note_Date": note_date,
                "Note_Month": note_month,
                "Source" :  "Manual",
        }               

            row_manual.update(truth)
            data_row.append(row_manual)

            # Line Predict
            row_model = {
                "Patient_ID": patient_id,
                "Note_Date": note_date,
                "Note_Month": note_month,
                "Source" :  "Model",
            }

            row_model.update(prediction)
            data_row.append(row_model)

        except Exception as e:
            print(f"Error line: {e}") 

# -----------------------------------------------------------------------------
# 3. Save 
# -----------------------------------------------------------------------------
df = pd.DataFrame(data_row)
# Col Organisation: Patient_ID, Note_Date, Note_Month, Source, puis features  
fixed_cols = ["Patient_ID", "Note_Date", "Note_Month", "Source"]
feature_cols = sorted([c for c in df.columns if c not in fixed_cols])

df = df[fixed_cols + feature_cols]
df.to_excel(table_path, index=False)
print(f"Save file {table_path}")

