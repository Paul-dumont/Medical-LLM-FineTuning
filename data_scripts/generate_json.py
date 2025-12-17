import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
import pandas as pd
import json
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. Path Configuration
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent #.resolve convert Relatif path into absolut path (./../home) into (~user/inux/home), .parent to keep the parent folder of the current file 
project_root = script_folder.parent # move up one level, to get the root project folder

table_path =  project_root/"data"/"1_manual_table"/"patients_table1.xlsm"
json_path = project_root/"data"/"2_input_model"/"training_data_1.jsonl"

# -----------------------------------------------------------------------------
# 2. Data Preprocessing
# -----------------------------------------------------------------------------
sheets = pd.read_excel(table_path, sheet_name=None) # Load table as a Dataframme, name=None to collect every sheets and not only one specific 
patients = list(sheets.keys())[1:-2] # Create List with patients sheets name  
print("\n")
print(f"Patients found : {len(patients)} ")
print("\n")

# Merge patients sheets together
patients_sheets = []
for patient in patients:
    patients_sheets.append(sheets[patient])

sheet = pd.concat(patients_sheets, ignore_index=True) # ignore_index = True to adjuste indix with new one and not keep previus index value, not (45 46 1 2 ) but (45 46 47 48)
sheet = sheet.drop(sheet.columns[:4], axis=1) # Delete 4 first column (we dont care), axis=1 to delete columns and not row
sheet = sheet.astype(object).where(pd.notnull(sheet), None) # convert blank by None, and convert into obj to dodge issues like mxt value ( Number, text)
sheet = sheet.dropna(subset=["Raw Note Text"]) # Delete all Row with no Notes (keep only clean table with value)
print(sheet)
print("\n")


# -----------------------------------------------------------------------------
# 3. Json Construction
# -----------------------------------------------------------------------------
# Prompt Strategy : Sparse + Synthetic CoT
system_prompt = (
    "You are an expert Orthodontist Assistant. "
    "First, analyze the clinical note in the 'thought' field, citing evidence from the text. "
    "Then, extract the positive findings into the 'extraction' field as JSON."
)

training_data = []

# Loop patients
for idx, row in sheet.iterrows(): 
    note = row["Raw Note Text"] # Extract patient note
    raw_features = row.drop("Raw Note Text").to_dict() # Extract patient features
    sparse_features= {}
    thought_list = []

    # Loop Value     
    for key, value in raw_features.items():
        if pd.notna(value): # Sparse : keep only positive values 
            if value != 1.0: # Beacause Panda transformed True into 1 in previus step 
                sparse_features[key] = value 
            else: 
                sparse_features[key] = True         
            thought_list.append(f"Found evidence for {key}.") # 1 value = 1 thought 
   
    if not thought_list: # Rare case but security (Patient has no value)
        thoughts = "No specific orthodontic procedures detected."
    else: # Common case 
        thoughts = "Analysis: " + " ".join(thought_list)

    # Ouput = thought + value 
    final_output_structure = {"thought": thoughts, "extraction": sparse_features}

    # Convert Python Object into text (Json String) 
    json_response = json.dumps(final_output_structure, ensure_ascii=False)

    # Chat format (used by OpenAi etc ...)
    training_data.append({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(note)},
            {"role": "assistant", "content": json_response},
        ]
    })

# -----------------------------------------------------------------------------
# 4. Json save
# -----------------------------------------------------------------------------
with open(json_path, "w", encoding="utf-8") as json_file:
    for item in training_data:
        json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved as {json_path}")

 
