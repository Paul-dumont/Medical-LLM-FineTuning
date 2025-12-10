import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
import pandas as pd
import json

# Load table as a Dataframme
data_file = 'data_training_jonas.xlsm'
sheets = pd.read_excel(data_file, sheet_name=None)

# Create List with patients sheets name  
patients = list(sheets.keys())[1:-2]
print("\n")
print(f"Patients found : {len(patients)} ")
print("\n")

# Merge patients sheets together
patients_sheets = []
for patient in patients:
    patients_sheets.append(sheets[patient])

sheet = pd.concat(patients_sheets, ignore_index=True)
sheet = sheet.drop(sheet.columns[:4], axis=1)
sheet = sheet.astype(object).where(pd.notnull(sheet), None)
sheet = sheet.dropna(subset=["Raw Note Text"])
print(sheet)
print("\n")

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

    # Convert Python Obeject into text (Json String) 
    json_response = json.dumps(final_output_structure, ensure_ascii=False)

    # Chat format (used by OpenAi etc ...)
    training_data.append({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(note)},
            {"role": "assistant", "content": json_response},
        ]
    })

# Save as .jsonl file
json_name =  "training_data_CoT.jsonl"
with open(json_name, "w", encoding="utf-8") as json_file:
    for item in training_data:
        json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved as {json_name}")

 
