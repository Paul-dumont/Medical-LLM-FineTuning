import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
import pandas as pd
import json
from pathlib import Path
from datetime import datetime


#TO RUN:
table_number = 4


# -----------------------------------------------------------------------------
# 1. Path Configuration
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent #.resolve convert Relatif path into absolut path (./../home) into (~user/inux/home), .parent to keep the parent folder of the current file 
project_root = script_folder.parent # move up one level, to get the root project folder

table_path =  project_root/"data"/"1_manual_table"/f"patients_table{table_number}.xlsm"
json_path_with_cot = project_root/"data"/"2_input_model"/"with_cot"/f"training_data_with_cot{table_number}.jsonl"
json_path_without_cot = project_root/"data"/"2_input_model"/"without_cot"/f"training_data_without_cot{table_number}.jsonl"
json_path_dry_run = project_root/"data"/"2_input_model"/"dry_run"/f"training_data_dry_run{table_number}.jsonl"

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

# Extract and keep Patient ID, Note Date and Note Month columns before dropping
metadata_df = sheet[[sheet.columns[1], sheet.columns[2], sheet.columns[3]]].copy()

sheet = sheet.drop(sheet.columns[:4], axis=1) # Delete 4 first column (we dont care), axis=1 to delete columns and not row

# Remove "Annotated" column if it exists
if "Annotated" in sheet.columns:
    sheet = sheet.drop("Annotated", axis=1)

sheet = sheet.astype(object).where(pd.notnull(sheet), None) # convert blank by None, and convert into obj to dodge issues like mxt value ( Number, text)
sheet = sheet.dropna(subset=["Raw Note Text"]) # Delete all Row with no Notes (keep only clean table with value)

# Keep only metadata for rows that remain and reset index
metadata_df = metadata_df.loc[sheet.index].reset_index(drop=True)
sheet = sheet.reset_index(drop=True)
print(sheet)
print("\n")


# -----------------------------------------------------------------------------
# 3. Json Construction
# -----------------------------------------------------------------------------
# Prompt Strategy : Sparse + Synthetic CoT
system_prompt_with_cot = (
    "You are an expert Orthodontist Assistant. "
    "First, analyze the clinical note in the 'thought' field, citing evidence from the text. "
    "Then, extract the positive findings into the 'extraction' field as JSON."
)

system_prompt_without_cot = (
    "You are an expert Orthodontist Assistant. "
    "Extract the positive findings from the clinical note as JSON."
)

system_prompt_dry_run = (
    "You are an expert Orthodontist Assistant. Extract orthodontic findings from the clinical note as JSON.\n\n"
    "Format your response as: {\"extraction\": {...}}\n\n"
    "Potential features to extract:\n"
    "Patient_ID, Note_Date, Note_Month, Annotated, Upper Wire Size, Upper Wire Material, Lower Wire Size, "
    "Lower Wire Material, Changed Upper Arch Wire, Changed Lower Arch Wire, Ligature Method, Oral Hygiene, "
    "Elastic Pattern Left, Right Canine Class, Left Canine Class, Right Molar Class, Left Molar Class, "
    "Class II elastic, Elastic Pattern Right, Compliance, Overjet (mm), Elastic Type Left, Elastic Type Right, "
    "Overbite (mm), Debonded Bracket, Lower Retainer, Emergency Type, Upper Retainer, Space closure sliding mechanics, "
    "Photos taken, Upper Arch Bends, Class I elastic, Class III elastic, Appliance, Lower Arch Bends, Retainer Check, "
    "Xrays taken, Intra Oral Scanning Taken, EMERGENCY, Lower Arch Reverse Curve of Spee, Bracket OR BAND Repositioning, "
    "Open Spring, Upper Bonding, IPR, Re-tie Appointment, Lower Bonding, Posterior Bite Turbos, Cross Elastic, "
    "Upper Arch Accentuated Curve of Spee, Upper Debond, Lower Debond, TADs, Prescription and Bracket Slot, TMJ symptoms, "
    "Enameloplasty, Referral, Unilateral Posterior Crossbite, Extractions, TPA, Space closure loop mechanics, Upper Banding, "
    "Relapse, Upper Active movement, Lower Active movement, Closed Spring, Lower Banding, Patient ID.1, NiTi Closing Spring, "
    "Anterior Bite Turbos, TADs.1, Upper Arch Reverse Curve of Spee, Maxillary Expander, LLHA, Anterior Crossbite, "
    "Debonded Bracket/Band, Lower Arch Curve of Spee, Intrusion Arch, Bilateral Posterior Crossbite, "
    "Lower Arch Accentuated Curve of Spee, Active Traction, Active Tooth Traction, Mandibular Advancement Appliance, "
    "Teeth Pain, Arch Coordination\n\n"
    "Rules:\n"
    "- Only include features that are explicitly mentioned or clearly implied in the note\n"
    "- Use boolean (true) for binary features when found, or the actual value\n"
    "- Omit features not mentioned in the note"
)

training_data_with_cot = []
training_data_without_cot = []
training_data_dry_run = []

# Loop patients
for idx, row in sheet.iterrows(): 
    note = row["Raw Note Text"] # Extract patient note
    
    # Extract Patient ID, Note Date and Note Month from metadata
    patient_id_val = metadata_df.iloc[idx, 0]
    if pd.notna(patient_id_val):
        try:
            patient_id = str(int(patient_id_val))
        except (ValueError, TypeError):
            patient_id = str(patient_id_val)
    else:
        patient_id = "Unknown"
    
    # Keep full date, remove only the time part (00:00:00)
    note_date_str = str(metadata_df.iloc[idx, 1])
    note_date = note_date_str.replace(' 00:00:00', '') if ' 00:00:00' in note_date_str else note_date_str
    
    # Extract Note Month
    note_month_val = metadata_df.iloc[idx, 2]
    note_month = str(int(note_month_val)) if pd.notna(note_month_val) else "Unknown"
    
    # Create enhanced user message (without metadata in content)
    enhanced_note = f"{note}"
    
    raw_features = row.drop("Raw Note Text").to_dict() # Extract patient features
    sparse_features = {}
    thought_list = []

    # Loop Value (WITHOUT adding metadata as features)     
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

    # Output = thought + value 
    final_output_with_cot = {"thought": thoughts, "extraction": sparse_features}
    final_output_without_cot = {"extraction": sparse_features}

    # Convert Python Object into text (Json String) 
    json_response_with_cot = json.dumps(final_output_with_cot, ensure_ascii=False)
    json_response_without_cot = json.dumps(final_output_without_cot, ensure_ascii=False)

    # Chat format WITH CoT (used by OpenAi etc ...)
    training_data_with_cot.append({
        "metadata": {
            "patient_id": patient_id,
            "note_date": note_date,
            "note_month": note_month
        },
        "messages": [
            {"role": "system", "content": system_prompt_with_cot},
            {"role": "user", "content": enhanced_note},
            {"role": "assistant", "content": json_response_with_cot},
        ]
    })
    
    # Chat format WITHOUT CoT
    training_data_without_cot.append({
        "metadata": {
            "patient_id": patient_id,
            "note_date": note_date,
            "note_month": note_month
        },
        "messages": [
            {"role": "system", "content": system_prompt_without_cot},
            {"role": "user", "content": enhanced_note},
            {"role": "assistant", "content": json_response_without_cot},
        ]
    })
    
    # Chat format for DRY RUN (baseline with minimal prompt)
    training_data_dry_run.append({
        "metadata": {
            "patient_id": patient_id,
            "note_date": note_date,
            "note_month": note_month
        },
        "messages": [
            {"role": "system", "content": system_prompt_dry_run},
            {"role": "user", "content": enhanced_note},
            {"role": "assistant", "content": json_response_without_cot},
        ]
    })

# -----------------------------------------------------------------------------
# 4. Json save
# -----------------------------------------------------------------------------
print(f"TABLE NUMBER : {table_number}\n")
with open(json_path_with_cot, "w", encoding="utf-8") as json_file:
    for item in training_data_with_cot:
        json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved WITH CoT as {json_path_with_cot}")

with open(json_path_without_cot, "w", encoding="utf-8") as json_file:
    for item in training_data_without_cot:
        json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved WITHOUT CoT as {json_path_without_cot}")

with open(json_path_dry_run, "w", encoding="utf-8") as json_file:
    for item in training_data_dry_run:
        json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved DRY RUN (Baseline) as {json_path_dry_run}")

 
