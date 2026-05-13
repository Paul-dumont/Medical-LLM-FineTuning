import json
import os
import pandas as pd
from collections import defaultdict

def load_human_annotations(filepath):
    annotations = defaultdict(dict)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            patient_id = str(row['metadata']['patient_id'])
            note_date = str(row['metadata']['note_date'])[:10]
            
            try:
                content_str = row['messages'][1]['content']
                content = json.loads(content_str)
            except:
                content = {}
                
            extraction = content.get('extraction', {})
            normalized_ext = {k.lower().strip(): str(v).lower().strip() for k, v in extraction.items()}
            
            annotations[patient_id][note_date] = normalized_ext
    return annotations

def load_model_annotations(filepath, src_key='prediction'):
    annotations = defaultdict(dict)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            patient_id = str(row['metadata']['patient_id'])
            note_date = str(row['metadata']['note_date'])[:10]
            
            try:
                content = json.loads(row[src_key])
            except:
                content = {}
                
            extraction = content.get('extraction', {})
            normalized_ext = {k.lower().strip(): str(v).lower().strip() for k, v in extraction.items()}
            
            annotations[patient_id][note_date] = normalized_ext
    return annotations

def main():
    file_h1 = "/media/luciacev/Data/Medical-LLM-FineTuning/data/2_input_model/no_prompt/training_data_no_prompt6_Human1.jsonl"
    file_h2 = "/media/luciacev/Data/Medical-LLM-FineTuning/data/2_input_model/no_prompt/training_data_no_prompt6_Human2.jsonl"
    file_model = "/media/luciacev/Data/Medical-LLM-FineTuning/data/3_output_model/no_prompt/extraction_llama_no_prompt6_10_patients.jsonl"
    
    obs1 = load_human_annotations(file_h1)
    obs2 = load_human_annotations(file_h2)
    model_preds = load_model_annotations(file_model, src_key='prediction')
    
    feature_order = [
        "Upper Wire Size", "Upper Wire Material", "Lower Wire Size", "Lower Wire Material",
        "Changed Upper Arch Wire", "Changed Lower Arch Wire", "Oral Hygiene", "Ligature Method",
        "Right Molar Class", "Left Molar Class", "Left Canine Class", "Right Canine Class",
        "Elastic Pattern Left", "Elastic Pattern Right", "Compliance", "Elastic Type Left",
        "Elastic Type Right", "Overjet (mm)", "Overbite (mm)", "Recods Taken (x-rays; IOS; Photos, Facial scanning)",
        "Upper Arch Bends", "Lower Arch Bends", "Debonded Bracket", "Retainer Check",
        "Re-tie Appointment", "Appliance", "Emergency Type", "Prescription and Bracket Slot",
        "Bracket OR BAND Repositioning", "Retainer", "COS lower", "EMERGENCY",
        "Upper Bonding", "IPR", "Lower Bonding", "Open Spring",
        "Active Space Closure", "COS upper", "Posterior Bite Turbos", "Upper Debond",
        "Lower Debond", "Enameloplasty", "TADs", "Relapse",
        "TMJ symptoms", "Upper Banding", "TPA", "Referral",
        "Extractions", "Posterior Crossbite", "Lower Banding", "Closed Spring",
        "Anterior Bite Turbos", "Maxillary Expander", "Anterior Crossbite", "LLHA",
        "Mandibular Advancement Appliance", "NANCE", "Cantilever", "Tongue Crib",
        "FaceMask"
    ]
    
    empty_vals = {"none", "0.0", "false", "0", "", "-", "nan"}
    
    # Get all patients
    all_patients = set(obs1.keys()) | set(obs2.keys()) | set(model_preds.keys())
    
    out_xlsx = "/media/luciacev/Data/Medical-LLM-FineTuning/graph/compare_inter_rater/patient_by_patient_comparison.xlsx"
    
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        for patient in sorted(all_patients):
            # Get all dates for this patient
            dates = set(obs1.get(patient, {}).keys()) | set(obs2.get(patient, {}).keys()) | set(model_preds.get(patient, {}).keys())
            
            patient_data = []
            
            for date in sorted(dates):
                h1_date = obs1.get(patient, {}).get(date, {})
                h2_date = obs2.get(patient, {}).get(date, {})
                m_date = model_preds.get(patient, {}).get(date, {})
                
                for feature in feature_order:
                    f_lower = feature.lower()
                    v_h1 = h1_date.get(f_lower, "-")
                    v_h2 = h2_date.get(f_lower, "-")
                    v_m = m_date.get(f_lower, "-")
                    
                    # Normalize empty-looking things to "-"
                    if v_h1 in empty_vals: v_h1 = "-"
                    if v_h2 in empty_vals: v_h2 = "-"
                    if v_m in empty_vals: v_m = "-"
                    
                    # Only add row if at least one of them has a non-empty value
                    # This prevents 61 rows per date where most are empty
                    if v_h1 != "-" or v_h2 != "-" or v_m != "-":
                        patient_data.append({
                            "Date": date,
                            "Feature": feature,
                            "Human 1": v_h1,
                            "Human 2": v_h2,
                            "Model": v_m,
                            "Match (H1 or H2)": "Yes" if (v_m == v_h1 and v_h1 != "-") or (v_m == v_h2 and v_h2 != "-") else "No" if v_m != "-" else "-"
                        })
            
            if patient_data:
                df = pd.DataFrame(patient_data)
                # Keep patient sheet name max 31 chars
                sheet_name = str(patient)[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
    print(f"Detailed visualizations generated: {out_xlsx}")

if __name__ == '__main__':
    main()