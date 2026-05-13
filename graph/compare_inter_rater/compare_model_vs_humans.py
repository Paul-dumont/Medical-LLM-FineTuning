import json
import os
import pandas as pd
from collections import defaultdict

def load_human_annotations(filepath):
    annotations = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            patient_id = str(row['metadata']['patient_id'])
            # Normalize date to YYYY-MM-DD
            note_date = str(row['metadata']['note_date'])[:10]
            
            try:
                # Assistant's message contains the extraction
                content_str = row['messages'][1]['content']
                content = json.loads(content_str)
            except:
                content = {}
                
            extraction = content.get('extraction', {})
            normalized_ext = {k.lower().strip(): str(v).lower().strip() for k, v in extraction.items()}
            
            key = f"{patient_id}_{note_date}"
            annotations[key] = normalized_ext
    return annotations

def load_model_annotations(filepath, src_key='prediction'):
    annotations = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            patient_id = str(row['metadata']['patient_id'])
            # Normalize date to YYYY-MM-DD
            note_date = str(row['metadata']['note_date'])[:10]
            
            try:
                content = json.loads(row[src_key])
            except:
                content = {}
                
            extraction = content.get('extraction', {})
            
            # Normalize keys and values
            normalized_ext = {k.lower().strip(): str(v).lower().strip() for k, v in extraction.items()}
            
            key = f"{patient_id}_{note_date}"
            annotations[key] = normalized_ext
    return annotations

def main():
    file_h1 = "/media/luciacev/Data/Medical-LLM-FineTuning/data/2_input_model/no_prompt/training_data_no_prompt6_Human1.jsonl"
    file_h2 = "/media/luciacev/Data/Medical-LLM-FineTuning/data/2_input_model/no_prompt/training_data_no_prompt6_Human2.jsonl"
    file_model = "/media/luciacev/Data/Medical-LLM-FineTuning/data/3_output_model/no_prompt/extraction_llama_no_prompt6_10_patients.jsonl"
    
    obs1 = load_human_annotations(file_h1)
    obs2 = load_human_annotations(file_h2)
    model_preds = load_model_annotations(file_model, src_key='prediction')
    
    common_keys = set(obs1.keys()).intersection(set(obs2.keys())).intersection(set(model_preds.keys()))
    print(f"Found {len(common_keys)} matching clinical notes across Hum1, Hum2, and Model.\n")
    
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
    
    empty_vals = {"none", "0.0", "false", "0", ""}
    
    def is_empty(val):
        return val in empty_vals

    def calc_f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    results_dict = {}
    
    # Macro metrics accumulators
    mac_h1_tp = mac_h1_fp = mac_h1_fn = 0
    mac_h2_tp = mac_h2_fp = mac_h2_fn = 0
    mac_soft_tp = mac_soft_fp = mac_soft_fn = 0
    
    for feature in feature_order:
        feat_lower = feature.lower()
        
        h1_tp = h1_fp = h1_fn = 0
        h2_tp = h2_fp = h2_fn = 0
        soft_tp = soft_fp = soft_fn = 0
        support = 0
        
        for key in common_keys:
            val_h1 = obs1[key].get(feat_lower, "none")
            val_h2 = obs2[key].get(feat_lower, "none")
            val_m  = model_preds[key].get(feat_lower, "none")
            
            # Support: at least one human annotated it
            if not is_empty(val_h1) or not is_empty(val_h2):
                support += 1
                
            # Model vs H1
            if val_m == val_h1:
                if not is_empty(val_h1): h1_tp += 1
            else:
                if not is_empty(val_m): h1_fp += 1
                if not is_empty(val_h1): h1_fn += 1
                
            # Model vs H2
            if val_m == val_h2:
                if not is_empty(val_h2): h2_tp += 1
            else:
                if not is_empty(val_m): h2_fp += 1
                if not is_empty(val_h2): h2_fn += 1

            # Soft Consensus (Model is correct if it matches H1 OR H2)
            if is_empty(val_h1) and is_empty(val_h2):
                if not is_empty(val_m):
                    soft_fp += 1
            else:
                valid_human_targets = set([v for v in [val_h1, val_h2] if not is_empty(v)])
                if not is_empty(val_m) and val_m in valid_human_targets:
                    soft_tp += 1
                else:
                    soft_fn += 1
                    if not is_empty(val_m):
                        soft_fp += 1
        
        # Calculate Feature metrics
        _, _, f1_h1 = calc_f1(h1_tp, h1_fp, h1_fn)
        _, _, f1_h2 = calc_f1(h2_tp, h2_fp, h2_fn)
        p_soft, r_soft, f1_soft = calc_f1(soft_tp, soft_fp, soft_fn)
        
        results_dict[feat_lower] = {
            'Feature': feature,
            'Count': support,
            'Count %': support / len(common_keys) if len(common_keys) > 0 else 0,
            'F1 vs Hum1': f1_h1 if support > 0 else '-',
            'F1 vs Hum2': f1_h2 if support > 0 else '-',
            'F1 vs Either (Soft)': f1_soft if support > 0 else '-',
            'Soft Precision': p_soft if support > 0 else '-',
            'Soft Recall': r_soft if support > 0 else '-',
            'Soft TP': soft_tp if support > 0 else '-',
            'Soft FP': soft_fp if support > 0 else '-',
            'Soft FN': soft_fn if support > 0 else '-'
        }
        
        # Add to macro metrics
        mac_h1_tp += h1_tp; mac_h1_fp += h1_fp; mac_h1_fn += h1_fn
        mac_h2_tp += h2_tp; mac_h2_fp += h2_fp; mac_h2_fn += h2_fn
        mac_soft_tp += soft_tp; mac_soft_fp += soft_fp; mac_soft_fn += soft_fn
        
    ordered_results = [results_dict[f.lower()] for f in feature_order]
    df = pd.DataFrame(ordered_results)
    
    # Format Count % to be a percentage in pandas if desired, 
    # but storing it as a float is better for excel.
    
    print("=== Model Performance (F1) vs Humans ===")
    for _, row in df.iterrows():
        if row['Count'] == 0:
            print(f"Feature: {row['Feature'][:30]:<30} | F1 Soft: - (Count: 0)")
        else:
            print(f"Feature: {row['Feature'][:30]:<30} | Count: {row['Count']:>3} ({row['Count %']:.1%}) | F1 vs H1: {row['F1 vs Hum1']:.3f} | F1 vs H2: {row['F1 vs Hum2']:.3f} | F1 Soft: {row['F1 vs Either (Soft)']:.3f}")

    # Macro 
    _, _, gb_f1_h1 = calc_f1(mac_h1_tp, mac_h1_fp, mac_h1_fn)
    _, _, gb_f1_h2 = calc_f1(mac_h2_tp, mac_h2_fp, mac_h2_fn)
    gb_p_soft, gb_r_soft, gb_f1_soft = calc_f1(mac_soft_tp, mac_soft_fp, mac_soft_fn)
    
    print("\n=== OVERALL MODEL METRICS ===")
    print(f"Overall F1 vs Hum1 : {gb_f1_h1:.3f}")
    print(f"Overall F1 vs Hum2 : {gb_f1_h2:.3f}")
    print(f"Overall F1 Soft    : {gb_f1_soft:.3f} (Model matched H1 OR H2)")
    print(f"Soft Precision     : {gb_p_soft:.3f}")
    print(f"Soft Recall        : {gb_r_soft:.3f}")

    # Save to Excel
    out_xlsx = "/media/luciacev/Data/Medical-LLM-FineTuning/graph/compare_inter_rater/model_vs_humans_evaluation.xlsx"
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Features Metrics', index=False)
        
        overall_data = {
            'Metric': [
                'Overall F1 vs Hum1',
                'Overall F1 vs Hum2',
                'Overall F1 Soft Consensus',
                'Overall Soft Precision',
                'Overall Soft Recall',
                'Total TP (Soft)',
                'Total FP (Soft)',
                'Total FN (Soft)'
            ],
            'Value': [
                gb_f1_h1,
                gb_f1_h2,
                gb_f1_soft,
                gb_p_soft,
                gb_r_soft,
                mac_soft_tp,
                mac_soft_fp,
                mac_soft_fn
            ]
        }
        pd.DataFrame(overall_data).to_excel(writer, sheet_name='Overall Metrics', index=False)
        
    print(f"\nModel evaluation saved to {out_xlsx}")

if __name__ == '__main__':
    main()
