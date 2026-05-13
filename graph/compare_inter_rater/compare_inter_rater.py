import json
import os
import pandas as pd
from collections import defaultdict
try:
    from sklearn.metrics import cohen_kappa_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

def load_annotations(filepath):
    annotations = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            patient_id = row['metadata']['patient_id']
            note_date = row['metadata']['note_date']
            
            # The actual labels are stored as a JSON string in 'original' or 'prediction'
            content = json.loads(row['original'])
            extraction = content.get('extraction', {})
            
            # Normalize keys to lowercase for robust matching
            normalized_ext = {k.lower().strip(): str(v).lower().strip() for k, v in extraction.items()}
            
            key = f"{patient_id}_{note_date}"
            annotations[key] = normalized_ext
    return annotations

def main():
    file1 = "/media/luciacev/Data/Medical-LLM-FineTuning/data/3_output_model/no_prompt/extraction_llama_no_prompt6_Human1.jsonl"
    file2 = "/media/luciacev/Data/Medical-LLM-FineTuning/data/3_output_model/no_prompt/extraction_llama_no_prompt6_Human2.jsonl"
    
    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"Error: One or both input files not found.")
        return
        
    obs1 = load_annotations(file1)
    obs2 = load_annotations(file2)
    
    common_keys = set(obs1.keys()).intersection(set(obs2.keys()))
    print(f"Found {len(common_keys)} matching clinical notes between both observers.\n")
    
    # Collect all unique features
    all_features = set()
    for key in common_keys:
        all_features.update(obs1[key].keys())
        all_features.update(obs2[key].keys())
        
    results = []
    
    # Store global values for macro-metrics
    global_y1 = []
    global_y2 = []
    
    for feature in sorted(all_features):
        y1 = []
        y2 = []
        
        for key in common_keys:
            val1 = obs1[key].get(feature, "none")
            val2 = obs2[key].get(feature, "none")
            y1.append(val1)
            y2.append(val2)
            
            global_y1.append(val1)
            global_y2.append(val2)
            
        # Calculate agreement and F1
        agreements = 0
        tp = fp = fn = 0
        
        empty_vals = {"none", "0.0", "false", "0", ""}
        
        for val1, val2 in zip(y1, y2):
            is_empty1 = val1 in empty_vals
            is_empty2 = val2 in empty_vals
            
            if val1 == val2:
                agreements += 1
                if not is_empty1:
                    tp += 1
            else:
                if not is_empty1:
                    fn += 1
                if not is_empty2:
                    fp += 1
        
        total = len(y1)
        acc = agreements / total
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        kappa = None
        if HAS_SKLEARN:
            # Only compute kappa if there's variation in both sets
            kappa = cohen_kappa_score(y1, y2)
            
        results.append({
            'Feature': feature,
            'Total Notes': total,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Agreement %': acc * 100,
            'Cohen Kappa': kappa,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
    df = pd.DataFrame(results)
    
    print("=== Inter-Annotator Agreement by Feature ===")
    
    # Formatting for console output
    for _, row in df.iterrows():
        k_str = f"{row['Cohen Kappa']:.3f}" if pd.notnull(row['Cohen Kappa']) else "N/A"
        print(f"Feature: {row['Feature'][:32]:<32} | Agree: {row['Agreement %']:>5.1f}% | Kap: {k_str:>5} | F1: {row['F1-Score']:.3f} (P: {row['Precision']:.2f}, R: {row['Recall']:.2f})")
        
    # Global metrics
    global_agreements = 0
    global_tp = global_fp = global_fn = 0
    
    empty_vals = {"none", "0.0", "false", "0", ""}
    
    for val1, val2 in zip(global_y1, global_y2):
        is_empty1 = val1 in empty_vals
        is_empty2 = val2 in empty_vals
        
        if val1 == val2:
            global_agreements += 1
            if not is_empty1:
                global_tp += 1
        else:
            if not is_empty1:
                global_fn += 1
            if not is_empty2:
                global_fp += 1
                
    global_acc = global_agreements / len(global_y1)
    g_prec = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    g_rec  = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    g_f1   = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0.0
    
    global_kappa = cohen_kappa_score(global_y1, global_y2) if HAS_SKLEARN else None

    print("\n=== Overall Metrics ===")
    print(f"Total Feature Comparisons: {len(global_y1)}")
    print(f"Overall Agreement %:       {global_acc * 100:.2f}%")
    if HAS_SKLEARN:
        print(f"Overall Cohen's Kappa:     {global_kappa:.3f}")
    print(f"Overall Precision / Recall:{g_prec:.3f} / {g_rec:.3f}")
    print(f"Overall F1-Score:          {g_f1:.3f}")
        
    # Save results
    out_xlsx = "inter_annotator_agreement.xlsx"
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Features Metrics', index=False)
        
        overall_data = {
            'Metric': [
                'Total Feature Comparisons', 
                'Overall Agreement %', 
                "Overall Cohen's Kappa", 
                'Overall Precision', 
                'Overall Recall', 
                'Overall F1-Score'
            ],
            'Value': [
                len(global_y1),
                global_acc * 100,
                global_kappa if HAS_SKLEARN else None,
                g_prec,
                g_rec,
                g_f1
            ]
        }
        df_overall = pd.DataFrame(overall_data)
        df_overall.to_excel(writer, sheet_name='Overall Metrics', index=False)
        
    print(f"\nDetailed metrics saved to {out_xlsx}")

if __name__ == '__main__':
    main()
