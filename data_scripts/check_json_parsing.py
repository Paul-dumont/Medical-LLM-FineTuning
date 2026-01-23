import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import re

# TO RUN:
table_number = 4
json_path = Path("/media/luciacev/Data/Medical-LLM-FineTuning/data/3_output_model/dry_run/dry_run_baseline_table4.jsonl")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

print("=" * 80)
print("JSON PARSING + F1 + SEMANTIC - DRY RUN BASELINE")
print("=" * 80)

total_lines = 0
json_errors = 0
missing_keys = 0
valid_pairs = 0
results = []

print(f"\nAnalyzing: {json_path}\n")

def get_pairs(d):
    """Recrée exactement votre logique de set de paires (clé, valeur_norm)"""
    if not d: return set()
    pairs = set()
    for k, v in d.items():
        k_norm = str(k).lower().strip()
        if isinstance(v, list):
            v_norm = str(sorted([str(x).lower().strip() for x in v]))
        else:
            v_norm = str(v).lower().strip()
        pairs.add((k_norm, v_norm))
    return pairs

with open(json_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        total_lines += 1
        
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            json_errors += 1
            continue
        
        # Vérifier les clés requises
        if "original" not in record or "prediction" not in record:
            missing_keys += 1
            continue
        
        # Essayer de parser original
        try:
            orig_dict = json.loads(record["original"]).get("extraction", {})
            orig_valid = True
        except:
            orig_valid = False
        
        # Essayer de parser prediction (extraire le JSON du texte)
        pred_text = record["prediction"]
        pred_valid = False
        pred_dict = {}
        
        # Chercher tous les { et prendre le dernier JSON valide trouvé
        json_blocks = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', pred_text)
        
        for json_candidate in reversed(json_blocks):
            try:
                parsed = json.loads(json_candidate)
                pred_dict = parsed.get("extraction", {})
                pred_valid = True
                break
            except:
                continue
        
        # Si both JSON sont valides, c'est une paire valide
        if orig_valid and pred_valid:
            valid_pairs += 1
            
            # Calculer TP, FP, FN
            s_orig = get_pairs(orig_dict)
            s_pred = get_pairs(pred_dict)
            
            # TP : intersection des paires
            for k, v in s_orig & s_pred:
                results.append({'feature': k, 'type': 'tp', 'val_orig': v, 'val_pred': v})
            
            # FP : Dans prédiction mais pas dans original
            for k, v in s_pred - s_orig:
                results.append({'feature': k, 'type': 'fp', 'val_orig': orig_dict.get(k, ""), 'val_pred': v})
            
            # FN : Dans original mais pas dans prédiction
            for k, v in s_orig - s_pred:
                results.append({'feature': k, 'type': 'fn', 'val_orig': v, 'val_pred': pred_dict.get(k, "")})
            
            # FN - Valeurs vides dans prediction
            for k in orig_dict.keys():
                k_norm = str(k).lower().strip()
                v_orig_norm = str(orig_dict[k]).lower().strip() if orig_dict[k] else ""
                
                if k in pred_dict:
                    v_pred = pred_dict[k]
                    v_pred_norm = str(v_pred).lower().strip() if v_pred else ""
                    
                    if v_orig_norm and not v_pred_norm:
                        results.append({'feature': k_norm, 'type': 'fn', 'val_orig': v_orig_norm, 'val_pred': ""})
        else:
            if not orig_valid or not pred_valid:
                json_errors += 1

# Calculs des taux de parsage
error_rate = (json_errors / total_lines * 100) if total_lines > 0 else 0
valid_rate = (valid_pairs / total_lines * 100) if total_lines > 0 else 0

print(f"Total lines: {total_lines}")
print(f"Valid JSON pairs: {valid_pairs}")
print(f"JSON parsing errors: {json_errors}")

print("\n" + "=" * 80)
print(f"✓ Valid pair rate: {valid_rate:.1f}%")
print(f"✗ JSON error rate: {error_rate:.1f}%")
print("=" * 80)

# --- Calcul F1 et Semantic si on a des résultats ---
if results:
    df = pd.DataFrame(results)
    
    # Calcul sémantique
    mask = (df['val_orig'] != "") & (df['val_pred'] != "")
    if not df[mask].empty:
        tp_mask = (df['type'] == 'tp')
        df.loc[tp_mask, 'sem_score'] = 1.0
        
        others_mask = mask & (df['type'] != 'tp')
        if not df[others_mask].empty:
            emb1 = semantic_model.encode(df.loc[others_mask, 'val_orig'].astype(str).tolist(), convert_to_tensor=True)
            emb2 = semantic_model.encode(df.loc[others_mask, 'val_pred'].astype(str).tolist(), convert_to_tensor=True)
            sims = util.cos_sim(emb1, emb2).diagonal().tolist()
            df.loc[others_mask, 'sem_score'] = sims
    else:
        df['sem_score'] = 0.0
    
    fn_empty_mask = (df['type'] == 'fn') & (df['val_pred'] == "")
    df.loc[fn_empty_mask, 'sem_score'] = 0.0
    df['sem_score'] = df['sem_score'].fillna(0.0)
    
    # Synthèse globale
    summary = []
    for feat, group in df.groupby('feature'):
        tp = len(group[group['type'] == 'tp'])
        fp = len(group[group['type'] == 'fp'])
        fn = len(group[group['type'] == 'fn'])
        
        missing_cases = total_lines - (tp + fp + fn)
        fn_total = fn + missing_cases
        
        prec = tp / (tp + fp + missing_cases) if (tp + fp + missing_cases) > 0 else 0
        rec = tp / (tp + fn_total) if (tp + fn_total) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        avg_sem = group['sem_score'].mean()
        sem_with_missing = (group['sem_score'].sum() + (missing_cases * 0.0)) / total_lines
        
        summary.append({'feat': feat, 'prec': prec, 'rec': rec, 'f1': f1, 'sem': sem_with_missing})
    
    summary_df = pd.DataFrame(summary)
    
    # Statistiques globales
    g_prec = summary_df['prec'].mean()
    g_rec = summary_df['rec'].mean()
    g_f1 = summary_df['f1'].mean()
    g_sem = summary_df['sem'].mean()
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"  Precision: {g_prec:.3f}")
    print(f"  Recall:    {g_rec:.3f}")
    print(f"  F1 Score:  {g_f1:.3f}")
    print(f"  Semantic:  {g_sem:.3f}")
    
else:
    print("\n⚠️ No valid pairs to analyze")
