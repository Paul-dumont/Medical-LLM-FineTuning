import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# --- Configuration ---
project_root = Path(__file__).resolve().parents[1]
json_path = project_root / "data" / "3_output_model" / "extraction.jsonl"
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

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

# --- 1. Collecte des données ---
results = []
total_lines = 0

with open(json_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        total_lines += 1
        record = json.loads(line)
        try:
            orig_dict = json.loads(record["original"]).get("extraction", {})
            pred_dict = json.loads(record["prediction"]).get("extraction", {})
        except:
            continue

        s_orig = get_pairs(orig_dict)
        s_pred = get_pairs(pred_dict)

        # TP : intersection des paires (Clé + Valeur identiques)
        for k, v in s_orig & s_pred:
            results.append({'feature': k, 'type': 'tp', 'val_orig': v, 'val_pred': v})

        # FP : Dans prédiction mais pas dans original
        for k, v in s_pred - s_orig:
            results.append({'feature': k, 'type': 'fp', 'val_orig': orig_dict.get(k, ""), 'val_pred': v})

        # FN : Dans original mais pas dans prédiction
        for k, v in s_orig - s_pred:
            results.append({'feature': k, 'type': 'fn', 'val_orig': v, 'val_pred': pred_dict.get(k, "")})

df = pd.DataFrame(results)

# --- 2. Calcul Sémantique (Optimisé) ---
# On calcule la similarité uniquement pour les lignes qui ont les deux valeurs
mask = (df['val_orig'] != "") & (df['val_pred'] != "")
if not df[mask].empty:
    # On évite de recalculer 1.0 pour les TP parfaits
    tp_mask = (df['type'] == 'tp')
    df.loc[tp_mask, 'sem_score'] = 1.0
    
    # Calcul sémantique pour les autres (FP/FN qui auraient une valeur partielle)
    others_mask = mask & (df['type'] != 'tp')
    if not df[others_mask].empty:
        emb1 = semantic_model.encode(df.loc[others_mask, 'val_orig'].astype(str).tolist(), convert_to_tensor=True)
        emb2 = semantic_model.encode(df.loc[others_mask, 'val_pred'].astype(str).tolist(), convert_to_tensor=True)
        sims = util.cos_sim(emb1, emb2).diagonal().tolist()
        df.loc[others_mask, 'sem_score'] = sims
else:
    df['sem_score'] = 0.0

df['sem_score'] = df['sem_score'].fillna(0.0)

# --- 3. Synthèse par Feature ---
print(f"\n{'FEATURE':<40} | {'F1':<5} | {'SEM':<5} | {'COUNT':<5} | {'%':<5}")
print("-" * 75)

summary = []
for feat, group in df.groupby('feature'):
    tp = len(group[group['type'] == 'tp'])
    fp = len(group[group['type'] == 'fp'])
    fn = len(group[group['type'] == 'fn'])
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    count = tp + fn
    count_pct = (count / total_lines) * 100
    avg_sem = group['sem_score'].mean()
    
    summary.append({'feat': feat, 'f1': f1, 'sem': avg_sem, 'count': count, 'pct': count_pct, 'tp': tp, 'fp': fp, 'fn': fn})

# Tri par occurence (count)
summary_df = pd.DataFrame(summary).sort_values('count', ascending=False)

for _, r in summary_df.iterrows():
    print(f"{r['feat'][:40]:<40} | {r['f1']:.2f}  | {r['sem']:.2f}  | {int(r['count']):<5} | {r['pct']:.0f}%")

# --- 4. Global ---
g_tp = summary_df['tp'].sum()
g_fp = summary_df['fp'].sum()
g_fn = summary_df['fn'].sum()

g_prec = g_tp / (g_tp + g_fp) if (g_tp + g_fp) > 0 else 0
g_rec = g_tp / (g_tp + g_fn) if (g_tp + g_fn) > 0 else 0
g_f1 = 2 * (g_prec * g_rec) / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0

print("-" * 75)
print(f"{'TOTAL GLOBAL':<40} | {g_f1:.3f} | {df['sem_score'].mean():.3f} | {total_lines} lignes\n")
