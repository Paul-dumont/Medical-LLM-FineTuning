import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util


#TO RUN:
table_number = 4


# --- Configuration ---
project_root = Path(__file__).resolve().parents[1]
json_path = project_root / "data" / "3_output_model" / f"extraction{table_number}.jsonl"
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
        
        # NOUVEAU: FN - Valeurs vides dans prediction alors qu'elles existent dans original
        for k in orig_dict.keys():
            k_norm = str(k).lower().strip()
            v_orig_norm = str(orig_dict[k]).lower().strip() if orig_dict[k] else ""
            
            # Vérifier si clé existe dans pred_dict mais est vide
            if k in pred_dict:
                v_pred = pred_dict[k]
                v_pred_norm = str(v_pred).lower().strip() if v_pred else ""
                
                # Si original a une valeur mais prediction est vide → FN
                if v_orig_norm and not v_pred_norm:
                    results.append({'feature': k_norm, 'type': 'fn', 'val_orig': v_orig_norm, 'val_pred': ""})

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

# FN avec valeur vide (omission) = pénalité sémantique 0
fn_empty_mask = (df['type'] == 'fn') & (df['val_pred'] == "")
df.loc[fn_empty_mask, 'sem_score'] = 0.0

df['sem_score'] = df['sem_score'].fillna(0.0)

# --- 3. Synthèse par Feature ---
summary = []
for feat, group in df.groupby('feature'):
    tp = len(group[group['type'] == 'tp'])
    fp = len(group[group['type'] == 'fp'])
    fn = len(group[group['type'] == 'fn'])
    
    # Cas manquants = total_lines - (tp + fp + fn)
    # Ces cas manquants doivent être comptés comme FN
    missing_cases = total_lines - (tp + fp + fn)
    fn_total = fn + missing_cases  # Total FN incluant cas manquants
    
    # Précision et Recall par rapport au total global (pas juste les cas présents)
    prec = tp / (tp + fp + missing_cases) if (tp + fp + missing_cases) > 0 else 0
    rec = tp / (tp + fn_total) if (tp + fn_total) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    count = tp + fn
    count_pct = (count / total_lines) * 100
    avg_sem = group['sem_score'].mean()
    
    # Pénaliser le score sémantique avec les cas manquants
    # Moyenne sémantique = (scores existants + 0 pour cas manquants) / total_lines
    sem_with_missing = (group['sem_score'].sum() + (missing_cases * 0.0)) / total_lines
    
    summary.append({'feat': feat, 'prec': prec, 'rec': rec, 'f1': f1, 'sem': sem_with_missing, 'count': count, 'pct': count_pct, 'tp': tp, 'fp': fp, 'fn': fn})

# Tri par occurence (count)
summary_df = pd.DataFrame(summary).sort_values('count', ascending=False)

# Ordre personnalisé des features
feature_order = [
    "Patient_ID", "Note_Date", "Note_Month", "Annotated",
    "Upper Wire Size", "Upper Wire Material", "Lower Wire Size", "Lower Wire Material",
    "Changed Upper Arch Wire", "Changed Lower Arch Wire", "Ligature Method", "Oral Hygiene",
    "Elastic Pattern Left", "Right Canine Class", "Left Canine Class", "Right Molar Class",
    "Left Molar Class", "Class II elastic", "Elastic Pattern Right", "Compliance",
    "Overjet (mm)", "Elastic Type Left", "Elastic Type Right", "Overbite (mm)",
    "Debonded Bracket", "Lower Retainer", "Emergency Type", "Upper Retainer",
    "Space closure sliding mechanics", "Photos taken", "Upper Arch Bends", "Class I elastic",
    "Class III elastic", "Appliance", "Lower Arch Bends", "Retainer Check",
    "Xrays taken", "Intra Oral Scanning Taken", "EMERGENCY", "Lower Arch Reverse Curve of Spee",
    "Bracket OR BAND Repositioning", "Open Spring", "Upper Bonding", "IPR",
    "Re-tie Appointment", "Lower Bonding", "Posterior Bite Turbos", "Cross Elastic",
    "Upper Arch Accentuated Curve of Spee", "Upper Debond", "Lower Debond", "TADs",
    "Prescription and Bracket Slot", "TMJ symptoms", "Enameloplasty", "Referral",
    "Unilateral Posterior Crossbite", "Extractions", "TPA", "Space closure loop mechanics",
    "Upper Banding", "Relapse", "Upper Active movement", "Lower Active movement",
    "Closed Spring", "Lower Banding", "Patient ID.1", "NiTi Closing Spring",
    "Anterior Bite Turbos", "TADs.1", "Upper Arch Reverse Curve of Spee", "Maxillary Expander",
    "LLHA", "Anterior Crossbite", "Debonded Bracket/Band", "Lower Arch Curve of Spee",
    "Intrusion Arch", "Bilateral Posterior Crossbite", "Lower Arch Accentuated Curve of Spee",
    "Active Traction", "Active Tooth Traction", "Mandibular Advancement Appliance",
    "Teeth Pain", "Arch Coordination"
]

# Créer un dictionnaire pour mapper chaque feature (lowercase) à son index
feature_order_lower = [f.lower() for f in feature_order]
feature_index_map = {f.lower(): idx for idx, f in enumerate(feature_order)}

# Ajouter les features non listées à la fin (dans l'ordre qu'elles apparaissent)
all_features = summary_df['feat'].tolist()
for feat in all_features:
    if feat.lower() not in feature_index_map:
        feature_index_map[feat.lower()] = len(feature_index_map)

# Trier summary_df selon l'ordre personnalisé
summary_df['sort_index'] = summary_df['feat'].str.lower().map(feature_index_map)
summary_df = summary_df.sort_values('sort_index').drop('sort_index', axis=1)

# Garder SEULEMENT les features dans feature_order
summary_df_filtered = summary_df[summary_df['feat'].str.lower().isin(feature_order_lower)]

# Créer un dictionnaire pour accès rapide aux features
feat_dict = {}
for idx, row in summary_df_filtered.iterrows():
    feat_dict[row['feat'].lower()] = row

# Compter les features détectées (présentes dans feat_dict)
features_detected = len(feat_dict)

# Afficher le titre et les en-têtes du tableau
print(f"\nTotal features: {features_detected} / 84\n")
print(f"{'FEATURE':<40} | {'PREC':<5} | {'REC':<5} | {'F1':<5} | {'SEM':<5} | {'COUNT':<5} | {'%':<5}")
print("-" * 95)

# Afficher toutes les features de feature_order, même celles manquantes
for order_feat in feature_order:
    order_feat_lower = order_feat.lower()
    if order_feat_lower in feat_dict:
        r = feat_dict[order_feat_lower]
        print(f"{r['feat'][:40]:<40} | {r['prec']:.2f}  | {r['rec']:.2f}  | {r['f1']:.2f}  | {r['sem']:.2f}  | {int(r['count']):<5} | {r['pct']:.0f}%")
    else:
        # Feature manquante : afficher avec tirets
        print(f"{order_feat[:40]:<40} | {'-':<5} | {'-':<5} | {'-':<5} | {'-':<5} | {'-':<5} | {'-':<5}")

# --- 4. Global ---
g_tp = summary_df_filtered['tp'].sum()
g_fp = summary_df_filtered['fp'].sum()
g_fn = summary_df_filtered['fn'].sum()

# Cas manquants au niveau global
# = nombre total de paires attendues - nombre de paires correctes
# Paires attendues = total_lines * nombre de features en base
expected_pairs = total_lines * len(feature_order_lower)
actual_pairs = g_tp + g_fn + g_fp
missing_pairs = expected_pairs - actual_pairs
g_fn_total = g_fn + missing_pairs

g_prec = g_tp / (g_tp + g_fp + missing_pairs) if (g_tp + g_fp + missing_pairs) > 0 else 0
g_rec = g_tp / (g_tp + g_fn_total) if (g_tp + g_fn_total) > 0 else 0
g_f1 = 2 * (g_prec * g_rec) / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0

# Score sémantique global avec pénalité des cas manquants
actual_sem_sum = df['sem_score'].sum()
g_sem = actual_sem_sum / expected_pairs if expected_pairs > 0 else 0

print("-" * 95)

# --- 5. Top 10, Top 20, Top 30 Features ---
for top_n in [10, 20, 30]:
    top_features = summary_df_filtered.nlargest(top_n, 'pct')
    mean_prec = top_features['prec'].mean()
    mean_rec = top_features['rec'].mean()
    mean_f1 = top_features['f1'].mean()
    mean_sem = top_features['sem'].mean()
    count_total = top_features['count'].sum()
    
    print(f"{'TOP ' + str(top_n):<40} | {mean_prec:.2f}  | {mean_rec:.2f}  | {mean_f1:.3f} | {mean_sem:.3f}")

print(f"{'TOTAL GLOBAL':<40} | {g_prec:.2f}  | {g_rec:.2f}  | {g_f1:.3f} | {g_sem:.3f}")

print()

# Comparison with base feature list (normalize to lowercase for comparison)
feature_order_lower = [f.lower() for f in feature_order]
feature_order_set = set(feature_order_lower)
output_features_set = set([f.lower() for f in summary_df['feat'].tolist()])

# Features in output but not in base list
extra_features = output_features_set - feature_order_set
if extra_features:
    print(f"⚠️  EXTRA Features ({len(extra_features)} - model hallucinations - not in base list):")
    for feat in sorted(extra_features):
        pass
    # print(f"  - {feat}")
        