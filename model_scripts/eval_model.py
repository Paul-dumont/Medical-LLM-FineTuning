import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns


#TO RUN:
table_number = 1
mode = "tmj"

print("-" * 95)
print(f" {mode}, Table {table_number}")
print("-" * 95)

# --- Configuration ---
project_root = Path(__file__).resolve().parents[1]
json_path = project_root / "data" / "3_output_model" / f"{mode}" / f"extraction_{mode}{table_number}.jsonl"
results_dir = project_root / "data" / "5_results" / f"{mode}"
results_dir.mkdir(parents=True, exist_ok=True)

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

# Ordre personnalisé des features selon le mode
if mode == "tmj":
    feature_order = [
        "patient_id", "patient_age", "headache_intensity", "tmj_pain_rating",
        "disc_displacement", "joint_arthritis_location", "jaw_function_score", "maximum_opening",
        "diet_score", "disability_rating", "tinnitus_present", "vertigo_present",
        "joint_pain_areas", "earache_present", "pain_aggravating_factors", "average_daily_pain_intensity",
        "airway_obstruction_present", "pain_onset_date", "appliance_history", "current_medications",
        "headache_location", "muscle_pain_location", "muscle_symptoms_present", "muscle_pain_score",
        "hearing_loss_present", "jaw_clicking", "headache_frequency", "sleep_disorder_type",
        "maximum_opening_without_pain", "neck_pain_present", "current_appliance", "onset_triggers",
        "physical_therapy_status", "adverse_reactions", "jaw_crepitus", "jaw_locking",
        "pain_relieving_factors", "back_pain_present", "sleep_apnea_diagnosed", "autoimmune_condition",
        "migraine_history", "previous_medications", "pain_frequency", "depression_present",
        "pain_duration", "fibromyalgia_present", "migraine_frequency", "bipap_used",
        "apap_used", "cpap_used", "hearing_sensitivity_present", "anxiety_present",
        "muscle_spasm_present", "muscle_stiffness_present", "muscle_soreness_present"
    ]
    total_features = 55
else:
    # Orthodontie (défaut)
    feature_order = [
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
    total_features = 84

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
print(f"\nTotal features: {features_detected} / {total_features}\n")
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
# Moyenne simple des scores de toutes les features (comme TOP 10/20/30 et SEM)
g_prec = summary_df_filtered['prec'].mean() if len(summary_df_filtered) > 0 else 0
g_rec = summary_df_filtered['rec'].mean() if len(summary_df_filtered) > 0 else 0
g_f1 = summary_df_filtered['f1'].mean() if len(summary_df_filtered) > 0 else 0

# Score sémantique global : moyenne des SEM scores des features qui existent (les "-" ne sont pas comptées)
g_sem = summary_df_filtered['sem'].mean() if len(summary_df_filtered) > 0 else 0

print("-" * 95)

# --- 5. TOTAL SIMPLE et TOTAL PONDÉRÉ ---
# TOTAL SIMPLE : moyenne simple (toutes les features ont le même poids)
print(f"{'TOTAL SIMPLE':<40} | {g_prec:.2f}  | {g_rec:.2f}  | {g_f1:.3f} | {g_sem:.3f}")

# TOTAL PONDÉRÉ : moyenne pondérée par fréquence (count)
total_count = summary_df_filtered['count'].sum()
g_prec_weighted = (summary_df_filtered['prec'] * summary_df_filtered['count']).sum() / total_count if total_count > 0 else 0
g_rec_weighted = (summary_df_filtered['rec'] * summary_df_filtered['count']).sum() / total_count if total_count > 0 else 0
g_f1_weighted = (summary_df_filtered['f1'] * summary_df_filtered['count']).sum() / total_count if total_count > 0 else 0
g_sem_weighted = (summary_df_filtered['sem'] * summary_df_filtered['count']).sum() / total_count if total_count > 0 else 0
print(f"{'TOTAL PONDÉRÉ':<40} | {g_prec_weighted:.2f}  | {g_rec_weighted:.2f}  | {g_f1_weighted:.3f} | {g_sem_weighted:.3f}")

print("-" * 95)

# --- 6. SAVE RESULTS ---
print("\n💾 Saving results...")

# Create base filename
base_filename = f"result_{mode}_{table_number}"

# Save XLSX (Excel format)
xlsx_file = results_dir / f"{base_filename}.xlsx"
with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
    # Sheet 1: Features
    summary_df_filtered.to_excel(writer, sheet_name='Features', index=False)
    
    # Sheet 2: Global Metrics
    metrics_data = {
        'Metric': ['Precision', 'Recall', 'F1 Score', 'Semantic Similarity'],
        'Simple Average': [g_prec, g_rec, g_f1, g_sem],
        'Weighted Average': [g_prec_weighted, g_rec_weighted, g_f1_weighted, g_sem_weighted]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_excel(writer, sheet_name='Global Metrics', index=False)
    
    # Sheet 3: Summary Info
    summary_info = {
        'Parameter': ['Mode', 'Table Number', 'Total Samples', 'Features Detected', 'Features in Base List'],
        'Value': [mode, table_number, total_lines, features_detected, total_features]
    }
    summary_info_df = pd.DataFrame(summary_info)
    summary_info_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # Format the workbook
    workbook = writer.book
    for sheet_name in workbook.sheetnames:
        worksheet = writer.sheets[sheet_name]
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

print(f"  ✓ XLSX: {xlsx_file}")
print(f"\n✅ Results saved to: {results_dir}\n")

print("-" * 95)
print(f" {mode}, Table {table_number}")
print("-" * 95)