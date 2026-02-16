import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns


def main(table_number: int, mode: str):
    """
    Main function for evaluating model predictions.
    
    Args:
        table_number: Table number (1-4)
        mode: Training mode (no_prompt, with_cot, without_cot, tmj, dry_run)
    """
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
            
            # Ignorer les valeurs "unknown" - elles ne sont pas pertinentes pour l'évaluation
            if v_norm != "unknown":
                pairs.add((k_norm, v_norm))
        return pairs

    # --- 1. Collecte des données ---
    results = []
    total_lines = 0
    record_data = []  # Store original/prediction dicts for semantic scoring
    skipped = 0

    with open(json_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            total_lines += 1
            try:
                record = json.loads(line)
            except Exception as e:
                print(f"⚠️  Erreur parsing line {i+1}: {e}")
                skipped += 1
                continue
            
            try:
                orig_dict = json.loads(record["original"]).get("extraction", {})
                # Handle prediction as either dict or JSON string
                if isinstance(record["prediction"], dict):
                    # Pour tmj, la structure est nested avec "extraction" à l'intérieur
                    if mode == "tmj":
                        pred_dict = record["prediction"].get("extraction", {})
                    else:
                        pred_dict = record["prediction"]
                else:
                    pred_dict = json.loads(record["prediction"]).get("extraction", {})
            except Exception as e:
                print(f"⚠️  Erreur extraction line {i+1}: {e}")
                skipped += 1
                continue

            record_data.append((orig_dict, pred_dict))
            s_orig = get_pairs(orig_dict)
            s_pred = get_pairs(pred_dict)

            # TP : intersection des paires (Clé + Valeur identiques)
            for k, v in s_orig & s_pred:
                results.append({'feature': k, 'type': 'tp', 'val_orig': v, 'val_pred': v, 'key': k})

            # FP : Dans prédiction mais pas dans original
            for k, v in s_pred - s_orig:
                results.append({'feature': k, 'type': 'fp', 'val_orig': orig_dict.get(k, ""), 'val_pred': v, 'key': k})

            # FN : Dans original mais pas dans prédiction
            for k, v in s_orig - s_pred:
                results.append({'feature': k, 'type': 'fn', 'val_orig': v, 'val_pred': pred_dict.get(k, ""), 'key': k})
            
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
                        results.append({'feature': k_norm, 'type': 'fn', 'val_orig': v_orig_norm, 'val_pred': "", 'key': k_norm})

    df = pd.DataFrame(results)

    # Vérification
    if df.empty:
        print(f"❌ ERREUR: Aucune donnée à traiter!")
        print(f"   Total lignes lues: {total_lines}")
        print(f"   Lignes ignorées: {skipped}")
        print(f"   Résultats: {len(results)}")
        exit(1)

    # --- 2. Calcul Sémantique (AMÉLIORÉ v2) ---
    # NOUVELLE APPROCHE: Pour chaque feature, comparer les valeurs originals vs predictions
    # même si elles ne matchent pas exactement (formats différents)
    # Cela capture: "0.019\"x0.025\"" vs "19 x 25" → similarité sémantique!

    df['sem_score'] = 0.0

    # TP parfait (match exact) = 1.0
    tp_mask = (df['type'] == 'tp')
    df.loc[tp_mask, 'sem_score'] = 1.0

    # FP/FN: Comparer sémantiquement les valeurs de la MÊME feature
    for feature in df['feature'].unique():
        feature_mask = (df['feature'] == feature)
        feature_data = df[feature_mask].copy()
        
        # Récupérer les valeurs originales et prédites pour cette feature
        origins = feature_data[feature_data['val_orig'].astype(str).str.strip() != '']
        predictions = feature_data[feature_data['val_pred'].astype(str).str.strip() != '']
        
        # CAS 1 : PLUSIEURS occurrences (mismatch de format avec patterns)
        # CAS 2 : UNE SEULE occurrence (single-occurrence format mismatch)
        if len(origins) > 0 and len(predictions) > 0:
            # Obtenir les indices pour cette feature
            indices_orig = origins.index.tolist()
            indices_pred = predictions.index.tolist()
            
            # Comparer les valeurs
            for idx_o in indices_orig:
                for idx_p in indices_pred:
                    val_o = str(df.loc[idx_o, 'val_orig']).strip()
                    val_p = str(df.loc[idx_p, 'val_pred']).strip()
                    
                    # Si pas un TP parfait ET les deux valeurs existent
                    if val_o and val_p and not ((df.loc[idx_o, 'type'] == 'tp') or (df.loc[idx_p, 'type'] == 'tp')):
                        # Calculer similitude sémantique (même pour single-occurrence)
                        emb1 = semantic_model.encode(val_o, convert_to_tensor=True)
                        emb2 = semantic_model.encode(val_p, convert_to_tensor=True)
                        sim = util.cos_sim(emb1, emb2).item()
                        
                        # Appliquer ce score aux lignes concernées si elles sont FP/FN
                        if df.loc[idx_o, 'type'] == 'fn':
                            df.loc[idx_o, 'sem_score'] = max(df.loc[idx_o, 'sem_score'], sim)
                        if df.loc[idx_p, 'type'] == 'fp':
                            df.loc[idx_p, 'sem_score'] = max(df.loc[idx_p, 'sem_score'], sim)

    df['sem_score'] = df['sem_score'].fillna(0.0)

    # --- 3. Synthèse par Feature ---
    summary = []
    for feat, group in df.groupby('feature'):
        tp = len(group[group['type'] == 'tp'])
        fp = len(group[group['type'] == 'fp'])
        fn = len(group[group['type'] == 'fn'])
        
        # Précision et Recall : calculés uniquement sur les cas présents (cohérent avec SEM)
        # Ne pas pénaliser les features rares avec missing_cases
        # Le poids de la feature (count) sera pris en compte au niveau global si nécessaire
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        count = tp + fn
        count_pct = (count / total_lines) * 100
        avg_sem = group['sem_score'].mean()
        
        # Score sémantique moyen pour cette feature
        # Moyenne = somme des sem_scores / nombre d'occurrences de la feature
        sem_with_missing = group['sem_score'].mean()
        
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

    # Créer un dictionnaire pour accès rapide aux features
    feat_dict = {}
    for idx, row in summary_df.iterrows():
        feat_dict[row['feat'].lower()] = row

    # Compter les features détectées (présentes dans feat_dict)
    features_detected = len(feat_dict)

    # CRÉER LA TABLE COMPLÈTE avec TOUTES les features dans l'ordre prévu (même manquantes)
    complete_features_list = []
    for order_feat in feature_order:
        order_feat_lower = order_feat.lower()
        if order_feat_lower in feat_dict:
            r = feat_dict[order_feat_lower]
            complete_features_list.append({
                'feat': r['feat'],
                'prec': r['prec'],
                'rec': r['rec'],
                'f1': r['f1'],
                'sem': r['sem'],
                'count': r['count'],
                'pct': r['pct'],
                'tp': r['tp'],
                'fp': r['fp'],
                'fn': r['fn']
            })
        else:
            # Feature manquante : ajouter avec "-"
            complete_features_list.append({
                'feat': order_feat,
                'prec': '-',
                'rec': '-',
                'f1': '-',
                'sem': '-',
                'count': '-',
                'pct': '-',
                'tp': '-',
                'fp': '-',
                'fn': '-'
            })

    # Convertir en DataFrame pour la sauvegarde
    summary_df_filtered = pd.DataFrame(complete_features_list)

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
    # Filtrer SEULEMENT les features détectées (exclure les "-")
    detected_mask = summary_df_filtered['prec'] != '-'
    summary_df_detected = summary_df_filtered[detected_mask]

    # Moyenne simple des scores de toutes les features détectées
    g_prec = summary_df_detected['prec'].mean() if len(summary_df_detected) > 0 else 0
    g_rec = summary_df_detected['rec'].mean() if len(summary_df_detected) > 0 else 0
    g_f1 = summary_df_detected['f1'].mean() if len(summary_df_detected) > 0 else 0

    # Score sémantique global : moyenne des SEM scores des features qui existent (les "-" ne sont pas comptées)
    g_sem = summary_df_detected['sem'].mean() if len(summary_df_detected) > 0 else 0

    print("-" * 95)

    # --- 5. TOTAL SIMPLE et TOTAL PONDÉRÉ ---
    # TOTAL SIMPLE : moyenne simple (toutes les features ont le même poids)
    print(f"{'TOTAL SIMPLE':<40} | {g_prec:.3f} | {g_rec:.3f} | {g_f1:.3f} | {g_sem:.3f}")

    # TOTAL PONDÉRÉ : moyenne pondérée par fréquence (count)
    total_count = summary_df_detected['count'].sum()
    g_prec_weighted = (summary_df_detected['prec'] * summary_df_detected['count']).sum() / total_count if total_count > 0 else 0
    g_rec_weighted = (summary_df_detected['rec'] * summary_df_detected['count']).sum() / total_count if total_count > 0 else 0
    g_f1_weighted = (summary_df_detected['f1'] * summary_df_detected['count']).sum() / total_count if total_count > 0 else 0
    g_sem_weighted = (summary_df_detected['sem'] * summary_df_detected['count']).sum() / total_count if total_count > 0 else 0
    print(f"{'TOTAL PONDÉRÉ':<40} | {g_prec_weighted:.3f} | {g_rec_weighted:.3f} | {g_f1_weighted:.3f} | {g_sem_weighted:.3f}")

    print("-" * 95)

    # --- 6. SAVE RESULTS ---
    print("\n💾 Saving results...")

    # Create base filename
    base_filename = f"result_{mode}_{table_number}"

    # Save XLSX (Excel format)
    xlsx_file = results_dir / f"{base_filename}.xlsx"
    with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
        # Convertir les points en virgules pour XLSX (format français) - 3 décimales
        summary_df_xlsx = summary_df_filtered.copy()
        for col in ['prec', 'rec', 'f1', 'sem', 'count']:
            if col in summary_df_xlsx.columns:
                summary_df_xlsx[col] = summary_df_xlsx[col].apply(
                    lambda x: f"{float(x):.3f}".replace('.', ',') if isinstance(x, (int, float)) else x
                )
        # pct en format décimal (5% → 0,05)
        if 'pct' in summary_df_xlsx.columns:
            summary_df_xlsx['pct'] = summary_df_xlsx['pct'].apply(
                lambda x: f"{float(x)/100:.2f}".replace('.', ',') if isinstance(x, (int, float)) else x
            )
        
        # Sheet 1: Features
        summary_df_xlsx.to_excel(writer, sheet_name='Features', index=False)
        
        # Sheet 2: Global Metrics
        metrics_data = {
            'Mode': [mode],
            'F1': [f"{g_f1:.3f}".replace('.', ',')],
            'SEM': [f"{g_sem:.3f}".replace('.', ',')],
            'W F1': [f"{g_f1_weighted:.3f}".replace('.', ',')],
            'W SEM': [f"{g_sem_weighted:.3f}".replace('.', ',')]
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


if __name__ == "__main__":
    main(table_number=4, mode="without_cot")