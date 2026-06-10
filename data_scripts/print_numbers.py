import json
from pathlib import Path

# TO RUN:
table_number = "6"
mode = "no_prompt"  # "with_cot", "without_cot", or "dry_run"

# -----------------------------------------------------------------------------
# 1. Path Configuration
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent

json_path = project_root / "data" / "2_input_model" / mode / f"training_data_{mode}{table_number}.jsonl"

# Check if file exists
if not json_path.exists():
    print(f"❌ File not found: {json_path}")
    exit(1)

# -----------------------------------------------------------------------------
# 2. Initialisation des compteurs
# -----------------------------------------------------------------------------
unique_patients = set()  # Un set (ensemble) ne garde que les valeurs uniques
total_records = 0
total_features = 0

# -----------------------------------------------------------------------------
# 3. Lecture et traitement du fichier
# -----------------------------------------------------------------------------
print(f"Lecture des données depuis : {json_path} ...\n")

with open(json_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
            
        try:
            # Charger la ligne JSON
            record = json.loads(line)
            
            # 1. Compter le record (1 ligne = 1 record)
            total_records += 1
            
            # 2. Compter le patient (ajout au set)
            patient_id = record.get("metadata", {}).get("patient_id")
            if patient_id:
                unique_patients.add(patient_id)
                
            # 3. Compter les features annotées
            # On cherche le message de l'assistant qui contient les extractions
            for message in record.get("messages", []):
                if message.get("role") == "assistant":
                    content_str = message.get("content", "{}")
                    try:
                        # Le 'content' est une string contenant du JSON, il faut le parser
                        content_json = json.loads(content_str)
                        extraction = content_json.get("extraction", {})
                        
                        # On ajoute le nombre de features extraites (la taille du dictionnaire)
                        total_features += len(extraction)
                        
                    except json.JSONDecodeError:
                        print(f"⚠️ Erreur de décodage du 'content' pour le patient {patient_id}")
                        
        except json.JSONDecodeError:
            print("⚠️ Ligne JSON invalide ignorée.")

# -----------------------------------------------------------------------------
# 4. Affichage des résultats
# -----------------------------------------------------------------------------
print("=" * 40)
print("📊 RÉSUMÉ DES DONNÉES")
print("=" * 40)
print(f"Total de patients uniques : {len(unique_patients)}")
print(f"Total de records (notes)  : {total_records}")
print(f"Total de features annotées: {total_features}")
print("=" * 40)