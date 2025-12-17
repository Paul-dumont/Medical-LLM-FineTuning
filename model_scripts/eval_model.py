import json 
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. Path Configuration
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent #.resolve convert Relatif path into absolut path (./../home) into (~user/inux/home), .parent to keep the parent folder of the current file 
project_root = script_folder.parent # move up one level, to get the root project folder

json_path = str(project_root / "data" / "3_output_model" / "extraction.jsonl")


# -----------------------------------------------------------------------------
# 2. Function
# -----------------------------------------------------------------------------
def get_pairs(dict):
    if dict is None: return set()
    pairs = set()
    for key, value in dict.items():
        key_norm = str(key).lower().strip()

        if isinstance(value, list):
            sorted_list = sorted([str(x).lower().strip() for x in value])
            value_norm = str(sorted_list)
        else: value_norm = str(value).lower().strip()
        
        pairs.add((key_norm, value_norm))
    return pairs

# -----------------------------------------------------------------------------
# 3. Evaluation Loop 
# -----------------------------------------------------------------------------
stats = {}
total_lines = 0

with open(json_path, "r", encoding="utf-8") as json_file:
    for i, line in enumerate(json_file): # Loop on records
        total_lines +=1
        patient_record = json.loads(line)

        original = patient_record["original"]
        prediction = patient_record["prediction"]

        # Parsing original (truth) - extract only "extraction" field
        try : 
            original_dict = json.loads(original).get("extraction", {})
        except: 
            original_dict = {}
            print(f" ERROR : json not valid for original extraction, line : {i+1}")

        # Parsing prediction - extract only "extraction" field
        try : 
            prediction_dict = json.loads(prediction).get("extraction", {})
        except: 
            prediction_dict = {}
            print(f" ERROR : json not valid for prediction extraction, line : {i+1} ")

        original_set = get_pairs(original_dict)
        prediction_set = get_pairs(prediction_dict)

        for item in original_set.intersection(prediction_set):
            feature_name = item[0]
            if feature_name not in stats: stats[feature_name] = {'tp': 0, 'fp': 0, 'fn': 0}
            stats[feature_name]['tp'] += 1
        
        for item in (prediction_set - original_set):
            feature_name = item[0]
            if feature_name not in stats: stats[feature_name] = {'tp': 0, 'fp': 0, 'fn': 0}
            stats[feature_name]['fp'] += 1

        for item in (original_set - prediction_set):
            feature_name = item[0]
            if feature_name not in stats: stats[feature_name] = {'tp': 0, 'fp': 0, 'fn': 0}
            stats[feature_name]['fn'] += 1


global_true_positif = 0
global_false_positif = 0
global_false_negatif = 0


print(f"{'FEATURE':<40} | {'F1':<6} | {'COUNT':<5} | {'COUNT %':<5}" )

# Tri des features par importance (nombre d'occurences)
sorted_stats = sorted(stats.items(), key=lambda x: (x[1]['tp'] + x[1]['fn']), reverse=True)

for feature, counts in sorted_stats:
    tp = counts['tp']
    fp = counts['fp']
    fn = counts['fn']
    
    # On ajoute au total global
    global_true_positif += tp
    global_false_positif += fp
    global_false_negatif += fn

    # Calcul métriques locales
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    count = tp + fn 
    count_percent = count / total_lines * 100

    print(f"{feature:<40} | {f1:.2f}  | {count:<5} | {count_percent:.0f} %")

print("-" * 65)

# --- CALCUL DU F1 GLOBAL (Comme dans ton code original) ---
if (global_true_positif + global_false_positif) > 0:
    precision = global_true_positif / (global_true_positif + global_false_positif)
else : precision = 0

if (global_true_positif + global_false_negatif) > 0:
    recall = global_true_positif / (global_true_positif + global_false_negatif)
else : recall = 0

if (precision + recall) > 0:
    f1 = 2 * (precision * recall) / (precision + recall)
else : f1 = 0

print(f"{'TOTAL GLOBAL':<40} | {f1:.4f} | {total_lines}")



