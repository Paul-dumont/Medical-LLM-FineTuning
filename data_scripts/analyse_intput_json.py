import json
from pathlib import Path
from collections import Counter

script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent
json_path = project_root / "data" / "2_input_model" / "training_data_2.jsonl"

feature_counts = Counter()
total_records = 0
valid = 0
corrupted = 0

with open(json_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            total_records += 1
            
            for msg in record.get("messages", []):
                if msg.get("role") == "assistant":
                    content = json.loads(msg.get("content", "{}"))
                    extraction = content.get("extraction", {})
                    for feature in extraction.keys():
                        feature_counts[feature] += 1
            valid += 1
        except:
            corrupted += 1

print(f"\n✓ Valid: {valid}  |  ✗ Corrupted: {corrupted}")
print("\nFEATURE" + " "*37 + "| COUNT | COUNT %")
print("-" * 70)

for feature, count in feature_counts.most_common():
    percentage = (count * 100) // total_records if total_records > 0 else 0
    print(f"{feature:<45} | {count:5} | {percentage:3} %")

print("-" * 70)
print(f"TOTAL: {total_records}")
