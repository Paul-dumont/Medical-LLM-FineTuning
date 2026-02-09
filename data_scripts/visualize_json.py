import json
from pathlib import Path


# TO RUN:
table_number = 1
mode = "dry_run"  # "with_cot", "without_cot", or "dry_run"
element_index = 1  # Index of the element to visualize (0 = first, 1 = second, etc.)


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

print(f"📂 Loading from: {json_path}")
print(f"📋 Mode: {mode}")
print(f"🔍 Visualizing element index: {element_index}\n")

# -----------------------------------------------------------------------------
# 2. Load and Display Element
# -----------------------------------------------------------------------------
with open(json_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Check if index is valid
if element_index >= len(lines) or element_index < 0:
    print(f"❌ Index out of range! File has {len(lines)} elements (indices 0-{len(lines)-1})")
    exit(1)

# Load the specific element
element = json.loads(lines[element_index])

# Display information
print(f"✅ Total elements in file: {len(lines)}")
print(f"✅ Displaying element #{element_index}\n")

print("=" * 80)
print("METADATA")
print("=" * 80)
if "metadata" in element:
    print(json.dumps(element["metadata"], indent=2, ensure_ascii=False))
else:
    print("⚠️  No metadata found")

print("\n" + "=" * 80)
print("MESSAGES")
print("=" * 80)
if "messages" in element:
    messages = element["messages"]
    for i, msg in enumerate(messages):
        print(f"\n--- Message {i+1} ({msg['role'].upper()}) ---")
        print(msg["content"][:500] if len(msg["content"]) > 500 else msg["content"])
        if len(msg["content"]) > 500:
            print(f"... ({len(msg['content'])} chars total)")
else:
    print("⚠️  No messages found")

print("\n" + "=" * 80)
print("FULL ELEMENT (JSON FORMAT)")
print("=" * 80)
print(json.dumps(element, indent=2, ensure_ascii=False))

# Additional statistics
print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)
if "metadata" in element:
    print(f"Patient ID: {element['metadata'].get('patient_id', 'N/A')}")
    print(f"Note Date: {element['metadata'].get('note_date', 'N/A')}")
    print(f"Note Month: {element['metadata'].get('note_month', 'N/A')}")

if "messages" in element and len(element["messages"]) > 2:
    assistant_msg = element["messages"][-1]["content"]
    extraction = json.loads(assistant_msg)
    features_count = len(extraction.get("extraction", {}))
    print(f"Features extracted: {features_count}")
    if "thought" in extraction:
        print(f"CoT thoughts: {extraction['thought'][:100]}...")
