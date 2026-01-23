import json
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer


#TO RUN:
table_number = 4
mode = "with_cot"  # Change to "with_cot" or "without_cot" to compare

print("-" * 95)
print(f" {mode}, Table {table_number}")
print("-" * 95)


# -----------------------------------------------------------------------------
# 1. Path Configuration
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent
json_path = str(project_root / "data"/"2_input_model"/f"{mode}"/f"training_data_{mode}{table_number}.jsonl")

# Load tokenizer for Phi-3.5
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Phi-3.5-mini-instruct")

# -----------------------------------------------------------------------------
# 2. Process Loop 
# -----------------------------------------------------------------------------
feature_counts = Counter()
total_records = 0
valid = 0
corrupted = 0
token_stats = {
    "total_tokens": 0,
    "record_tokens": [],
    "min_tokens": float('inf'),
    "max_tokens": 0
}

with open(json_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): # Ignore empty line
            continue
        try:
            record = json.loads(line)
            total_records += 1
            
            # Extract all messages and format them
            conversation = record.get("messages", [])
            formatted_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Count tokens for this record
            tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
            num_tokens = len(tokens)
            token_stats["total_tokens"] += num_tokens
            token_stats["record_tokens"].append(num_tokens)
            token_stats["min_tokens"] = min(token_stats["min_tokens"], num_tokens)
            token_stats["max_tokens"] = max(token_stats["max_tokens"], num_tokens)
            
            # Feature counting
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

# ============ TOKENS ANALYSIS ============
print("\n" + "="*70)
print("TOKENS ANALYSIS (Context Window)")
print("="*70)
if token_stats["record_tokens"]:
    avg_tokens = token_stats["total_tokens"] / len(token_stats["record_tokens"])
    print(f"Total Tokens:           {token_stats['total_tokens']:,}")
    print(f"Average Tokens/Record:  {avg_tokens:.1f}")
    print(f"Min Tokens:             {token_stats['min_tokens']}")
    print(f"Max Tokens:             {token_stats['max_tokens']}")
    
    # Comparaison 1024 vs 2048
    print("\n" + "-"*70)
    print("COMPARISON: 1024 vs 2048 max_seq_length")
    print("-"*70)
    
    # Compter records > 1024
    records_above_1024 = sum(1 for t in token_stats["record_tokens"] if t > 1024)
    pct_above_1024 = (records_above_1024 / len(token_stats["record_tokens"])) * 100
    
    print(f"Records > 1024 tokens:  {records_above_1024} ({pct_above_1024:.1f}%)")
    print(f"Records ≤ 1024 tokens:  {len(token_stats['record_tokens']) - records_above_1024} ({100-pct_above_1024:.1f}%)")
    
    # Tokens perdus avec troncature à 1024
    tokens_lost_1024 = sum(max(0, t - 1024) for t in token_stats["record_tokens"])
    pct_tokens_lost = (tokens_lost_1024 / token_stats["total_tokens"]) * 100
    
    print(f"\nTokens perdus avec 1024: {tokens_lost_1024:,} ({pct_tokens_lost:.1f}%)")
    print(f"\nVotre choix:")
    print(f"  • 1024: Perd {pct_above_1024:.1f}% des records et {pct_tokens_lost:.1f}% des tokens")
    print(f"  • 2048: 0% perdu mais 2x plus de mémoire")
    
    print(f"\nMax seq_length (train): 2048")
    print(f"Tokens/2048 max:        {token_stats['total_tokens'] / 2048:.1f}x")
    print(f"\nVotre dataset aura besoin de ~{token_stats['total_tokens']:,} tokens au total")

print("\n" + "="*70)
print("FEATURES ANALYSIS")
print("="*70)
print("\nFEATURE" + " "*37 + "| COUNT | COUNT %")
print("-" * 70)

for feature, count in feature_counts.most_common():
    percentage = (count * 100) // total_records if total_records > 0 else 0
    print(f"{feature:<45} | {count:5} | {percentage:3} %")

print("-" * 70)
print(f"TOTAL: {total_records}")
    
print("-" * 95)
print(f" {mode}, Table {table_number}")
print("-" * 95)