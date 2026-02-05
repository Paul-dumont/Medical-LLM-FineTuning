import json
from pathlib import Path
from transformers import AutoTokenizer
import os

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Phi-3.5-mini-instruct")

# Path to data
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "2_input_model"

# Modes to analyze
modes = ["tmj", "no_prompt", "unknow", "with_cot", "without_cot", "dry_run"]

results = {}

for mode in modes:
    mode_path = data_path / mode
    if not mode_path.exists():
        print(f"⚠️  {mode} folder not found")
        continue
    
    # Find all jsonl files
    jsonl_files = list(mode_path.glob("training_data_*.jsonl"))
    if not jsonl_files:
        continue
    
    max_tokens = 0
    min_tokens = float('inf')
    total_tokens = 0
    count = 0
    
    for jsonl_file in jsonl_files[:2]:  # Analyze first 2 files per mode
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # Get the assistant response (JSON extraction)
                response = data['messages'][-1]['content']
                
                # Tokenize
                tokens = tokenizer.encode(response)
                num_tokens = len(tokens)
                
                max_tokens = max(max_tokens, num_tokens)
                min_tokens = min(min_tokens, num_tokens)
                total_tokens += num_tokens
                count += 1
    
    if count > 0:
        avg_tokens = total_tokens / count
        results[mode] = {
            'min': min_tokens,
            'max': max_tokens,
            'avg': avg_tokens,
            'count': count,
            'recommended_max_new_tokens': int(max_tokens * 1.2)  # Add 20% buffer
        }

print("\n" + "="*80)
print("TOKEN ANALYSIS FOR JSON EXTRACTION RESPONSES")
print("="*80)

for mode, stats in sorted(results.items()):
    print(f"\n📊 {mode.upper()}:")
    print(f"   Min tokens: {stats['min']}")
    print(f"   Max tokens: {stats['max']}")
    print(f"   Avg tokens: {stats['avg']:.1f}")
    print(f"   Samples analyzed: {stats['count']}")
    print(f"   ✅ Recommended max_new_tokens: {stats['recommended_max_new_tokens']}")

# Calculate global max
global_max = max([s['recommended_max_new_tokens'] for s in results.values()])
print(f"\n🎯 GLOBAL RECOMMENDATION: max_new_tokens = {global_max}")
print(f"   (This will safely handle all modes)")

print("\n" + "="*80)
