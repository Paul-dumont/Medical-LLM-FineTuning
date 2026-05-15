import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
DIR_PATH = '/media/luciacev/Data/Medical-LLM-FineTuning/data/5_results/no_prompt'
OUTPUT_DIR = '/media/luciacev/Data/Medical-LLM-FineTuning/graph/compare_splits/'

files = {
    'All': os.path.join(DIR_PATH, 'result_llama_no_prompt_6_all.xlsx'),
    'Validation': os.path.join(DIR_PATH, 'result_llama_no_prompt_6_validation.xlsx'),
    'Test': os.path.join(DIR_PATH, 'result_llama_no_prompt_6_test.xlsx')
}

def parse_value(val):
    if pd.isna(val) or val == '-':
        return None
    if isinstance(val, str):
        return float(val.replace(',', '.').replace(' ', ''))
    return float(val)

results = []

print("Loading datasets...")
for split_name, path in files.items():
    if not os.path.exists(path):
        print(f"File missing: {path}")
        continue
    df_raw = pd.read_excel(path)
    
    # Clean F1 and SEM
    df = df_raw.copy()
    for col in ['f1', 'sem']:
        if col in df.columns:
            df[col] = df[col].apply(parse_value)
    
    # Calculate means dropping NaNs
    f1_mean = df['f1'].dropna().mean()
    sem_mean = df['sem'].dropna().mean()
    f1_weighted = np.average(df['f1'].dropna(), weights=df.loc[df['f1'].notna(), 'count'].apply(parse_value)) if 'count' in df.columns else f1_mean
    sem_weighted = np.average(df['sem'].dropna(), weights=df.loc[df['sem'].notna(), 'count'].apply(parse_value)) if 'count' in df.columns else sem_mean

    results.append({
        'Split': split_name,
        'Mean_F1': f1_mean,
        'Mean_SEM': sem_mean,
    })
    print(f"{split_name}: F1 Mean = {f1_mean:.3f}, SEM Mean = {sem_mean:.3f}")

if not results:
    print("No data loaded. Exiting.")
    exit(0)

df_results = pd.DataFrame(results)

# --- Plotting ---
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

# Palette "scientifique" et colorblind-friendly (Nuances de bleu/gris élégantes)
colors = ['#1f4e79', '#4682b4', '#9bc2e6']

# F1 Plot
bars1 = ax1.bar(df_results['Split'], df_results['Mean_F1'], color=colors, edgecolor='black', linewidth=1.2, alpha=0.9, width=0.6)
ax1.set_title('Average F1 Score Comparison', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('Mean F1 Score (Macro)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
ax1.set_axisbelow(True)  # La grille derrière les barres
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.015, f'{yval:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# SEM Plot
bars2 = ax2.bar(df_results['Split'], df_results['Mean_SEM'], color=colors, edgecolor='black', linewidth=1.2, alpha=0.9, width=0.6)
ax2.set_title('Average Semantic Score (SEM) Comparison', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylabel('Mean Semantic Score (Macro)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.0)
ax2.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
ax2.set_axisbelow(True)  # La grille derrière les barres
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.015, f'{yval:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle("Comparison of Outputs across Splits", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, 'splits_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved Figure to: {output_path}")