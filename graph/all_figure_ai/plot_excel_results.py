import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
EXCEL_PATH = '/media/luciacev/Data/Medical-LLM-FineTuning/data/5_results/no_prompt/result_llama_no_prompt_6_all.xlsx'
OUTPUT_DIR = '/media/luciacev/Data/Medical-LLM-FineTuning/graph/performance_representation/'

# --- Load & Clean data ---
print(f"Loading data from: {EXCEL_PATH}")
df_raw = pd.read_excel(EXCEL_PATH)

def parse_value(val):
    """Convert French decimal format (comma) to float, handle dashes and NaNs."""
    if pd.isna(val) or val == '-':
        return None
    if isinstance(val, str):
        return float(val.replace(',', '.').replace(' ', ''))
    return float(val)

df = df_raw.copy()
for col in ['prec', 'rec', 'f1', 'sem', 'pct']:
    df[col] = df[col].apply(parse_value)

# Filter out features with no F1 score
df = df.dropna(subset=['f1', 'pct']).copy()
df['pct'] = df['pct'] * 100  # Convert to percentage (0-100 scale)

# --- Plot 1: F1 and SEM vs Feature Frequency (Scatter plot) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: F1 Score vs Frequency
ax1.scatter(df['pct'], df['f1'], alpha=0.7, color='#1f77b4', s=80, edgecolor='black', linewidth=0.5)
if len(df) > 1:
    z1 = np.polyfit(df['pct'], df['f1'], 1)
    p1 = np.poly1d(z1)
    x_smooth = np.linspace(df['pct'].min(), df['pct'].max(), 100)
    ax1.plot(x_smooth, p1(x_smooth), color='red', linestyle='--', linewidth=2, label=f'Trend line')

ax1.set_xlabel('Feature Frequency in Ground Truth (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax1.set_title('F1 Score vs Feature Frequency', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.05)
ax1.legend()

# Subplot 2: Semantic Score vs Frequency
df_sem = df.dropna(subset=['sem'])
ax2.scatter(df_sem['pct'], df_sem['sem'], alpha=0.7, color='#2ca02c', s=80, edgecolor='black', linewidth=0.5)
if len(df_sem) > 1:
    z2 = np.polyfit(df_sem['pct'], df_sem['sem'], 1)
    p2 = np.poly1d(z2)
    x_smooth2 = np.linspace(df_sem['pct'].min(), df_sem['pct'].max(), 100)
    ax2.plot(x_smooth2, p2(x_smooth2), color='red', linestyle='--', linewidth=2, label=f'Trend line')

ax2.set_xlabel('Feature Frequency in Ground Truth (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Semantic Score (SEM)', fontsize=12, fontweight='bold')
ax2.set_title('Semantic Similarity vs Feature Frequency', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 1.05)
ax2.legend()

plt.tight_layout()
scatter_path = os.path.join(OUTPUT_DIR, 'scatter_f1_sem_vs_frequency.png')
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved Figure 1: {scatter_path}")

# --- Plot 2: Top 15 Best and Bottom 15 Worst Features by F1 Score ---
df_sorted = df.sort_values('f1', ascending=False)
top_15 = df_sorted.head(15)
bottom_15 = df_sorted.tail(15)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Top 15
ax1.barh(top_15['feat'][::-1], top_15['f1'][::-1], color='#2ca02c', edgecolor='black')
ax1.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
ax1.set_title('Top 15 BEST Features (F1 Score)', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 1.05)
ax1.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_15['f1'][::-1]):
    ax1.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=10)

# Bottom 15
ax2.barh(bottom_15['feat'][::-1], bottom_15['f1'][::-1], color='#d62728', edgecolor='black')
ax2.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
ax2.set_title('Top 15 WORST Features (F1 Score)', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 1.05)
ax2.grid(axis='x', alpha=0.3)
for i, v in enumerate(bottom_15['f1'][::-1]):
    ax2.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=10)

plt.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, 'bar_best_worst_features.png')
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved Figure 2: {bar_path}")

print("Done! You can now view the graphs in the output folder.")