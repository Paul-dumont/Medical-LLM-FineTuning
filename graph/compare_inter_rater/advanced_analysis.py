import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Paths ---
DATA_DIR = '/media/luciacev/Data/Medical-LLM-FineTuning/graph/compare_inter_rater/'
OUT_DIR = '/media/luciacev/Data/Medical-LLM-FineTuning/graph/compare_inter_rater/figures/'

# Load data
df_inter = pd.read_excel(os.path.join(DATA_DIR, 'inter_annotator_agreement.xlsx'))
df_model = pd.read_excel(os.path.join(DATA_DIR, 'model_vs_humans_evaluation.xlsx'))

# Set global style
plt.style.use('default')
colors = ['#1f4e79', '#4682b4', '#9bc2e6', '#c9daf8', '#e2efda']

# Convert columns if necessary
df_inter['Cohen Kappa'] = pd.to_numeric(df_inter['Cohen Kappa'], errors='coerce')
df_inter['F1-Score'] = pd.to_numeric(df_inter['F1-Score'], errors='coerce')
df_inter['Agreement %'] = pd.to_numeric(df_inter['Agreement %'], errors='coerce')

df_model['Soft TP'] = pd.to_numeric(df_model['Soft TP'], errors='coerce')
df_model['Soft FP'] = pd.to_numeric(df_model['Soft FP'], errors='coerce')
df_model['Soft FN'] = pd.to_numeric(df_model['Soft FN'], errors='coerce')
df_model['F1 vs Either (Soft)'] = pd.to_numeric(df_model['F1 vs Either (Soft)'], errors='coerce')

# Merge to compare Human vs Model F1 directly
df_comp = pd.merge(df_inter[['Feature', 'F1-Score', 'Agreement %']],
                   df_model[['Feature', 'F1 vs Either (Soft)', 'Count']],
                   on='Feature', how='inner')
df_comp = df_comp.sort_values(by='Count', ascending=False).head(20)

# ==============================================================================
# Figure 1: Inter-Annotator Agreement (Cohen Kappa & % Agreement)
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(14, 7), facecolor='white')
df_inter_sorted = df_inter.sort_values(by='Cohen Kappa', ascending=True).tail(20) # Top 20 features
y = np.arange(len(df_inter_sorted))

ax1.barh(y - 0.2, df_inter_sorted['Cohen Kappa'], height=0.4, label='Cohen Kappa', color=colors[0], edgecolor='black')
ax1.barh(y + 0.2, df_inter_sorted['Agreement %'], height=0.4, label='Absolute Agreement Rate', color=colors[2], edgecolor='black')

ax1.set_yticks(y)
ax1.set_yticklabels(df_inter_sorted['Feature'])
ax1.set_xlabel('Score', fontweight='bold')
ax1.set_title('Inter-Annotator Reliability (Human 1 vs Human 2) - Top 20 Features', fontweight='bold', pad=15)
ax1.set_xlim(0, 1.1)
ax1.legend()
ax1.grid(axis='x', linestyle='--', alpha=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, '01_inter_annotator_reliability.png'), dpi=300)

# ==============================================================================
# Figure 2: Model Performance vs Human F1 Score
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(14, 8), facecolor='white')
y = np.arange(len(df_comp))

ax2.barh(y - 0.2, df_comp['F1-Score'][::-1], height=0.4, label='Human Inter-Rater F1', color='#4682b4', edgecolor='black')
ax2.barh(y + 0.2, df_comp['F1 vs Either (Soft)'][::-1], height=0.4, label='Model "Soft" F1', color='#ff9933', edgecolor='black')

ax2.set_yticks(y)
ax2.set_yticklabels(df_comp['Feature'][::-1])
ax2.set_xlabel('F1 Score', fontweight='bold')
ax2.set_title('Model Intelligence vs Human Consistency (Top 20 Frequent Features)', fontweight='bold', pad=15)
ax2.set_xlim(0, 1.1)
ax2.legend(loc='lower left')
ax2.grid(axis='x', linestyle='--', alpha=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for i, feature_y in enumerate(y):
    h_f1 = df_comp['F1-Score'][::-1].iloc[i]
    m_f1 = df_comp['F1 vs Either (Soft)'][::-1].iloc[i]
    if pd.notna(h_f1) and pd.notna(m_f1):
        diff = m_f1 - h_f1
        color = '#2ca02c' if diff > 0 else '#d62728'
        sign = '+' if diff > 0 else ''
        # N'afficher le texte que s'il est positif (vert), et masquer le rouge
        if diff > 0:
            ax2.text(max(h_f1, m_f1) + 0.02, feature_y, f"{sign}{diff*100:.1f} pts", va='center', color=color, fontweight='bold', fontsize=9)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, '02_model_vs_human_f1.png'), dpi=300)

# ==============================================================================
# Figure 3: Model Error Breakdown (False Positives vs False Negatives)
# ==============================================================================
df_err = df_model.sort_values(by='Count', ascending=False).head(15).copy()
df_err['Total Errors'] = df_err['Soft FP'] + df_err['Soft FN']

# Only keep those with actual errors
df_err = df_err[df_err['Total Errors'] > 0].sort_values(by='Total Errors', ascending=True)

fig3, ax3 = plt.subplots(figsize=(12, 7), facecolor='white')
y = np.arange(len(df_err))

ax3.barh(y, df_err['Soft FP'], label='False Positives (Hallucinations)', color='#d62728', edgecolor='black', alpha=0.8)
ax3.barh(y, df_err['Soft FN'], left=df_err['Soft FP'], label='False Negatives (Missed Extractions)', color='#ff9896', edgecolor='black', alpha=0.8)

ax3.set_yticks(y)
ax3.set_yticklabels(df_err['Feature'])
ax3.set_xlabel('Number of Errors', fontweight='bold')
ax3.set_title('Model Error Analysis: Hallucinations vs Misses (Top 15 Features)', fontweight='bold', pad=15)
ax3.legend(loc='lower right')
ax3.grid(axis='x', linestyle='--', alpha=0.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

for i, total in enumerate(df_err['Total Errors']):
    ax3.text(total + 0.5, y[i], f'{int(total)}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, '03_model_error_types.png'), dpi=300)

# ==============================================================================
# Figure 4: The Triangle Comparison (H1 vs H2, Model vs H1, Model vs H2)
# ==============================================================================
# Merge for triangle comparison
df_triangle = pd.merge(df_inter[['Feature', 'F1-Score']], 
                       df_model[['Feature', 'F1 vs Hum1', 'F1 vs Hum2', 'Count']], 
                       on='Feature', how='inner')
                       
# Sort by feature frequency to show the most important ones
df_triangle = df_triangle.sort_values(by='Count', ascending=False).head(15)

fig4, ax4 = plt.subplots(figsize=(16, 8), facecolor='white')
x = np.arange(len(df_triangle))
width = 0.25

# Plotting the three comparisons side by side
ax4.bar(x - width, df_triangle['F1-Score'], width, label='Human 1 vs Human 2 (Inter-Annotator)', color='#1f4e79', edgecolor='black')
ax4.bar(x, df_triangle['F1 vs Hum1'], width, label='Model vs Human 1', color='#ff7f0e', edgecolor='black')
ax4.bar(x + width, df_triangle['F1 vs Hum2'], width, label='Model vs Human 2', color='#2ca02c', edgecolor='black')

ax4.set_xticks(x)
ax4.set_xticklabels(df_triangle['Feature'], rotation=45, ha='right', fontweight='bold')
ax4.set_ylabel('F1 Score', fontweight='bold')
ax4.set_title('Agreement Comparison: Humans vs Humans AND Model vs Humans (Top 15 Features)', fontweight='bold', fontsize=14, pad=15)
ax4.set_ylim(0, 1.15)
ax4.legend(loc='upper right', fontsize=11)
ax4.grid(axis='y', linestyle='--', alpha=0.5)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, '04_triangle_comparison.png'), dpi=300)

# ==============================================================================
# Figure 5: Overall Average Performance Comparison (Macro F1)
# ==============================================================================
# Merge on all common features to get a true overall view
df_all_common = pd.merge(df_inter[['Feature', 'F1-Score']], 
                         df_model[['Feature', 'F1 vs Hum1', 'F1 vs Hum2', 'F1 vs Either (Soft)']], 
                         on='Feature', how='inner')

# Convert remaining columns to numeric just in case there are strings left
for col in ['F1-Score', 'F1 vs Hum1', 'F1 vs Hum2', 'F1 vs Either (Soft)']:
    df_all_common[col] = pd.to_numeric(df_all_common[col], errors='coerce')

# Calculate the Macro Average across all common features (dropping NaNs if any)
mean_h1_h2 = df_all_common['F1-Score'].dropna().mean()
mean_m_h1 = df_all_common['F1 vs Hum1'].dropna().mean()
mean_m_h2 = df_all_common['F1 vs Hum2'].dropna().mean()
mean_m_soft = df_all_common['F1 vs Either (Soft)'].dropna().mean()

fig5, ax5 = plt.subplots(figsize=(12, 7), facecolor='white')
categories = ['Human 1\nvs Human 2\n(Inter-Rater)', 'Model\nvs Human 1', 'Model\nvs Human 2', 'Model\nvs Either\n(Soft Match)', 'Latest AI model\nwithout training\n(Gemini 3 Pro)']
means = [mean_h1_h2, mean_m_h1, mean_m_h2, mean_m_soft, 0.448]
bar_colors = ['#2ca02c', '#2ca02c', '#2ca02c', '#2ca02c', '#d62728']

bars5 = ax5.bar(categories, means, color=bar_colors, edgecolor='black', width=0.5, alpha=0.9)
ax5.set_ylabel('Overall Macro Average F1 Score', fontsize=12, fontweight='bold')
ax5.set_title('Overall Model Performance vs Human Agreement', fontsize=14, fontweight='bold', pad=20)
ax5.set_ylim(0, 1.05)
ax5.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax5.set_axisbelow(True)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

for bar in bars5:
    yval = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width() / 2, yval + 0.015, f'{yval:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.tight_layout()
fig5.savefig(os.path.join(OUT_DIR, '05_overall_comparison.png'), dpi=300)

# ==============================================================================
# Figure 6: Overall Weighted Average Performance Comparison
# ==============================================================================
df_weighted = pd.merge(df_inter[['Feature', 'F1-Score']], 
                         df_model[['Feature', 'F1 vs Hum1', 'F1 vs Hum2', 'F1 vs Either (Soft)', 'Count']], 
                         on='Feature', how='inner')

for col in ['F1-Score', 'F1 vs Hum1', 'F1 vs Hum2', 'F1 vs Either (Soft)', 'Count']:
    df_weighted[col] = pd.to_numeric(df_weighted[col], errors='coerce')
    
df_weighted = df_weighted.dropna(subset=['Count'])

def get_weighted_avg(df, score_col, weight_col='Count'):
    mask = df[score_col].notna() & df[weight_col].notna()
    if not mask.any():
        return 0
    return np.average(df.loc[mask, score_col], weights=df.loc[mask, weight_col])

w_h1_h2 = get_weighted_avg(df_weighted, 'F1-Score')
w_m_h1 = get_weighted_avg(df_weighted, 'F1 vs Hum1')
w_m_h2 = get_weighted_avg(df_weighted, 'F1 vs Hum2')
w_m_soft = get_weighted_avg(df_weighted, 'F1 vs Either (Soft)')

fig6, ax6 = plt.subplots(figsize=(12, 7), facecolor='white')
means_w = [w_h1_h2, w_m_h1, w_m_h2, w_m_soft, 0.448]

bars6 = ax6.bar(categories, means_w, color=bar_colors, edgecolor='black', width=0.5, alpha=0.9)
ax6.set_ylabel('Overall *Weighted* Average F1 Score', fontsize=12, fontweight='bold')
ax6.set_title('Overall Weighted Performance vs Human Agreement\n(Weighted by feature frequency)', fontsize=14, fontweight='bold', pad=20)
ax6.set_ylim(0, 1.05)
ax6.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax6.set_axisbelow(True)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)

for bar in bars6:
    yval = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width() / 2, yval + 0.015, f'{yval:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.tight_layout()
fig6.savefig(os.path.join(OUT_DIR, '06_overall_weighted_comparison.png'), dpi=300)

print("✓ Successfully generated advanced analysis plots in graph/compare_inter_rater/figures/")
