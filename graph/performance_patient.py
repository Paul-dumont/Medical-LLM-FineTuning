import matplotlib.pyplot as plt
import pandas as pd

# Données F1
data_f1 = {
    'Categorie': ['Macro F1', 'Weighted F1'],
    '14 Patients': [0.069, 0.190],
    '18 Patients': [0.093, 0.225],
    '22 Patients': [0.087, 0.220],
    '44 Patients': [0.101, 0.300]
}

# Données Semantic (Ordre : [Macro, Weighted])
data_sem = {
    'Categorie': ['Macro Semantic Similarity', 'Weighted Semantic Similarity'],
    '14 Patients': [0.059, 0.163],
    '18 Patients': [0.078, 0.184],
    '22 Patients': [0.073, 0.178],
    '44 Patients': [0.086, 0.247]
}

df_f1 = pd.DataFrame(data_f1)
df_sem = pd.DataFrame(data_sem)

patient_counts = {
    '14 Patients': 14,
    '18 Patients': 18,
    '22 Patients': 22,
    '44 Patients': 44
}

# Créer 4 graphiques (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Macro F1
row = df_f1[df_f1['Categorie'] == 'Macro F1'].iloc[0]
x_vals = [patient_counts[col] for col in df_f1.columns if col != 'Categorie']
y_vals = [row[col] for col in df_f1.columns if col != 'Categorie']
sorted_points = sorted(zip(x_vals, y_vals))
x_sorted, y_sorted = zip(*sorted_points)
axes[0, 0].plot(x_sorted, y_sorted, marker='o', linewidth=2.5, markersize=8, color='#1f77b4')
for i, (x, y) in enumerate(zip(x_sorted, y_sorted)):
    axes[0, 0].text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=10)
axes[0, 0].set_title('F1 - Macro (Unweighted)', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('F1 Score', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks([14, 18, 22, 44])
axes[0, 0].set_ylim(0, 1)

# Weighted F1
row = df_f1[df_f1['Categorie'] == 'Weighted F1'].iloc[0]
y_vals = [row[col] for col in df_f1.columns if col != 'Categorie']
sorted_points = sorted(zip(x_vals, y_vals))
x_sorted, y_sorted = zip(*sorted_points)
axes[0, 1].plot(x_sorted, y_sorted, marker='o', linewidth=2.5, markersize=8, color='#ff7f0e')
for i, (x, y) in enumerate(zip(x_sorted, y_sorted)):
    axes[0, 1].text(x, y + 0.003, f'{y:.3f}', ha='center', va='bottom', fontsize=10)
axes[0, 1].set_title('F1 - Weighted', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks([14, 18, 22, 44])
axes[0, 1].set_ylim(0, 1)

# Macro Semantic Similarity
row = df_sem[df_sem['Categorie'] == 'Macro Semantic Similarity'].iloc[0]
x_vals = [patient_counts[col] for col in df_sem.columns if col != 'Categorie']
y_vals = [row[col] for col in df_sem.columns if col != 'Categorie']
sorted_points = sorted(zip(x_vals, y_vals))
x_sorted, y_sorted = zip(*sorted_points)
axes[1, 0].plot(x_sorted, y_sorted, marker='o', linewidth=2.5, markersize=8, color='#2ca02c')
for i, (x, y) in enumerate(zip(x_sorted, y_sorted)):
    axes[1, 0].text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=10)
axes[1, 0].set_title('Semantic - Macro (Unweighted)', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Number of Patients', fontsize=12)
axes[1, 0].set_ylabel('Semantic Score', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks([14, 18, 22, 44])
axes[1, 0].set_ylim(0, 1)

# Weighted Semantic Similarity
row = df_sem[df_sem['Categorie'] == 'Weighted Semantic Similarity'].iloc[0]
y_vals = [row[col] for col in df_sem.columns if col != 'Categorie']
sorted_points = sorted(zip(x_vals, y_vals))
x_sorted, y_sorted = zip(*sorted_points)
axes[1, 1].plot(x_sorted, y_sorted, marker='o', linewidth=2.5, markersize=8, color='#d62728')
for i, (x, y) in enumerate(zip(x_sorted, y_sorted)):
    axes[1, 1].text(x, y + 0.003, f'{y:.3f}', ha='center', va='bottom', fontsize=10)
axes[1, 1].set_title('Semantic - Weighted', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Number of Patients', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks([14, 18, 22, 44])
axes[1, 1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('/media/luciacev/Data/Medical-LLM-FineTuning/graph/performance_patient.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved: performance_patient.png")
plt.show()