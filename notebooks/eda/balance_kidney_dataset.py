"""
Apply SMOTE to Chronic Kidney Disease Dataset
File: src/data/balance_kidney_dataset.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

BASE = 'C:/Users/DELL/OneDrive/ドキュメント/BTP/PROJECT'
INPUT_PATH = f'{BASE}/datasets/chronic_kidney_disease.csv'
OUTPUT_PATH = f'{BASE}/datasets/chronic_kidney_disease_balanced.csv'
FIG_PATH = f'{BASE}/notebooks/eda/figures/kidney/smote_comparison.png'

print("SMOTE OVERSAMPLING - CKD Dataset (Recommended 1:3 Ratio)")
print("="*60)

df = pd.read_csv(INPUT_PATH)
print(f"Original: {len(df)} samples")

target = 'Diagnosis'
X = df.drop(['PatientID', target], axis=1)
y = df[target]

print(f"Before: Healthy={sum(y==0)}, CKD={sum(y==1)}, Ratio={sum(y==1)/sum(y==0):.2f}:1")

# Encode categoricals
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
X_enc = X.copy()
enc_maps = {}
for col in cat_cols:
    X_enc[col] = pd.Categorical(X[col]).codes
    enc_maps[col] = dict(enumerate(pd.Categorical(X[col]).categories))
X_enc = X_enc.fillna(X_enc.median())

# SMOTE with 1:3 ratio (CKD:Healthy) - balanced for model learning
# Target: ~508 healthy samples (373 synthetic = 73% synthetic)
target_minority = int(sum(y==1) / 3)  # CKD count / 3
print(f"Applying SMOTE with 1:3 ratio (recommended for imbalanced medical data)...")
print(f"Target minority class size: {target_minority}")
smote = SMOTE(sampling_strategy={0: target_minority}, random_state=42, k_neighbors=5)
X_bal, y_bal = smote.fit_resample(X_enc, y)

print(f"After: Healthy={sum(y_bal==0)}, CKD={sum(y_bal==1)}, Ratio={sum(y_bal==1)/sum(y_bal==0):.2f}:1")
synthetic_added = sum(y_bal==0) - sum(y==0)
print(f"Synthetic samples added: {synthetic_added}")
print(f"Synthetic percentage: {synthetic_added/sum(y_bal==0)*100:.1f}% of minority class")

# Create balanced df
df_bal = pd.DataFrame(X_bal, columns=X_enc.columns)
df_bal[target] = y_bal
for col in cat_cols:
    df_bal[col] = df_bal[col].astype(int).map(enc_maps[col])

# Add IDs
orig_ids = df['PatientID'].tolist()
synth_ids = [f'SYNTH_{i:05d}' for i in range(len(y_bal)-len(y))]
df_bal['PatientID'] = orig_ids + synth_ids
cols = ['PatientID'] + [c for c in df_bal.columns if c not in ['PatientID', target]] + [target]
df_bal = df_bal[cols]

df_bal.to_csv(OUTPUT_PATH, index=False)
print(f"Saved: {OUTPUT_PATH}")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Bar comparison
ax = axes[0, 0]
x = np.arange(2)
w = 0.35
before = [sum(y==0), sum(y==1)]
after = [sum(y_bal==0), sum(y_bal==1)]
ax.bar(x - w/2, before, w, label='Before', color='#e74c3c', edgecolor='black')
ax.bar(x + w/2, after, w, label='After', color='#27ae60', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(['Healthy', 'CKD'])
ax.set_title('Class Distribution: Before vs After SMOTE', fontweight='bold')
ax.legend()
for i, (b, a) in enumerate(zip(before, after)):
    ax.text(i - w/2, b + 30, str(b), ha='center', fontweight='bold')
    ax.text(i + w/2, a + 30, str(a), ha='center', fontweight='bold')

# Pie before
ax = axes[0, 1]
ax.pie(before, labels=['Healthy', 'CKD'], autopct='%1.1f%%', colors=['#27ae60', '#e74c3c'], explode=(0.05, 0))
ax.set_title('Before SMOTE\n(11.29:1 Imbalance)', fontweight='bold')

# Pie after
ax = axes[1, 0]
ax.pie(after, labels=['Healthy', 'CKD'], autopct='%1.1f%%', colors=['#27ae60', '#e74c3c'], explode=(0.05, 0))
ratio_text = f"{sum(y_bal==1)/sum(y_bal==0):.1f}:1"
ax.set_title(f'After SMOTE\n({ratio_text} Recommended Ratio)', fontweight='bold')

# PCA
ax = axes[1, 1]
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_enc)
X_bal_pca = pca.transform(X_bal)

# Original points
ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='#27ae60', label='Healthy (orig)', alpha=0.6, s=40)
ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='#e74c3c', label='CKD (orig)', alpha=0.6, s=40)
# Synthetic points
synth_mask = np.arange(len(y_bal)) >= len(y)
ax.scatter(X_bal_pca[synth_mask, 0], X_bal_pca[synth_mask, 1], c='#9b59b6', marker='^', 
           label='Healthy (synthetic)', alpha=0.5, s=30)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA: Original + Synthetic Samples', fontweight='bold')
ax.legend()

plt.suptitle('SMOTE Oversampling - Chronic Kidney Disease (Recommended 1:3 Ratio)\n25% Minority Class for Better Model Learning', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=150, bbox_inches='tight')
print(f"Saved: {FIG_PATH}")
print("\nDone!")
