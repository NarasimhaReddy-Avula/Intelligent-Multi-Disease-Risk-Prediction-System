"""Chronic Kidney Disease EDA - Generate All Figures"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# Paths
df = pd.read_csv('../../datasets/chronic_kidney_disease.csv')
FIGS = './figures/kidney'

target = 'Diagnosis'
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['PatientID', target]]
X = df[numeric_cols]
y = df[target]

print(f"Dataset: {len(df)} samples, {len(numeric_cols)} numeric features")
print(f"CKD: {y.sum()} | Healthy: {len(y)-y.sum()} | Ratio: {y.sum()/(len(y)-y.sum()):.2f}:1")

# 1. Target Distribution
fig, ax = plt.subplots(figsize=(10, 6))
counts = y.value_counts().sort_index()
bars = ax.bar(['Healthy (0)', 'CKD (1)'], counts.values, color=['#27ae60', '#e74c3c'], edgecolor='black')
for bar, c in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, f'{c}\n({c/len(y)*100:.1f}%)', ha='center', fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Target Distribution - Chronic Kidney Disease\n(CKD is Majority Class)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGS}/01_target_distribution.png', dpi=150)
plt.close()
print("1. Target distribution saved")

# 2. Random Forest Feature Importance
print("2. Computing RF importance...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf.fit(X.fillna(X.median()), y)
imp = pd.DataFrame({'Feature': numeric_cols, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 14))
ax.barh(imp['Feature'], imp['Importance'], color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(imp)))[::-1])
ax.axvline(imp['Importance'].mean(), color='red', linestyle='--', label=f"Mean: {imp['Importance'].mean():.4f}")
ax.set_xlabel('Importance')
ax.set_title('Feature Importance (Random Forest)', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGS}/02_feature_importance_rf.png', dpi=150)
plt.close()
print("   RF importance saved")

# 3. Correlation with Target
corr = X.corrwith(y).sort_values()
fig, ax = plt.subplots(figsize=(10, 14))
colors = ['#e74c3c' if c > 0 else '#3498db' for c in corr.values]
ax.barh(corr.index, corr.values, color=colors)
ax.axvline(0, color='black')
ax.set_xlabel('Correlation with CKD')
ax.set_title('Feature Correlation with Target', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGS}/03_correlation_with_target.png', dpi=150)
plt.close()
print("3. Correlation saved")

# 4. Mutual Information
print("4. Computing MI...")
mi = mutual_info_classif(X.fillna(X.median()), y, random_state=42)
mi_df = pd.DataFrame({'Feature': numeric_cols, 'MI': mi}).sort_values('MI', ascending=True)

fig, ax = plt.subplots(figsize=(10, 14))
ax.barh(mi_df['Feature'], mi_df['MI'], color=plt.cm.plasma(np.linspace(0.1, 0.9, len(mi_df)))[::-1])
ax.set_xlabel('Mutual Information')
ax.set_title('Feature Importance (Mutual Information)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGS}/04_feature_importance_mi.png', dpi=150)
plt.close()
print("   MI saved")

# 5. Top 10 Features - Boxplots
top10 = imp.tail(10)['Feature'].tolist()
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, feat in zip(axes.flatten(), top10):
    df.boxplot(column=feat, by=target, ax=ax)
    ax.set_title(feat)
    ax.set_xlabel('')
plt.suptitle('Top 10 Features by CKD Status', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIGS}/05_top10_boxplots.png', dpi=150)
plt.close()
print("5. Top 10 boxplots saved")

# 6. Kidney Biomarkers
kidney = ['SerumCreatinine', 'GFR', 'BUNLevels', 'ProteinInUrine']
kidney = [k for k in kidney if k in df.columns]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, feat in zip(axes.flatten(), kidney):
    for diag, color, lbl in [(0, '#27ae60', 'Healthy'), (1, '#e74c3c', 'CKD')]:
        data = df[df[target]==diag][feat].dropna()
        ax.hist(data, bins=30, alpha=0.6, color=color, label=lbl, density=True)
    ax.set_title(feat, fontweight='bold')
    ax.legend()
plt.suptitle('Kidney Biomarkers Distribution', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIGS}/06_kidney_biomarkers.png', dpi=150)
plt.close()
print("6. Kidney biomarkers saved")

# 7. GFR vs Creatinine
fig, ax = plt.subplots(figsize=(10, 8))
colors = df[target].map({0: '#27ae60', 1: '#e74c3c'})
ax.scatter(df['SerumCreatinine'], df['GFR'], c=colors, alpha=0.5, edgecolors='black', linewidth=0.3)
ax.axhline(60, color='orange', linestyle='--', label='GFR 60 (Stage 3)')
ax.axhline(30, color='red', linestyle='--', label='GFR 30 (Stage 4)')
ax.set_xlabel('Serum Creatinine')
ax.set_ylabel('GFR')
ax.set_title('GFR vs Serum Creatinine', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGS}/07_gfr_vs_creatinine.png', dpi=150)
plt.close()
print("7. GFR vs Creatinine saved")

# 8. Correlation Heatmap
top15 = imp.tail(15)['Feature'].tolist()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df[top15].corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Correlation Heatmap (Top 15 Features)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGS}/08_correlation_heatmap.png', dpi=150)
plt.close()
print("8. Correlation heatmap saved")

# 9. Age Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for diag, color, lbl in [(0, '#27ae60', 'Healthy'), (1, '#e74c3c', 'CKD')]:
    axes[0].hist(df[df[target]==diag]['Age'], bins=30, alpha=0.6, color=color, label=lbl, density=True)
axes[0].set_xlabel('Age')
axes[0].set_title('Age Distribution', fontweight='bold')
axes[0].legend()

df['AgeGrp'] = pd.cut(df['Age'], bins=[0,40,50,60,70,100], labels=['<40','40-50','50-60','60-70','>70'])
age_ckd = df.groupby('AgeGrp')[target].mean()*100
axes[1].bar(age_ckd.index.astype(str), age_ckd.values, color=plt.cm.Reds(np.linspace(0.3,0.9,5)))
axes[1].set_ylabel('CKD Prevalence (%)')
axes[1].set_title('CKD Prevalence by Age Group', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGS}/09_age_analysis.png', dpi=150)
plt.close()
print("9. Age analysis saved")

# 10. Blood Pressure
fig, ax = plt.subplots(figsize=(10, 8))
colors = df[target].map({0: '#27ae60', 1: '#e74c3c'})
ax.scatter(df['SystolicBP'], df['DiastolicBP'], c=colors, alpha=0.5)
ax.axhline(90, color='red', linestyle='--', alpha=0.7)
ax.axvline(140, color='red', linestyle='--', alpha=0.7, label='Hypertension threshold')
ax.set_xlabel('Systolic BP')
ax.set_ylabel('Diastolic BP')
ax.set_title('Blood Pressure Distribution', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGS}/10_blood_pressure.png', dpi=150)
plt.close()
print("10. Blood pressure saved")

# 11. Statistical Summary Table
top10_stats = []
for feat in top10:
    h = df[df[target]==0][feat].dropna()
    c = df[df[target]==1][feat].dropna()
    _, p = stats.mannwhitneyu(h, c)
    top10_stats.append({'Feature': feat, 'Healthy_Mean': h.mean(), 'CKD_Mean': c.mean(), 'P-value': p})
stats_df = pd.DataFrame(top10_stats)
stats_df.to_csv(f'{FIGS}/statistical_summary.csv', index=False)
print("11. Stats saved to CSV")

print("\n✅ All 11 figures generated in:", FIGS)
