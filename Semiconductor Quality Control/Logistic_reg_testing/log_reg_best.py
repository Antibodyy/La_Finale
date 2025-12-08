import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
FILE_PATH = '/Users/aryankulkarni/Documents/UC Berkeley Coursework/SDSE/La_Finale/Semiconductor Quality Control/semiconductor_quality_control.csv'
RANDOM_STATE = 1
FOLDS = 4

print(f">>> Loading Data from {FILE_PATH}...")
df = pd.read_csv(FILE_PATH)

# ==========================================
# 2. FEATURE CORRELATION ANALYSIS
# ==========================================
print("\n>>> Analyzing Feature Correlations...")

# Prepare data for correlation (Drop non-predictive IDs)
corr_drop_cols = ['Process_ID', 'Timestamp', 'Wafer_ID', 'Join_Status']
df_corr = df.drop(columns=corr_drop_cols)

# One-Hot Encode 'Tool_Type' for correlation check
df_corr = pd.get_dummies(df_corr, columns=['Tool_Type'], drop_first=True)

# Calculate correlations with Target
correlations = df_corr.corr()['Defect'].drop('Defect')

# Sort by magnitude (absolute value) and take top 15
top_features = correlations.abs().sort_values(ascending=False).head(15)
top_features_signed = correlations[top_features.index]

# --- PLOT 1: Feature Correlations (Matplotlib only) ---
plt.figure(figsize=(10, 8))
y_pos = np.arange(len(top_features_signed))
colors = ['red' if x > 0 else 'blue' for x in top_features_signed.values]

plt.barh(y_pos, top_features_signed.values, align='center', color=colors, alpha=0.7)
plt.yticks(y_pos, top_features_signed.index)
plt.xlabel('Pearson Correlation Coefficient')
plt.title('Top 15 Features Correlated with Defects')
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==========================================
# 3. DATA PREP FOR MODELING
# ==========================================
# Separate X and y
drop_cols_model = ['Process_ID', 'Timestamp', 'Wafer_ID', 'Join_Status', 'Defect']
X = df.drop(columns=drop_cols_model)
y = df['Defect']

# One-Hot Encode X
X = pd.get_dummies(X, columns=['Tool_Type'], drop_first=True)

# Identify numerical columns for Scaling in pipeline
numerical_cols = ['Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
                  'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
                  'Vibration_Level', 'UV_Exposure_Intensity', 'Particle_Count']

# ==========================================
# 4. GRID SEARCH (Regularization)
# ==========================================
penalties = ['l1', 'l2']
C_range = np.logspace(-4, 4, 10) 
f2_scorer = make_scorer(fbeta_score, beta=2)

print(f"\n>>> Starting Grid Search with {FOLDS}-Fold Cross-Validation...")
results = []

for penalty in penalties:
    for C in C_range:
        # Pipeline: Scale Numerical -> LogReg
        preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), numerical_cols)],
            remainder='passthrough'
        )
        
        pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(
                penalty=penalty, 
                C=C, 
                solver='liblinear', 
                class_weight='balanced', 
                max_iter=5000, 
                random_state=RANDOM_STATE
            ))
        ])
        
        # Cross Validate (Optimized to run once for both metrics)
        cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {'f2': f2_scorer, 'recall': 'recall'}
        
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
        
        mean_f2 = np.mean(scores['test_f2'])
        mean_rec = np.mean(scores['test_recall'])
        
        results.append({
            'Penalty': penalty,
            'C': C,
            'F2': mean_f2,
            'Recall': mean_rec
        })
        print(f"Penalty: {penalty} | C: {C:.5f} | F2: {mean_f2:.4f} | Recall: {mean_rec:.4f}")

# ==========================================
# 5. REGULARIZATION PLOTS
# ==========================================
results_df = pd.DataFrame(results)
df_l1 = results_df[results_df['Penalty'] == 'l1']
df_l2 = results_df[results_df['Penalty'] == 'l2']

# --- PLOT 2: Recall & F2 vs Regularization ---
plt.figure(figsize=(12, 7))

# Plot Recall
plt.plot(df_l1['C'], df_l1['Recall'], label='Recall (L1/Lasso)', 
         color='tab:blue', marker='o', linestyle='-')
plt.plot(df_l2['C'], df_l2['Recall'], label='Recall (L2/Ridge)', 
         color='tab:blue', marker='o', linestyle='--', alpha=0.6)

# Plot F2
plt.plot(df_l1['C'], df_l1['F2'], label='F2 Score (L1/Lasso)', 
         color='tab:red', marker='s', linestyle='-')
plt.plot(df_l2['C'], df_l2['F2'], label='F2 Score (L2/Ridge)', 
         color='tab:red', marker='s', linestyle='--', alpha=0.6)

plt.xscale('log')
plt.xlabel('Inverse Regularization Strength (C)')
plt.ylabel('Score')
plt.title('Impact of Regularization on Defect Detection (Recall & F2)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.4)

# Annotate Best F2
best_idx = results_df['F2'].idxmax()
best_f2 = results_df.loc[best_idx, 'F2']
best_c = results_df.loc[best_idx, 'C']
plt.annotate(f'Max F2: {best_f2:.3f}', xy=(best_c, best_f2), 
             xytext=(best_c, best_f2 + 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center')

plt.tight_layout()
plt.show()