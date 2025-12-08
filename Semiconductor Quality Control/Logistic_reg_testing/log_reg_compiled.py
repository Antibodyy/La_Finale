import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, fbeta_score, recall_score
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

# Drop identifiers and leakage columns
drop_cols = ['Process_ID', 'Timestamp', 'Wafer_ID', 'Join_Status', 'Defect']
X = df.drop(columns=drop_cols)
y = df['Defect']

# One-Hot Encoding for Categorical 'Tool_Type'
X = pd.get_dummies(X, columns=['Tool_Type'], drop_first=True)

# Identify numerical columns for Scaling
numerical_cols = ['Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
                  'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
                  'Vibration_Level', 'UV_Exposure_Intensity', 'Particle_Count']

# ==========================================
# 2. GRID SEARCH SETUP
# ==========================================
# We define the grid of hyperparameters to test
penalties = ['l1', 'l2']
C_range = np.logspace(-4, 4, 10) # 10 values from 0.0001 to 10000

# Define Custom Scorer for F2 (Recall-Weighted)
f2_scorer = make_scorer(fbeta_score, beta=2)

print(f"\n>>> Starting Grid Search with {FOLDS}-Fold Cross-Validation...")
print(f"{'Penalty':<10} {'C_Value':<12} {'Mean_F2':<10} {'Mean_Recall':<12}")
print("-" * 50)

results = []

# ==========================================
# 3. THE LOOP
# ==========================================
for penalty in penalties:
    for C in C_range:
        # A. Create Pipeline
        # Step 1: Scale ONLY numerical columns (inside the fold)
        # Step 2: Logistic Regression with specific Penalty/C
        
        preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), numerical_cols)],
            remainder='passthrough'
        )
        
        pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(
                penalty=penalty, 
                C=C, 
                solver='liblinear', # liblinear supports both l1 and l2
                class_weight='balanced', 
                max_iter=5000, 
                random_state=RANDOM_STATE
            ))
        ])
        
        # B. Run Cross-Validation
        cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        # We compute scores for both metrics
        f2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=f2_scorer)
        rec_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='recall')
        
        mean_f2 = np.mean(f2_scores)
        mean_rec = np.mean(rec_scores)
        
        # C. Store & Print
        results.append({
            'Penalty': penalty,
            'C': C,
            'F2': mean_f2,
            'Recall': mean_rec
        })
        
        print(f"{penalty:<10} {C:<12.5f} {mean_f2:<10.4f} {mean_rec:<12.4f}")

# ==========================================
# 4. FINAL RESULTS
# ==========================================
results_df = pd.DataFrame(results)

# Find the Best Configurations
best_f2_idx = results_df['F2'].idxmax()
best_rec_idx = results_df['Recall'].idxmax()

print("\n" + "="*50)
print(" WINNING CONFIGURATIONS ")
print("="*50)

print(f"Best F2 Score:   {results_df.loc[best_f2_idx, 'F2']:.4f}")
print(f"  -> Params:     Penalty={results_df.loc[best_f2_idx, 'Penalty']}, C={results_df.loc[best_f2_idx, 'C']:.5f}")

print(f"\nBest Recall:     {results_df.loc[best_rec_idx, 'Recall']:.4f}")
print(f"  -> Params:     Penalty={results_df.loc[best_rec_idx, 'Penalty']}, C={results_df.loc[best_rec_idx, 'C']:.5f}")

# Optional: Save to CSV for your report
# results_df.to_csv('log_reg_grid_search_results.csv', index=False)