import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import recall_score, fbeta_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

FILE_PATH = '/Users/aryankulkarni/Documents/UC Berkeley Coursework/SDSE/La_Finale/Semiconductor Quality Control/semiconductor_quality_control.csv'
RANDOM_STATE = 1
FOLDS = 4

def load_data(filepath):
    print(f">>> Loading Data from {filepath}...")
    df = pd.read_csv(filepath)
    drop_cols = ['Process_ID', 'Timestamp', 'Wafer_ID', 'Join_Status', 'Defect']
    X = df.drop(columns=drop_cols)
    y = df['Defect']
    X = pd.get_dummies(X, columns=['Tool_Type'], drop_first=True)
    
    numerical_cols = ['Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
                      'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
                      'Vibration_Level', 'UV_Exposure_Intensity', 'Particle_Count']
    
    return X, y, numerical_cols


def get_best_threshold(y_true, y_probs, metric='f2'):
    best_score = 0
    best_thresh = 0.5
    
    for t in np.arange(0.1, 0.96, 0.05):
        preds = (y_probs >= t).astype(int)
        
        if metric == 'f2': 
            score = fbeta_score(y_true, preds, beta=3)
        elif metric == 'recall':
            score = recall_score(y_true, preds)
        
        if score > best_score:
            best_score = score
            best_thresh = t
            
    return best_thresh, best_score

def run_cv_pipeline():
    X, y, num_cols = load_data(FILE_PATH)
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    
    print(f"\n>>> Running {FOLDS}-Fold Cross-Validation on ALL Variants...")
    
    # ---------------------------------------------------------
    # PART A: BASE MODEL (For Vanilla & Caramel)
    # ---------------------------------------------------------
    print("\n--- A. Base Logistic Regression (Standard Features) ---")
    
    # Pipeline: Scale Num -> Model
    preprocessor_base = ColumnTransformer(
        transformers=[('num', StandardScaler(), num_cols)],
        remainder='passthrough'
    )
    
    pipe_base = Pipeline([
        ('prep', preprocessor_base),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE))
    ])
    
    # Generate Cross-Validated Probabilities
    # This gives us a prediction for every sample when it was in the TEST fold
    y_probs_base = cross_val_predict(pipe_base, X, y, cv=cv, method='predict_proba')[:, 1]
    
    # --- VARIANT 1: VANILLA (F2 Optimized) ---
    thresh_vanilla, score_vanilla = get_best_threshold(y, y_probs_base, metric='f2')
    preds_vanilla = (y_probs_base >= thresh_vanilla).astype(int)
    results['Vanilla (Base + F2 Opt)'] = confusion_matrix(y, preds_vanilla)
    
    print(f"[Vanilla] Best Threshold: {thresh_vanilla:.2f}")
    print(f"          CV F2-Score:    {score_vanilla:.4f}")
    print(f"          CV Recall:      {recall_score(y, preds_vanilla):.4f}")
    
    # --- VARIANT 2: CARAMEL (Recall Optimized) ---
    thresh_caramel, score_caramel = get_best_threshold(y, y_probs_base, metric='recall')
    preds_caramel = (y_probs_base >= thresh_caramel).astype(int)
    results['Caramel (Base + Recall Opt)'] = confusion_matrix(y, preds_caramel)
    
    print(f"[Caramel] Best Threshold: {thresh_caramel:.2f}")
    print(f"          CV Recall:      {score_caramel:.4f}")
    print(f"          CV Precision:   {precision_score(y, preds_caramel):.4f}")

    # ---------------------------------------------------------
    # PART B: POLY MODEL (For Poly Variant)
    # ---------------------------------------------------------
    print("\n--- B. Polynomial Logistic Regression (Deg 2 Features) ---")
    
    # Pipeline: Poly Num -> Scale -> Model
    # Note: PolynomialFeatures MUST be inside the pipeline to avoid leakage!
    poly_pipeline_steps = [
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler())
    ]
    
    preprocessor_poly = ColumnTransformer(
        transformers=[('num_poly', Pipeline(poly_pipeline_steps), num_cols)],
        remainder='passthrough'
    )
    
    pipe_poly = Pipeline([
        ('prep', preprocessor_poly),
        ('clf', LogisticRegression(class_weight='balanced', C=0.1, max_iter=3000, random_state=RANDOM_STATE))
    ])
    
    # Generate Cross-Validated Probabilities
    y_probs_poly = cross_val_predict(pipe_poly, X, y, cv=cv, method='predict_proba')[:, 1]
    
    # --- VARIANT 3: POLY (F2 Optimized) ---
    thresh_poly, score_poly = get_best_threshold(y, y_probs_poly, metric='f2')
    preds_poly = (y_probs_poly >= thresh_poly).astype(int)
    results['Poly (Deg2 + F2 Opt)'] = confusion_matrix(y, preds_poly)
    
    print(f"[Poly]    Best Threshold: {thresh_poly:.2f}")
    print(f"          CV F2-Score:    {score_poly:.4f}")
    print(f"          CV Recall:      {recall_score(y, preds_poly):.4f}")

    # ==========================================
    # 5. VISUALIZATION (No Seaborn)
    # ==========================================
    print("\n>>> Plotting Consolidated Results...")
    # Create subplots: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (name, cm) in enumerate(results.items()):
        # Use Sklearn's built-in plotter which uses Matplotlib
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Safe', 'Defect'])
        disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
        
        axes[i].set_title(name, fontsize=14, fontweight='bold')
        axes[i].grid(False) # Turn off grid for cleaner look
    
    plt.suptitle(f"Confusion Matrices ({FOLDS}-Fold Cross Validation)", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_cv_pipeline()