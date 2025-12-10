import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv('semiconductor_quality_control.csv')

# -------------------------------
# Preprocessing
# -------------------------------
X = df.drop(['Defect', 'Process_ID', 'Wafer_ID', 'Timestamp', 'Join_Status'], axis=1)
le = LabelEncoder()
X['Tool_Type'] = le.fit_transform(X['Tool_Type'])
y = df['Defect']

# -------------------------------
# Global train/test split
# -------------------------------
X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=67
)

# -------------------------------
# Split minority and majority for Step 1 ensemble
# -------------------------------
n_chunks = 3
train_df = pd.concat([X_train_global, y_train_global], axis=1)
minority_train = train_df[train_df['Defect'] == 1]
majority_train = train_df[train_df['Defect'] == 0]
majority_chunks = np.array_split(majority_train, n_chunks)

# -------------------------------
# Step 1 configuration
# -------------------------------
class_weights_list = [
    {0:1, 1:1.2},
    {0:1, 1:1.3},
    {0:1, 1:1.4},
    {0:1, 1:1.5}
]
step1_thresholds = np.arange(0.33, 0.42, 0.01)

best_overall_f1 = 0
best_config = None
best_step1_thresh = 0
best_cm_step1 = None
best_cm_step2 = None

for cw in class_weights_list:
    print(f"\n=== Testing Class Weight: {cw} ===")
    rf_models = []

    # Step 1: Train ensemble on balanced chunks
    for maj_chunk in majority_chunks:
        balanced_train = pd.concat([maj_chunk, minority_train])
        X_train_bal = balanced_train.drop('Defect', axis=1)
        X_train_bal = pd.get_dummies(X_train_bal, columns=['Tool_Type'], drop_first=True)
        X_train_bal = X_train_bal.reindex(columns=X_train_global.columns, fill_value=0)
        y_train_bal = balanced_train['Defect']

        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=2,
            class_weight=cw,
            random_state=67,
            n_jobs=-1
        )
        rf.fit(X_train_bal, y_train_bal)
        rf_models.append(rf)

    # Step 1: Ensemble prediction on test set
    probas = np.zeros((X_test_global.shape[0], len(rf_models)))
    for i, rf in enumerate(rf_models):
        probas[:, i] = rf.predict_proba(X_test_global)[:,1]
    avg_probas = probas.mean(axis=1)

    # Optimize Step 1 threshold
    step1_best_f1 = 0
    step1_best_t = 0.5
    for t in step1_thresholds:
        y_pred_step1 = (avg_probas >= t).astype(int)
        f1 = f1_score(y_test_global, y_pred_step1, zero_division=0)
        if f1 > step1_best_f1:
            step1_best_f1 = f1
            step1_best_t = t

    # Step 1 final prediction
    y_pred_step1 = (avg_probas >= step1_best_t).astype(int)
    cm_step1 = confusion_matrix(y_test_global, y_pred_step1)
    print(f"Step 1 Best Threshold: {step1_best_t:.3f}, F1: {step1_best_f1:.4f}")
    print(f"Step 1 Confusion Matrix:\n{cm_step1}")

    # -------------------------------
    # Step 2: Refined RF on full training set
    # -------------------------------
    rf_step2 = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=2,
        class_weight=cw,
        random_state=67,
        n_jobs=-1
    )
    rf_step2.fit(X_train_global, y_train_global)

    # Apply Step 2 only on Step 1 predicted normals
    normal_idx = np.where(y_pred_step1 == 0)[0]
    if len(normal_idx) > 0:
        X_test_normal = X_test_global.iloc[normal_idx]
        y_pred_step2_normal = rf_step2.predict(X_test_normal)
        y_pred_joint = y_pred_step1.copy()
        y_pred_joint[normal_idx] = y_pred_step2_normal
    else:
        y_pred_joint = y_pred_step1.copy()

    cm_step2 = confusion_matrix(y_test_global, y_pred_joint)
    print("Step 2 Confusion Matrix (joint prediction):")
    print(cm_step2)

    f1_joint = f1_score(y_test_global, y_pred_joint, zero_division=0)
    if f1_joint > best_overall_f1:
        best_overall_f1 = f1_joint
        best_config = cw
        best_step1_thresh = step1_best_t
        best_cm_step1 = cm_step1
        best_cm_step2 = cm_step2

# -------------------------------
# Print summary
# -------------------------------
print("\n==================================================")
print("Best Overall Joint Modelling Configuration (by F1):")
print(f"Class Weight: {best_config}")
print(f"Step 1 Threshold: {best_step1_thresh:.3f}")
print(f"Best F1: {best_overall_f1:.4f}")
print("Step 1 Confusion Matrix:")
print(best_cm_step1)
print("Step 2 Confusion Matrix:")
print(best_cm_step2)
