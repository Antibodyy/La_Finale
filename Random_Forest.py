import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv('semiconductor_quality_control.csv')

# -------------------------------
# Preprocessing
# -------------------------------
X = df.drop(['Defect', 'Process_ID', 'Wafer_ID', 'Timestamp', 'Join_Status'], axis=1)
X = pd.get_dummies(X, columns=['Tool_Type'], drop_first=True)
y = df['Defect']

# -------------------------------
# Split minority and majority
# -------------------------------
minority = df[df['Defect'] == 1]
majority = df[df['Defect'] == 0]
n_chunks = 6
majority_chunks = np.array_split(majority, n_chunks)

# -------------------------------
# Global test set
# -------------------------------
X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=67
)

# -------------------------------
# Define class weights and thresholds to try
# -------------------------------
class_weights_list = [
    {0: 1, 1: 1.25}, {0: 1, 1: 1.5}, {0: 1, 1: 1.75},
    {0: 1, 1: 2}, {0: 1, 1: 2.5}, {0: 1, 1: 3},
    {0: 1, 1: 4}, {0: 1, 1: 5}, {0: 1, 1: 6}, "balanced"
]
thresholds = np.arange(0.3, 0.9, 0.01)

# -------------------------------
# Train and evaluate ensemble for each class weight
# -------------------------------
results = []

for cw in class_weights_list:
    print(f"\nTesting Class Weight: {cw}")
    rf_models = []

    # Train ensemble models
    for i, maj_chunk in enumerate(majority_chunks):
        print(f"Training Model {i+1}/{n_chunks} with class_weight={cw}")
        balanced_train = pd.concat([maj_chunk, minority])

        X_train_bal = balanced_train.drop('Defect', axis=1)
        X_train_bal = pd.get_dummies(X_train_bal, columns=['Tool_Type'], drop_first=True)
        X_train_bal = X_train_bal.reindex(columns=X_train_global.columns, fill_value=0)
        y_train_bal = balanced_train['Defect']

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=cw,
            random_state=67
        )
        rf.fit(X_train_bal, y_train_bal)
        rf_models.append(rf)

    # Ensemble prediction
    probas = np.zeros((X_test_global.shape[0], len(rf_models)))
    for i, rf in enumerate(rf_models):
        probas[:, i] = rf.predict_proba(X_test_global)[:, 1]
    avg_probas = probas.mean(axis=1)

    # Threshold optimization
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        y_pred_thresh = (avg_probas >= t).astype(int)
        f1 = f1_score(y_test_global, y_pred_thresh, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # Final prediction and evaluation
    y_pred_final = (avg_probas >= best_thresh).astype(int)
    acc = accuracy_score(y_test_global, y_pred_final)
    prec = precision_score(y_test_global, y_pred_final, zero_division=0)
    rec = recall_score(y_test_global, y_pred_final, zero_division=0)
    f1 = f1_score(y_test_global, y_pred_final, zero_division=0)

    # Store results
    results.append({
        "class_weight": cw,
        "threshold": best_thresh,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

    # Print current configuration
    print(f"Best Threshold: {best_thresh:.3f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

# -------------------------------
# Print summary of all results
# -------------------------------
print("\n==================================================")
print("Summary of All Class Weight Experiments:")
for r in results:
    print(f"class_weight: {r['class_weight']}, threshold: {r['threshold']:.3f}, "
          f"accuracy: {r['accuracy']:.4f}, precision: {r['precision']:.4f}, "
          f"recall: {r['recall']:.4f}, f1: {r['f1']:.4f}")
