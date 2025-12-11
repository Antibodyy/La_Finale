import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv('../../semiconductor_quality_control.csv')

# -------------------------------
# Preprocessing
# -------------------------------
X = df.drop(['Defect', 'Process_ID', 'Wafer_ID', 'Timestamp', 'Join_Status'], axis=1)
le = LabelEncoder()
X['Tool_Type'] = le.fit_transform(X['Tool_Type'])
y = df['Defect']

# -------------------------------
# Global test set
# -------------------------------
X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=67
)

train_df = pd.concat([X_train_global, y_train_global], axis=1)
minority_train = train_df[train_df['Defect'] == 1]
majority_train = train_df[train_df['Defect'] == 0]

# -------------------------------
# Initialize trackers
# -------------------------------
all_results = []

best_cm_f1 = 0
best_cm = None
best_cm_model_info = None
best_cm_threshold = None

best_f2 = 0
best_f2_cm = None
best_f2_config = None
best_f2_threshold = None
best_f2_f1 = 0
best_f2_recall = 0

best_recall = 0
best_recall_cm = None
best_recall_config = None
best_recall_threshold = None

# -------------------------------
# Run experiments
# -------------------------------
for n_chunks in [3, 4]:
    majority_chunks = np.array_split(majority_train, n_chunks)

    class_weights_list = [{0:1, 1:w} for w in range(1,6)]
    thresholds = np.arange(0.4, 0.8, 0.01)

    for cw in class_weights_list:
        rf_models = []

        # Train ensemble
        for maj_chunk in majority_chunks:
            balanced_train = pd.concat([maj_chunk, minority_train])
            X_train_bal = balanced_train.drop('Defect', axis=1)
            X_train_bal = pd.get_dummies(X_train_bal, columns=['Tool_Type'], drop_first=True)
            X_train_bal = X_train_bal.reindex(columns=X_train_global.columns, fill_value=0)
            y_train_bal = balanced_train['Defect']

            rf = RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=2,
                class_weight=cw,
                random_state=67,
                n_jobs=-1
            )
            rf.fit(X_train_bal, y_train_bal)
            rf_models.append(rf)

        # Ensemble prediction
        probas = np.zeros((X_test_global.shape[0], len(rf_models)))
        for i, rf in enumerate(rf_models):
            probas[:, i] = rf.predict_proba(X_test_global)[:, 1]
        avg_probas = probas.mean(axis=1)

        # Threshold optimization
        best_f1_local = 0
        best_thresh = 0.5
        for t in thresholds:
            y_pred_thresh = (avg_probas >= t).astype(int)
            f1_local = f1_score(y_test_global, y_pred_thresh, zero_division=0)
            if f1_local > best_f1_local:
                best_f1_local = f1_local
                best_thresh = t

        # Final evaluation
        y_pred_final = (avg_probas >= best_thresh).astype(int)
        acc = accuracy_score(y_test_global, y_pred_final)
        prec = precision_score(y_test_global, y_pred_final, zero_division=0)
        rec = recall_score(y_test_global, y_pred_final, zero_division=0)
        f1 = f1_score(y_test_global, y_pred_final, zero_division=0)
        f2 = fbeta_score(y_test_global, y_pred_final, beta=2, zero_division=0)
        cm = confusion_matrix(y_test_global, y_pred_final)

        # Track best F1
        if f1 > best_cm_f1:
            best_cm_f1 = f1
            best_cm = cm
            best_cm_threshold = best_thresh
            best_cm_model_info = cw

        # Track best F2
        if f2 > best_f2:
            best_f2 = f2
            best_f2_cm = cm
            best_f2_config = cw
            best_f2_threshold = best_thresh
            best_f2_f1 = f1
            best_f2_recall = rec

        # Track best recall
        if rec > best_recall:
            best_recall = rec
            best_recall_cm = cm
            best_recall_config = cw
            best_recall_threshold = best_thresh

        # Store result
        all_results.append({
            "n_chunks": n_chunks,
            "class_weight": cw[1],
            "threshold": best_thresh,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f2": f2
        })

# -------------------------------
# Convert results to DataFrame
# -------------------------------
df_results = pd.DataFrame(all_results)

# -------------------------------
# Print summary
# -------------------------------
print("\n==================================================")
print("Best Configuration by Metric")
print(f"Highest F1 Score: class_weight={best_cm_model_info}, threshold={best_cm_threshold}, F1={best_cm_f1}")
print(f"Highest F2 Score: class_weight={best_f2_config}, threshold={best_f2_threshold}, F2={best_f2}, Recall={best_f2_recall}")
print(f"Highest Recall: class_weight={best_recall_config}, threshold={best_recall_threshold}, Recall={best_recall}")

# -------------------------------
# Confusion Matrices
# -------------------------------
disp_f1 = ConfusionMatrixDisplay(best_cm, display_labels=['Pass', 'Defect'])
disp_f1.plot(cmap='Blues', values_format='d')
plt.title('Random Forest – Best F1')
plt.grid(False)
plt.show()

disp_f2 = ConfusionMatrixDisplay(best_f2_cm, display_labels=['Pass', 'Defect'])
disp_f2.plot(cmap='Blues', values_format='d')
plt.title('Random Forest – Best F2')
plt.grid(False)
plt.show()

# -------------------------------
# Plot Recall and F2 per n_chunks
# -------------------------------
for metric in ['recall', 'f2']:
    plt.figure(figsize=(8,5))
    for n_chunks in df_results['n_chunks'].unique():
        subset = df_results[df_results['n_chunks'] == n_chunks]
        plt.plot(subset['class_weight'], subset[metric], marker='o', label=f'{n_chunks} chunks')
    plt.xlabel('Minority Class Weight')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} vs Minority Class Weight per n_chunks')
    plt.legend()
    plt.grid(True)
    plt.show()
