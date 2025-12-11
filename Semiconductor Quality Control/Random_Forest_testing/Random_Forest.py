import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import fbeta_score

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

# -------------------------------
# Split minority and majority
# -------------------------------

train_df = pd.concat([X_train_global, y_train_global], axis=1)
minority_train = train_df[train_df['Defect'] == 1]
majority_train = train_df[train_df['Defect'] == 0]

for n_chunks in [3, 4]:


    majority_chunks = np.array_split(majority_train, n_chunks)

    # -------------------------------
    # Define class weights and thresholds to try
    # -------------------------------
    class_weights_list = [
        {0:1, 1:1},
        {0:1, 1:2},
        {0:1, 1:3},
        {0:1, 1:4},
        {0:1, 1:5}
    ]

    thresholds = np.arange(0.4, 0.8, 0.01)

    # -------------------------------
    # Train and evaluate ensemble for each class weight
    # -------------------------------
    results = []

    best_cm = None
    best_cm_model_info = None
    best_cm_f1 = 0
    best_cm_threshold = None

    for cw in class_weights_list:
        print(f"\nTesting Class Weight: {cw}")
        rf_models = []

        # Train ensemble models
        for i, maj_chunk in enumerate(majority_chunks):
            print(f"Training Model {i+1}/{n_chunks} with class_weight={cw}")
            balanced_train = pd.concat([maj_chunk, minority_train])

            X_train_bal = balanced_train.drop('Defect', axis=1)
            X_train_bal = pd.get_dummies(X_train_bal, columns=['Tool_Type'], drop_first=True)
            X_train_bal = X_train_bal.reindex(columns=X_train_global.columns, fill_value=0)
            y_train_bal = balanced_train['Defect']

            rf = RandomForestClassifier(
                n_estimators=400,
                max_depth=30,
                min_samples_split=10,
                min_samples_leaf=2,
                class_weight=cw,
                random_state=67,
                oob_score=True,
                n_jobs=-1
            )
            rf.fit(X_train_bal, y_train_bal)
            rf_models.append(rf)

        # Ensemble prediction
        probas = np.zeros((X_test_global.shape[0], len(rf_models)))
        for i, rf in enumerate(rf_models):
            probas[:, i] = rf.predict_proba(X_test_global)[:, 1]
        avg_probas = probas.mean(axis=1)

        # # Threshold optimization
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
        f2 = fbeta_score(y_test_global, y_pred_final, beta=2, zero_division=0)

        cm = confusion_matrix(y_test_global, y_pred_final)

        # If this model has the best F1, store its confusion matrix
        if f1 > best_cm_f1:
            best_cm_f1 = f1
            best_cm = cm
            best_cm_threshold = best_thresh
            best_cm_model_info = cw


        # Store results
        results.append({
            "class_weight": cw,
            "threshold": best_thresh,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f2": f2
        })

        # Print current configuration
        print(f"Best Threshold: {best_thresh:.3f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"F2 Score: {f2:.4f}")
        print(f"Confusion Matrix:\n{cm}")

# -------------------------------
# Print summary of all results
# -------------------------------
print("\n==================================================")
print("Summary of All Class Weight Experiments:")
for r in results:
    print(f"class_weight: {r['class_weight']}, threshold: {r['threshold']:.3f}, "
          f"accuracy: {r['accuracy']:.4f}, precision: {r['precision']:.4f}, "
          f"recall: {r['recall']:.4f}, f1: {r['f1']:.4f}")

# -------------------------------
# Print best by each metric
# -------------------------------
best_acc = max(results, key=lambda x: x['accuracy'])
best_prec = max(results, key=lambda x: x['precision'])
best_rec = max(results, key=lambda x: x['recall'])
best_f1 = max(results, key=lambda x: x['f1'])

print("\n==================================================")
print("Best Configuration by Each Metric:")
print(f"Highest Accuracy: class_weight={best_acc['class_weight']}, threshold={best_acc['threshold']:.3f}, accuracy={best_acc['accuracy']:.4f}")
print(f"Highest Precision: class_weight={best_prec['class_weight']}, threshold={best_prec['threshold']:.3f}, precision={best_prec['precision']:.4f}")
print(f"Highest Recall: class_weight={best_rec['class_weight']}, threshold={best_rec['threshold']:.3f}, recall={best_rec['recall']:.4f}")
print(f"Highest F1 Score: class_weight={best_f1['class_weight']}, threshold={best_f1['threshold']:.3f}, f1={best_f1['f1']:.4f}")

for r in results:
    r['combined_score'] = r['accuracy'] + r['precision'] + r['recall'] + r['f1']

# Find the best overall configuration
best_overall = max(results, key=lambda x: x['combined_score'])

print("\n==================================================")
print("Best Overall Ensemble Configuration (considering all 4 metrics):")
print(f"class_weight: {best_overall['class_weight']}")
print(f"threshold: {best_overall['threshold']:.3f}")
print(f"accuracy: {best_overall['accuracy']:.4f}")
print(f"precision: {best_overall['precision']:.4f}")
print(f"recall: {best_overall['recall']:.4f}")
print(f"f1: {best_overall['f1']:.4f}")

print("\n==================================================")
print("BEST CONFUSION MATRIX (by highest F1 score)")
print(f"Class Weight: {best_cm_model_info}")
print(f"Threshold: {best_cm_threshold}")
print(f"Best F1: {best_cm_f1}")

print("\nConfusion Matrix:")
print(best_cm)


