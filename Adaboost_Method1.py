import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, f1_score



#Based on what I know, you have three choices: 
#downsampling
#keep the 15% same (617), but just use 15% of the 85% (that i,s sample 617 out of 3602) to keep the balance. This is the easiest one, but have a problem with using less data. You can sample 15% of 85% differently for every epoch to make sure the model sees the whole data. 
#upsampling
#This is what people usually do. It's called data augmentation. You make synthetic data (3602-617) to match the number of positive samples.
#None of the above
#In this case, you have to divide train/val/test with the same ratio of 85:15. That is, suppose that you divide your dataset into train/val/test with a ratio of 7:2:1. Then, for all three train/val/test, datasets, each of those should keep the ratio of 85:15 of 0,1 label data.



# Method 1, Downsampling
rng_seed = 42

# 1. Load Data
raw_data = pd.read_csv('C:\\Users\\0hmse\\PycharmProjects\\Generic Environment\\Statistics\\Final Project\\semiconductor_quality_control.csv')
features = raw_data[['Tool_Type','Chamber_Temperature','Gas_Flow_Rate','RF_Power','Etch_Depth','Rotation_Speed','Vacuum_Pressure','Stage_Alignment_Error','Vibration_Level','UV_Exposure_Intensity','Particle_Count']]
features = pd.get_dummies(features, columns=['Tool_Type'], drop_first=False)
features['Defect'] = raw_data['Defect'] # Temporarily join Target back to features for easier filtering

# 2. GLOBAL SPLIT: Separate out 20% unseen data for final testing
train_df, test_df = train_test_split(features, test_size=0.2, stratify=features['Defect'], random_state=rng_seed)

# Separate X and y for the final test set
X_test_global = test_df.drop('Defect', axis=1)
y_test_global = test_df['Defect']

# 3. PREPARE TRAINING CHUNKS
# Filter Majority (0) and Minority (1) from the Training set
train_majority = train_df[train_df['Defect'] == 0]
train_minority = train_df[train_df['Defect'] == 1]

# Shuffle majority data to ensure random chunks
train_majority = train_majority.sample(frac=1, random_state=rng_seed)

# Split Majority into 6 chunks
# np.array_split will handle cases where it doesn't divide perfectly evenly
majority_chunks = np.array_split(train_majority, 6)

print(f"Total Training Majority Samples: {len(train_majority)}")
print(f"Total Training Minority Samples: {len(train_minority)}")
print(f"Splitting Majority into 6 chunks of approx {len(majority_chunks[0])} samples each.\n")

# 4. TRAIN 6 MODELS (The Ensemble)
models = []
print(f"{'Model Run':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 35)

for i, maj_chunk in enumerate(majority_chunks):
    
    # Combine 1 chunk of Majority + ALL Minority
    # This creates a roughly 50/50 balanced dataset for this specific model
    balanced_train = pd.concat([maj_chunk, train_minority])
    
    X_train_bal = balanced_train.drop('Defect', axis=1)
    y_train_bal = balanced_train['Defect']
    
    # Initialize Model (Standard AdaBoost, no class_weight needed because data is balanced now!)
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1), # Stumps work well here because data is balanced
        n_estimators=100,
        learning_rate=0.1,
        random_state=rng_seed
    )
    
    ada.fit(X_train_bal, y_train_bal)
    models.append(ada)
    
    # Evaluate individual model on global test set
    y_pred = ada.predict(X_test_global)
    recall = recall_score(y_test_global, y_pred)
    f1 = f1_score(y_test_global, y_pred)
    
    print(f"Model {i+1:<4} | {recall:.4f}     | {f1:.4f}")


# 5. FINAL ENSEMBLE VOTING (Soft Voting)
# We average the predicted probabilities of all 6 models
print("\n--- Final Ensemble Results ---")

# Get probability predictions from all models (shape: [6, n_samples, 2])
all_preds = np.array([model.predict_proba(X_test_global)[:, 1] for model in models])

# Average the probabilities across the 6 models
avg_preds = np.mean(all_preds, axis=0)

# Convert probability to class (Threshold 0.5)
y_pred_ensemble = (avg_preds > 0.5).astype(int)

final_recall = recall_score(y_test_global, y_pred_ensemble)
final_f1 = f1_score(y_test_global, y_pred_ensemble)

print(f"Ensemble Recall:   {final_recall:.4f}")
print(f"Ensemble F1-Score: {final_f1:.4f}")