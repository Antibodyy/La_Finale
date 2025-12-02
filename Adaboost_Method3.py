import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score



#Based on what I know, you have three choices: 
#downsampling
#keep the 15% same (617), but just use 15% of the 85% (that i,s sample 617 out of 3602) to keep the balance. This is the easiest one, but have a problem with using less data. You can sample 15% of 85% differently for every epoch to make sure the model sees the whole data. 
#upsampling
#This is what people usually do. It's called data augmentation. You make synthetic data (3602-617) to match the number of positive samples.
#None of the above
#In this case, you have to divide train/val/test with the same ratio of 85:15. That is, suppose that you divide your dataset into train/val/test with a ratio of 7:2:1. Then, for all three train/val/test, datasets, each of those should keep the ratio of 85:15 of 0,1 label data.



# Method 3, None of the above
rng_seed = 42

raw_data = pd.read_csv('C:\\Users\\0hmse\\PycharmProjects\\Generic Environment\\Statistics\\Final Project\\semiconductor_quality_control.csv')
features = raw_data[['Tool_Type','Chamber_Temperature','Gas_Flow_Rate','RF_Power','Etch_Depth','Rotation_Speed','Vacuum_Pressure','Stage_Alignment_Error','Vibration_Level','UV_Exposure_Intensity','Particle_Count']]
features = pd.get_dummies(features, columns=['Tool_Type'], drop_first=False)
defect = raw_data['Defect']

N0 = defect.value_counts()[0]
N1 = defect.value_counts()[1]
print(f"Number of Non-Defective Samples: {N0}")
print(f"Number of Defective Samples: {N1}")
baseline_accuracy = N0 / (N0 + N1)
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")


X_train, X_test, y_train, y_test = train_test_split(features, defect, test_size=0.2, random_state=42, stratify=defect)

#stratify=defect makes sure that the data stays in the original proportions

print("Train class distribution:")
print(y_train.value_counts())
print("\nTest class distribution:")
print(y_test.value_counts())

print("\nTrain distribution (%):")
print(y_train.value_counts(normalize=True))

print("\nTest distribution (%):")
print(y_test.value_counts(normalize=True))

#n_list = [10, 25, 50, 100, 200, 400, 800]
n_list = [5000]
recall_scores = []

for n in n_list:
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n,
        learning_rate=0.1,   # Maybe try diff learning rater later???
        random_state=rng_seed
    )
    
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    recall = recall_score(y_test, y_pred)
    recall_scores.append(recall)
    print(f"n_estimators = {n}, Recall = {recall:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(n_list, recall_scores, marker='o')
plt.xlabel("Number of Estimators")
plt.ylabel("Recall (Defect = 1)")
plt.title("AdaBoost Recall vs. Number of Estimators")
plt.grid(True)
plt.show()

#even with 5000 weak learners still have a recall of 0