import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, fbeta_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/aryankulkarni/Documents/UC Berkeley Coursework/SDSE/La_Finale/Semiconductor Quality Control/semiconductor_quality_control.csv')
# NEED TO CHANGE THIS PATH TO WHEREVER THE DATASET IS LOCATED LATER!!!!!!!!

#print(df.describe())
#print(df.head())

y = df['Defect']
#sorry for the variable name lol
sybau = ['Process_ID', 'Timestamp', 'Wafer_ID', 'Defect', 'Join_Status']
x = df.drop(columns=sybau)
x = pd.get_dummies(x, columns=['Tool_Type'], drop_first=True) #tool type was text categorical, so one-hot encoded it, drop first cuz only one is needed to avoid multicollinearity
numerical_cols = ['Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
                  'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
                  'Vibration_Level', 'UV_Exposure_Intensity', 'Particle_Count']

pipeline = Pipeline([('scaler', StandardScaler()),('classifier', LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000, random_state=1))])
#i'll just use this nifty little thing called pipeline now
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=1) #stratified because of class imbalance
scoring = ['recall', 'f1', 'accuracy', 'precision']
scores = cross_validate(pipeline, x, y, cv=cv, scoring=scoring, return_train_score=False)
print(f"{'Fold':<6} {'Recall':<10} {'Precision':<10} {'F1':<10} {'Accuracy':<10}")
print("-" * 50)

for i in range(4):
    print(f"{i+1:<6} {scores['test_recall'][i]:<10.4f} {scores['test_precision'][i]:<10.4f} {scores['test_f1'][i]:<10.4f} {scores['test_accuracy'][i]:<10.4f}")

print("-" * 50)
print(f"Mean   {np.mean(scores['test_recall']):<10.4f} {np.mean(scores['test_precision']):<10.4f} {np.mean(scores['test_f1']):<10.4f} {np.mean(scores['test_accuracy']):<10.4f}")
print(f"Std    {np.std(scores['test_recall']):<10.4f} {np.std(scores['test_precision']):<10.4f} {np.std(scores['test_f1']):<10.4f} {np.std(scores['test_accuracy']):<10.4f}")        