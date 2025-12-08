#Because caramel is still kinda basic ngl, but definitely an improvement over vanilla.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, ConfusionMatrixDisplay, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/aryankulkarni/Documents/UC Berkeley Coursework/SDSE/La_Finale/Semiconductor Quality Control/semiconductor_quality_control.csv')
print(df.describe())
print(df.head())
y = df['Defect']
sybau = ['Process_ID', 'Timestamp', 'Wafer_ID', 'Defect', 'Join_Status']
x = df.drop(columns=sybau)
x = pd.get_dummies(x, columns=['Tool_Type'], drop_first=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y, random_state=1)
numerical_cols = ['Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
                  'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
                  'Vibration_Level', 'UV_Exposure_Intensity', 'Particle_Count']
scaler = StandardScaler()
scaler.fit(x_train[numerical_cols])
x_train.loc[:,numerical_cols] = scaler.transform(x_train[numerical_cols])
x_test.loc[:,numerical_cols] = scaler.transform(x_test[numerical_cols])
print('Pre-processing done!')
model = LogisticRegression(class_weight='balanced', random_state=1)
model.fit(x_train, y_train)
print('Done!')
y_pred = model.predict_proba(x_test)[:,1]
#If we now use proba instead of predict, we can improve the upon the shitshow of the last code by tuning the threshold.
# my_proba_threshold = 0.05
# y_pred = (y_pred >= my_proba_threshold).astype(int)

best_score = 0
best_thresh = 0.5

# for t in np.arange(0.1,1.0,0.05):
#     temp_pred = (y_pred>= t).astype(int)
#     f2 = fbeta_score(y_test, temp_pred, beta=2)
#     print('Threshold:',t, 'F2 Score: ', f2)
#     if f2>best_score: 
#         best_score= f2
#         best_thresh=t

for t in np.arange(0.1,1.0,0.05):
    temp_pred = (y_pred>= t).astype(int)
    f2 = recall_score(y_test, temp_pred)
    print('Threshold:',t, 'Recall: ', f2)
    if f2>best_score: 
        best_score= f2
        best_thresh=t

print(f"Best Threshold=", best_thresh, "Recall=", best_score)
print('Yezzur!')              
final_preds = (y_pred >= best_thresh).astype(int)
cm = confusion_matrix(y_test, final_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Defect', 'Defect'])
disp.plot()
plt.title(f'Confusion Matrix at Threshold {best_thresh}')
plt.show()

#However, as we can see, the model just predicts everything as defective by minimizing threshold. This is cooked out the wazoo.