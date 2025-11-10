import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/aryankulkarni/Documents/UC Berkeley Coursework/SDSE/La_Finale/Semiconductor Quality Control/semiconductor_quality_control.csv')
print(df.describe())
y = df['Defect']
sybau = ['Process_ID', 'Timestamp', 'Wafer_ID', 'Defect', 'Join_Status']
x = df.drop(columns=sybau)
x = pd.get_dummies(x, columns=['Tool_Type'], drop_first=False)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y)
numerical_cols = ['Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
                  'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
                  'Vibration_Level', 'UV_Exposure_Intensity', 'Particle_Count']
scaler = StandardScaler()
scaler.fit(x_train[numerical_cols])
x_train.loc[:,numerical_cols] = scaler.transform(x_train[numerical_cols])
x_test.loc[:,numerical_cols] = scaler.transform(x_test[numerical_cols])
print('Pre-processing done lesogogogogoogogogogogo')
model = LogisticRegression(class_weight='balanced')
model.fit(x_train, y_train)
print('Done!')
y_pred = model.predict(x_test)
print('Yezzur!')
print(classification_report(y_test, y_pred))            
ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=['No Defect', 'Defect'])
plt.title('Confusion Matrix')
plt.show()      