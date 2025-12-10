import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, fbeta_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y, random_state=35) #stratified y to maintain equal split for defects and non-defects
numerical_cols = ['Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
                  'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
                  'Vibration_Level', 'UV_Exposure_Intensity', 'Particle_Count']

scaler = StandardScaler() #scaling done to prevent features with larger ranges fucking over those with smaller ranges
scaler.fit(x_train[numerical_cols])
x_train.loc[:,numerical_cols] = scaler.transform(x_train[numerical_cols])
x_test.loc[:,numerical_cols] = scaler.transform(x_test[numerical_cols])
print('Pre-processing done!')

model = LogisticRegression(class_weight='balanced', random_state=35)
model.fit(x_train, y_train)
print('Fitting done!')
y_pred = model.predict(x_test)
print('Yezzur!')           
ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=['No Defect', 'Defect'])
plt.title('Confusion Matrix')
plt.show()      