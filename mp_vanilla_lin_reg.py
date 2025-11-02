import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv('/Users/aryankulkarni/Documents/UC Berkeley Coursework/SDSE/La_Finale/melting-point/train.csv')
print(df.describe())
y = df['Tm']
x = df.drop(['Tm', 'id', 'SMILES'], axis = 1)
print("Shape of features (x) = ", x.shape)
print("Shape of target (y) = ", y.shape)
model = LinearRegression()
model.fit(x,y)
print("Yessir!")
df_test = pd.read_csv('/Users/aryankulkarni/Documents/UC Berkeley Coursework/SDSE/La_Finale/melting-point/test.csv')
test_ids = df_test['id']
x_test = df_test.drop(['id', 'SMILES'], axis = 1)
y_pred = model.predict(x_test)
predictions = pd.DataFrame({
    'id' : test_ids,
    'Tm' : y_pred
})
predictions.to_csv('thermo_preds.csv', index = False)
print('Done ;)')
