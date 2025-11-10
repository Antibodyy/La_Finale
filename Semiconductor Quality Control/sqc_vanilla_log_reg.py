import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/aryankulkarni/Documents/UC Berkeley Coursework/SDSE/La_Finale/Thermophysical Property Detection/train.csv')
print(df.describe())
y = df['Tm']
x = df.drop(['Tm', 'id', 'SMILES'], axis = 1)

#Splitting training dataframe for testing different hyperparameters
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print("Shape of features (x) = ", x.shape)
print("Shape of target (y) = ", y.shape)
param_matrix = {'n_estimators' : [100,200,300,400,500], 'max_features' : [0.5,1,'sqrt'], 'max_depth' : [10,20,30,40,50], 'min_samples_leaf' : [1,5,10,15] }
param_range_nest = np.arange(10,500,10)
param_range_maxfea = np.arange(1,421,10)
param_range_maxdep = np.arange(5,100,5)
param_range_minsam = np.arange(1,50,1)
mae_scores_nest = []
mae_scores_maxfea = []
mae_scores_maxdep = []
mae_scores_minsam = []
for n in param_range_nest:
    print(f"Training with n_estimators={n}...")
    model = RandomForestRegressor(
        n_estimators=n,
        random_state=42,
        n_jobs=-1  
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores_nest.append(mae)


for n in param_range_maxfea:
    print(f"Training with max features={n}...")
    model = RandomForestRegressor(
        n_estimators=230,
        max_features=n,
        random_state=42,
        n_jobs=-1  
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores_maxfea.append(mae)

for n in param_range_maxdep:
    print(f"Training with max features={n}...")
    model = RandomForestRegressor(
        n_estimators = 230,
        max_depth=n,
        random_state=42,
        n_jobs=-1  
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores_maxdep.append(mae)

for n in param_range_minsam:
    print(f"Training with max features={n}...")
    model = RandomForestRegressor(
        n_estimators = 230,
        max_depth=n,
        random_state=42,
        n_jobs=-1  
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores_minsam.append(mae)

fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Hyperparameter Validation Curves for Random Forest', fontsize=16)
axs[0, 0].plot(param_range_nest, mae_scores_nest, marker='o', linestyle='-')
axs[0, 0].set_title('Effect of n_estimators')
axs[0, 0].set_xlabel('Number of Estimators (Trees)')
axs[0, 0].set_ylabel('Mean Absolute Error (MAE)')
axs[0, 0].grid(True)
axs[0, 1].plot(param_range_maxfea, mae_scores_maxfea, marker='o', linestyle='-')
axs[0, 1].set_title('Effect of max_features')
axs[0, 1].set_xlabel('Max Features')
axs[0, 1].set_ylabel('Mean Absolute Error (MAE)')
axs[0, 1].grid(True)
axs[1, 0].plot(param_range_maxdep, mae_scores_maxdep, marker='o', linestyle='-')
axs[1, 0].set_title('Effect of max_depth')
axs[1, 0].set_xlabel('Max Depth')
axs[1, 0].set_ylabel('Mean Absolute Error (MAE)')
axs[1, 0].grid(True)
axs[1, 1].plot(param_range_minsam, mae_scores_minsam, marker='o', linestyle='-')
axs[1, 1].set_title('Effect of min_samples_leaf')
axs[1, 1].set_xlabel('Min Samples per Leaf')
axs[1, 1].set_ylabel('Mean Absolute Error (MAE)')
axs[1, 1].grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.savefig('all_hyperparameter_plots.png')
plt.show()
# model = RandomForestRegressor() #why did the chicken cross the road?
# random_search = RandomizedSearchCV(
#     estimator=model,
#     param_distributions=param_matrix,
#     n_iter=20,
#     cv=5,
#     scoring='neg_mean_absolute_error',
#     n_jobs=-1,
#     random_state=42,
#     verbose=2 
# )

# random_search.fit(x,y)
# best_ranfor = random_search.best_estimator_
model = RandomForestRegressor(
        n_estimators = 230,
        max_depth=40,
        random_state=42,
        n_jobs=-1,
        max_features=350,
        min_samples_leaf=41

    )
model.fit(x,y)
print("Yessir!")
df_test = pd.read_csv('/Users/aryankulkarni/Documents/UC Berkeley Coursework/SDSE/La_Finale/Thermophysical Property Detection/test.csv')
test_ids = df_test['id']
x_test = df_test.drop(['id', 'SMILES'], axis = 1)
y_pred = model.predict(x_test)
predictions = pd.DataFrame({
    'id' : test_ids,
    'Tm' : y_pred
})
predictions.to_csv('thermo_preds_2.csv', index = False)
print('Done ;)')