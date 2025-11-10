import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

def calculate_all_properties(smiles):
    """Calculate EVERY available RDKit descriptor"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Get all descriptor names
    descriptor_names = [name[0] for name in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    
    try:
        descriptors = calculator.CalcDescriptors(mol)
        return dict(zip(descriptor_names, descriptors))
    except:
        # If any descriptor fails, return None for all
        return None

# Load your data
df = pd.read_csv('C:\\Users\\0hmse\\PycharmProjects\\Generic Environment\\Statistics\\Final Project\\melting-point\\my_train.csv')

print("Calculating ALL RDKit descriptors... This might take a while...")

# Calculate all descriptors
properties_list = []
for smiles in df['SMILES']:
    props = calculate_all_properties(smiles)
    properties_list.append(props)

properties_df = pd.DataFrame(properties_list)

# Combine with original dataframe
df_with_props = pd.concat([df, properties_df], axis=1)

# Drop rows where SMILES couldn't be parsed or descriptors failed
original_rows = len(df)
df_with_props = df_with_props.dropna(subset=properties_df.columns)
print(f"Dropped {original_rows - len(df_with_props)} rows due to failed descriptor calculation")

# Now use ALL descriptors for regression
X = df_with_props[properties_df.columns]
y = df_with_props['Tm']

print(f"Original dataset shape: {df.shape}")
print(f"Processed dataset shape: {df_with_props.shape}")
print(f"Number of features: {len(properties_df.columns)}")

# Remove low-variance features (often constant or near-constant)
selector = VarianceThreshold(threshold=0.01)  # Remove features with <1% variance
X_reduced = selector.fit_transform(X)

print(f"Features after variance threshold: {X_reduced.shape[1]}")

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Scale features for better linear regression performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"Training R²: {train_score:.4f}")
print(f"Test R²: {test_score:.4f}")
print(f"Overfitting gap: {train_score - test_score:.4f}")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

# Create a comprehensive visualization figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Linear Regression Model Performance (Test R² = {test_score:.3f})', fontsize=16, fontweight='bold')

# Plot 1: Training predictions vs actual (sorted) - Your template
sorted_indices = np.argsort(y_train.values)
y_train_sorted = y_train.values[sorted_indices]
y_train_pred_sorted = y_train_pred[sorted_indices]

axes[0,0].scatter(range(len(y_train_sorted)), y_train_sorted, 
           alpha=0.6, label='Actual', color='blue', s=30)
axes[0,0].scatter(range(len(y_train_pred_sorted)), y_train_pred_sorted, 
           alpha=0.6, label='Predicted', color='red', s=30)
axes[0,0].set_xlabel('Compounds (Sorted by Actual Value)')
axes[0,0].set_ylabel('Melting Point (Tm)')
axes[0,0].set_title('Training Data: Actual vs Predicted\n(Sorted by Actual Value)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Add a line showing perfect prediction
min_val = min(y_train_sorted.min(), y_train_pred_sorted.min())
max_val = max(y_train_sorted.max(), y_train_pred_sorted.max())
axes[0,0].plot([0, len(y_train_sorted)], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')

# Plot 2: Test predictions vs actual (sorted)
sorted_indices_test = np.argsort(y_test.values)
y_test_sorted = y_test.values[sorted_indices_test]
y_test_pred_sorted = y_test_pred[sorted_indices_test]

axes[0,1].scatter(range(len(y_test_sorted)), y_test_sorted, 
           alpha=0.6, label='Actual', color='blue', s=30)
axes[0,1].scatter(range(len(y_test_pred_sorted)), y_test_pred_sorted, 
           alpha=0.6, label='Predicted', color='red', s=30)
axes[0,1].set_xlabel('Compounds (Sorted by Actual Value)')
axes[0,1].set_ylabel('Melting Point (Tm)')
axes[0,1].set_title('Test Data: Actual vs Predicted\n(Sorted by Actual Value)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)
axes[0,1].plot([0, len(y_test_sorted)], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')

# Plot 3: Residuals plot for training data
residuals_train = y_train - y_train_pred
axes[0,2].scatter(y_train_pred, residuals_train, alpha=0.6, color='green', s=30)
axes[0,2].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0,2].set_xlabel('Predicted Values')
axes[0,2].set_ylabel('Residuals (Actual - Predicted)')
axes[0,2].set_title('Training Data: Residuals Plot')
axes[0,2].grid(True, alpha=0.3)

# Plot 4: Residuals plot for test data
residuals_test = y_test - y_test_pred
axes[1,0].scatter(y_test_pred, residuals_test, alpha=0.6, color='purple', s=30)
axes[1,0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1,0].set_xlabel('Predicted Values')
axes[1,0].set_ylabel('Residuals (Actual - Predicted)')
axes[1,0].set_title('Test Data: Residuals Plot')
axes[1,0].grid(True, alpha=0.3)

# Plot 5: Actual vs Predicted scatter (training)
axes[1,1].scatter(y_train, y_train_pred, alpha=0.6, color='blue', s=30, label='Training')
axes[1,1].scatter(y_test, y_test_pred, alpha=0.6, color='red', s=30, label='Test')
axes[1,1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', linewidth=2, label='Perfect Prediction')
axes[1,1].set_xlabel('Actual Melting Point')
axes[1,1].set_ylabel('Predicted Melting Point')
axes[1,1].set_title('Actual vs Predicted Values')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

# Plot 6: Feature importance (top 15 features)
feature_importance = pd.DataFrame({
    'feature': X.columns[selector.get_support()],
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

# Plot top 15 features
top_features = feature_importance.head(15)
colors = ['red' if coef < 0 else 'blue' for coef in top_features['coefficient']]
axes[1,2].barh(range(len(top_features)), top_features['abs_coefficient'], color=colors, alpha=0.7)
axes[1,2].set_yticks(range(len(top_features)))
axes[1,2].set_yticklabels(top_features['feature'])
axes[1,2].set_xlabel('Absolute Coefficient Value')
axes[1,2].set_title('Top 15 Most Important Features\n(Red=Negative, Blue=Positive Impact)')
axes[1,2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# =============================================================================
# ADDITIONAL VISUALIZATIONS
# =============================================================================

# Correlation heatmap of top features + target
plt.figure(figsize=(12, 10))
top_feature_names = top_features['feature'].tolist()
corr_data = pd.concat([X[top_feature_names], y], axis=1)
corr_matrix = corr_data.corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('Correlation Matrix: Top Features vs Target (Tm)')
plt.tight_layout()
plt.show()

# Distribution of melting points
plt.figure(figsize=(10, 6))
plt.hist(y, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(y.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {y.mean():.1f}°C')
plt.xlabel('Melting Point (°C)')
plt.ylabel('Frequency')
plt.title('Distribution of Melting Points in Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print performance metrics
print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"Training R²:   {train_score:.4f}")
print(f"Test R²:       {test_score:.4f}")
print(f"Training RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.2f}°C")
print(f"Test RMSE:     {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}°C")
print(f"Overfitting:   {train_score - test_score:.4f}")

print("\nTop 5 most important features:")
print(top_features[['feature', 'coefficient']].head())