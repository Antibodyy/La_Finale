import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats


# Your example molecule
#smiles = "Nc1ccc(O)cc1" # This is 4-aminophenol
#mol = Chem.MolFromSmiles(smiles)

#descrs = Descriptors.CalcMolDescriptors(mol)



# Load your data
df = pd.read_csv('C:\\Users\\0hmse\\PycharmProjects\\Generic Environment\\Statistics\\Final Project\melting-point\\my_train.csv')

def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # replace with bellow mentioned descriptors
    properties = {
        # Molecular Size
        'mol_weight': Descriptors.MolWt(mol),
        'heavy_atoms': Descriptors.HeavyAtomCount(mol),
        'molar_volume': Descriptors.MolMR(mol),  # Molar refractivity correlates with volume
        
        # Polarity
        'tpsa': Descriptors.TPSA(mol),  # Topological polar surface area
        'logp': Descriptors.MolLogP(mol),  # Hydrophobicity
        'fraction_csp3': Descriptors.FractionCSP3(mol),  # More sp3 = less planar = affects polarity
        
        # Flexibility
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'stereo_centers': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        'ring_count': Descriptors.RingCount(mol),  # More rings = less flexible
        
        # Hydrogen Bonding Capacity
        'h_bond_donors': Descriptors.NumHDonors(mol),
        'h_bond_acceptors': Descriptors.NumHAcceptors(mol),

        # Molecular Complexity
        'aromatic_rings': Descriptors.NumAromaticRings(mol),
        'heteroatom_count': Descriptors.NumHeteroatoms(mol),
        'complexity': Descriptors.BalabanJ(mol),  # Graph complexity index
    }
    return properties



properties_df = df['SMILES'].apply(lambda x: pd.Series(calculate_properties(x)))
# Combine with original dataframe
df_with_props = pd.concat([df, properties_df], axis=1)

# Drop rows where SMILES couldn't be parsed
df_with_props = df_with_props.dropna(subset=properties_df.columns)

# Now use all 12 RDKit features for regression
X = df_with_props[properties_df.columns]  # Your 12 chemical properties
y = df_with_props['Tm']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# =============================================================================
# 1. FEATURE SIGNIFICANCE ANALYSIS
# =============================================================================

print("=" * 50)
print("FEATURE SIGNIFICANCE ANALYSIS")
print("=" * 50)

# Method 1: Correlation with target
print("\n1. CORRELATION WITH TARGET:")
correlations = X_train.corrwith(y_train)
for feature, corr in correlations.items():
    print(f"  {feature:20s}: {corr:.4f}")

# Method 2: Feature coefficients (standardized for comparison)
print("\n2. STANDARDIZED COEFFICIENTS:")
# Standardize coefficients by multiplying by (std feature / std target)
coef_standardized = model.coef_ * (X_train.std().values / y_train.std())
for feature, coef in zip(X_train.columns, coef_standardized):
    print(f"  {feature:20s}: {coef:.4f}")

# Method 3: Statistical significance using permutation test
print("\n3. PERMUTATION IMPORTANCE:")
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(model, X_train, y_train, n_repeats=30, random_state=42)
for i, feature in enumerate(X_train.columns):
    print(f"  {feature:20s}: {perm_importance.importances_mean[i]:.4f} "
          f"(±{perm_importance.importances_std[i]:.4f})")

# Method 4: p-values using statsmodels (more rigorous)
print("\n4. STATSMODELS SUMMARY (with p-values):")
import statsmodels.api as sm

X_train_with_const = sm.add_constant(X_train)  # Adds intercept term
sm_model = sm.OLS(y_train, X_train_with_const).fit()
print(sm_model.summary())

# =============================================================================
# 2. VISUALIZATION: Training data vs Predictions (sorted)
# =============================================================================

# Create sorted indices for training data
sorted_indices = np.argsort(y_train.values)
y_train_sorted = y_train.values[sorted_indices]
y_train_pred_sorted = y_train_pred[sorted_indices]

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Training predictions vs actual (sorted)
ax1.scatter(range(len(y_train_sorted)), y_train_sorted, 
           alpha=0.6, label='Actual', color='blue', s=30)
ax1.scatter(range(len(y_train_pred_sorted)), y_train_pred_sorted, 
           alpha=0.6, label='Predicted', color='red', s=30)
ax1.set_xlabel('Compounds (Sorted by Actual Value)')
ax1.set_ylabel('Target Value')
ax1.set_title('Training Data: Actual vs Predicted\n(Sorted by Actual Value)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add a line showing perfect prediction
min_val = min(y_train_sorted.min(), y_train_pred_sorted.min())
max_val = max(y_train_sorted.max(), y_train_pred_sorted.max())
ax1.plot([0, len(y_train_sorted)], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')

# Plot 2: Residuals analysis
residuals = y_train - y_train_pred
ax2.scatter(y_train_pred, residuals, alpha=0.6, color='green')
ax2.axhline(y=0, color='red', linestyle='--')
ax2.set_xlabel('Predicted Values')
ax2.set_ylabel('Residuals (Actual - Predicted)')
ax2.set_title('Residuals Plot')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# 3. MODEL PERFORMANCE METRICS
# =============================================================================

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Training R²:   {train_r2:.4f}")
print(f"Test R²:       {test_r2:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Test RMSE:     {test_rmse:.4f}")

# =============================================================================
# 4. ADDITIONAL: Feature relationships visualization
# =============================================================================

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr_matrix = pd.concat([X_train, y_train], axis=1).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.3f')
plt.title('Feature-Target Correlation Matrix')
plt.tight_layout()
plt.show()

# Feature distributions
X_train.hist(bins=20, figsize=(12, 8))
plt.suptitle('Feature Distributions')
plt.tight_layout()
plt.show()