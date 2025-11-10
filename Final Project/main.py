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

# nope R^2 went down when only using the bellow hhhmmmm
  #mol_weight          : 0.2680 (±0.0138)
  #heavy_atoms         : 0.3704 (±0.0145)
 # molar_volume        : 0.3954 (±0.0173)
  #tpsa                : 0.5893 (±0.0227)

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

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"R² Score: {model.score(X_test, y_test):.4f}")



# Calculate descriptors
#mw = Descriptors.MolWt(mol)
#num_aromatic_rings = Descriptors.NumAromaticRings(mol)
#num_h_donors = Descriptors.NumHDonors(mol) # Counts NH and OH groups
#tpsa = Descriptors.TPSA(mol)




##    Large + polar + rigid = very high melting point (e.g., ionic compounds)

##  Small + nonpolar + flexible = very low melting point (e.g., gases)

## Hydrogen bonding can override size effects

##  Symmetry can compensate for lower molecular weight


#Positive correlation with melting point: MolWt, TPSA, NumHDonors, NumRings, BertzCT

#Negative correlation with melting point: NumRotatableBonds, MolLogP (for polar compounds)