import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('data.csv')

AA_MAPPING = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19
}


mol_objects = []
for smile in data['Drug']:
    mol = Chem.MolFromSmiles(str(smile))
    mol_objects.append(mol)

seq_objects = [str(seq) for seq in data['Protein'].tolist()]

labels = data['label'].tolist()

# Initialize empty list to store graph representations of molecules
graphs = []

# Loop over all molecule objects and create graph representations
for i, mol in enumerate(mol_objects):
    if mol is None:
        continue
    graph = {}

    # Add node features for atoms
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        if charge == 0:
            feature = [1, 0]  # neutral atom
        elif charge > 0:
            feature = [0, 1]  # positively charged atom
        else:
            feature = [0, -1]  # negatively charged atom
        graph[atom.GetIdx()] = {'symbol': symbol, 'Charge': feature}

    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        graph[begin_atom][end_atom] = {'bond_type': bond_type}
        graph[end_atom][begin_atom] = {'bond_type': bond_type}

    seq = seq_objects[i]
    for j, aa in enumerate(seq):
        if aa in AA_MAPPING:
            feature = [0] * 20
            feature[AA_MAPPING[aa]] = 1
            graph[j + mol.GetNumAtoms()] = {'symbol': aa, 'feature': feature}

    graph['size'] = mol.GetNumAtoms()
   
    
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            graph[atom.GetIdx()]['aromatic'] = True
        else:
            graph[atom.GetIdx()]['aromatic'] = False

    # Add label for drug target interaction
    graph['label'] = labels[i]

    graphs.append(graph)

print(graphs[1])
