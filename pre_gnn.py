import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs
from torch_geometric.nn import GATConv, global_add_pool
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import Pool
import numpy as np

device = torch.device('cpu')  # CPU only for multiprocessing
model_paths = ['gnn_0.pt', 'gnn_1.pt', 'gnn_2.pt']

# --- Feature extractors ---
def get_physical_chemistry(mol):
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return [
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumAtomStereoCenters(mol),
        *rdMolDescriptors.CalcCrippenDescriptors(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumHeterocycles(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        rdMolDescriptors.CalcNumSpiroAtoms(mol),
        rdMolDescriptors._CalcMolWt(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        mol.GetNumHeavyAtoms(),
        mol.GetNumAtoms()
    ]

def get_atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        int(atom.GetIsAromatic()),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs()
    ], dtype=torch.float)

def get_bond_features(bond):
    return torch.tensor([
        bond.GetBondTypeAsDouble(),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ], dtype=torch.float)

def extract_features(smiles):
    mfpgen = GetMorganGenerator(radius=2, fpSize=2048)
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    if mol is None:
        return None, None

    des = np.concatenate([
        get_physical_chemistry(mol),
        DataStructs.ConvertToNumpyArray(mfpgen.GetFingerprint(mol), np.zeros((2048,), dtype=np.float32)) or np.zeros(2048),
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), np.zeros((167,), dtype=np.float32)) or np.zeros(167)
    ]).astype(np.float32)

    x = torch.stack([get_atom_features(a) for a in mol.GetAtoms()])
    edge_index, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = get_bond_features(b)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    edge_attr = torch.stack(edge_attr, dim=0)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return des, graph

# --- GATNet definition ---
class GATNet(torch.nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, des_feat_dim, hidden_dim, dp=0.2):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(node_feat_dim, hidden_dim*4, heads=1, edge_dim=edge_feat_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim*4)
        self.conv2 = GATConv(hidden_dim*4, hidden_dim*4, heads=1, edge_dim=edge_feat_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim*4)
        self.conv3 = GATConv(hidden_dim*4, hidden_dim, heads=1, edge_dim=edge_feat_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + des_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch, des):
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.leaky_relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = F.leaky_relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        x = global_add_pool(x, batch)
        x_all = torch.cat([x, des], dim=1)
        return self.fc(x_all).view(-1)

# --- Model loader ---
def load_models():
    models = []
    for path in model_paths:
        model = GATNet(node_feat_dim=5, edge_feat_dim=3, des_feat_dim=2232, hidden_dim=48, dp=0.0)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        models.append(model)
    return models

# --- Worker function ---
def predict_worker(df_chunk):
    models = load_models()
    results = []
    for _, row in df_chunk.iterrows():
        smiles = row['SMILES']
        rid = row['RandomID']
        try:
            des, graph = extract_features(smiles)
            if graph is None:
                results.append([rid, None, None, None])
                continue
            graph.des = torch.tensor(des, dtype=torch.float).unsqueeze(0)
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
            graph = graph.to(device)
            with torch.no_grad():
                probs = [
                    float(torch.sigmoid(m(graph.x, graph.edge_index, graph.edge_attr, graph.batch, graph.des)))
                    for m in models
                ]
            results.append([rid] + probs)
        except Exception as e:
            print(f"Error processing SMILES {SMILES}: {e}")
            results.append([rid, None, None, None])
    return results

# --- Main parallel runner ---
def run_parallel_prediction(input_csv, output_csv, num_workers=30):
    df = pd.read_csv(input_csv)
    chunks = np.array_split(df, num_workers)

    with Pool(processes=num_workers) as pool:
        all_results = pool.map(predict_worker, chunks)

    flat_results = [item for sublist in all_results for item in sublist]
    result_df = pd.DataFrame(flat_results, columns=['RandomID', 'prob_0', 'prob_1', 'prob_2'])
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Saved predictions to: {output_csv}")

# --- Entry point ---
if __name__ == "__main__":
    run_parallel_prediction("total_sure_smi.csv", "gnn_3model_predictions_parallel.csv", num_workers=30)

