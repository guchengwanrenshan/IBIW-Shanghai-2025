import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import lightgbm as lgb
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import joblib
# ---------- Setup ----------
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, rdMolDescriptors, RDKFingerprint, rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, roc_auc_score,
    mean_absolute_error, mean_squared_error
)
from scipy.stats import pearsonr, kendalltau
def get_physical_chemistry(mol):
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    rtb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    psa = rdMolDescriptors.CalcTPSA(mol)
    stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    logp, mr = rdMolDescriptors.CalcCrippenDescriptors(mol)
    csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    nrings = rdMolDescriptors.CalcNumRings(mol)
    nrings_h = rdMolDescriptors.CalcNumHeterocycles(mol)
    nrings_ar = rdMolDescriptors.CalcNumAromaticRings(mol)
    nrings_ar_h = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    mw = rdMolDescriptors._CalcMolWt(mol)
    atm_hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
    atm_heavy = mol.GetNumHeavyAtoms()
    atm_all = mol.GetNumAtoms()
    return [hbd, hba, rtb, psa, stereo, logp, mr, csp3, nrings, nrings_h, nrings_ar, nrings_ar_h, spiro, mw,
            atm_hetero, atm_heavy, atm_all]

def extract_features(smiles):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
    else:
        pc_property = get_physical_chemistry(mol)
        fp_Morgan = np.zeros((2048,))
        DataStructs.ConvertToNumpyArray(mfpgen.GetFingerprint(mol), fp_Morgan)
        
        fp_MACCS = np.zeros((167,))
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), fp_MACCS)


        features = np.concatenate([pc_property,fp_Morgan,fp_MACCS]).astype(np.float32)

    return features

# ---------- Chunked Processing ----------
from joblib import Parallel, delayed
def process_in_chunks(df, model, chunk_size=10000, output_file='out.csv'):
    if os.path.exists(output_file):
        os.remove(output_file)
    header_written = False

    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        print(f"\n🔄 Processing chunk {i+1}/{num_chunks}...")

        smis = chunk["isosmiles"].tolist()
        ids = chunk["Catalog_ID"].tolist()

        # Parallel feature extraction (only SMILES)
        features_list = Parallel(n_jobs=-1)(
            delayed(extract_features)(smi) for smi in tqdm(smis)
        )

        # Filter valid ones
        valid_data = [
            (rid, smi, feat) for rid, smi, feat in zip(ids, smis, features_list) if feat is not None
        ]
        if not valid_data:
            print(f"⚠️ No valid molecules in chunk {i+1}")
            continue

        ids_valid, smis_valid, feats = zip(*valid_data)
        feats = np.array(feats, dtype=np.float32)

        preds = model.predict_proba(feats)[:, 1]

        chunk_df = pd.DataFrame({
            "Catalog_ID": ids_valid,
            "isosmiles": smis_valid,
            "prediction": preds
        })
        chunk_df.to_csv(output_file, mode='a', index=False, header=not header_written)
        header_written = True


# ---------- Main ----------
if __name__ == "__main__":
    modellist=['rf_0.pkl','rf_1.pkl','rf_2.pkl']
    #modellist=['lgbm_0.pkl','lgbm_1.pkl','lgbm_2.pkl']
    #modellist=['xgb_0.pkl','xgb_1.pkl','xgb_2.pkl']
    for i in modellist:
        model = joblib.load(i)
        for no in range(1,48):
            df=pd.read_csv(f'clean_smi3/chunk_{no}.smi')
            process_in_chunks(df, model, chunk_size=10000, output_file=f"{no}_{i}.csv")
    print("✅ All chunks processed and saved.")
    
