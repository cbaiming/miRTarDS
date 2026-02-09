import json
import time
import csv
import unicodedata
import re
import copy
import numpy as np
import pandas as pd
import torch
from scipy.stats import skew
from sentence_transformers import util

# Machine Learning Imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             precision_score, recall_score)
from sklearn.preprocessing import StandardScaler

# XGBoost
import xgboost as xgb
from xgboost import XGBClassifier

# Custom Data Module (Assumed to exist in your environment)
import data.load_disease as process_db

# ==========================================
# 1. Configuration Management
# ==========================================

def load_config(config_path: str = 'config.json') -> dict:
    """
    Loads configuration parameters from a JSON file.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: {config_path} not found. Please ensure the config file exists.")
        exit(1)

# ==========================================
# 2. Data Loading & Preprocessing Helpers
# ==========================================

def standardize_disease_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", name)
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = name.lower()
    return name

def load_embeddings(csv_file_path: str) -> dict:
    dic_str_vector = {}
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Skip header
        headers = next(reader)
        for row in reader:
            disease_name = row[0]
            embedding = [float(value) for value in row[1:]]
            dic_str_vector[disease_name] = embedding
    return dic_str_vector

def load_dictionaries(config: dict, unique_miRNAs: list, unique_genes: list):
    # Load miRNA -> Disease mapping
    dic_miRNA_disease = process_db.load_miRNA_disease_dict(config['paths']['miRNA_dict_source'])
    dic_miRNA_disease = {k: v for k, v in dic_miRNA_disease.items() if k in unique_miRNAs}
    
    for diseases in dic_miRNA_disease.values():
        for i in range(len(diseases)):
            diseases[i] = standardize_disease_name(diseases[i])

    # Load Gene -> Disease mapping
    dic_gene_disease = process_db.load_gene_disease_dict_csv(config['paths']['gene_dict_source'])
    dic_gene_disease = {k: v for k, v in dic_gene_disease.items() if k in unique_genes}
    
    for gene, diseases in dic_gene_disease.items():
        dic_gene_disease[gene] = [standardize_disease_name(d) for d in diseases]

    return dic_miRNA_disease, dic_gene_disease

# ==========================================
# 3. Feature Extraction Logic
# ==========================================

def calc_sim_stats(embeddings_1, embeddings_2):
    if len(embeddings_1) == 0 or len(embeddings_2) == 0:
        return 0.0, 0.0, 0.0, 0.0

    tensor_1 = torch.tensor(embeddings_1)
    tensor_2 = torch.tensor(embeddings_2)
    
    sim_matrix = util.dot_score(tensor_1, tensor_2).numpy()
    sim_matrix = np.clip(sim_matrix, 0, 1)
    
    flat_sim = sim_matrix.flatten()
    
    mean_sim = float(np.mean(flat_sim))
    max_sim  = float(np.max(flat_sim))
    std_sim  = float(np.std(flat_sim))
    skew_sim = float(skew(flat_sim)) if len(flat_sim) > 1 else 0.0
    
    return mean_sim, max_sim, std_sim, skew_sim

def generate_features(df: pd.DataFrame, dic_disease_emb: dict, 
                      dic_miRNA: dict, dic_gene: dict) -> np.ndarray:
    print("Generating features...")
    
    feature_list = []
    embedding_dim = 384
    zero_vec = np.zeros(embedding_dim)

    for _, row in df.iterrows():
        lst_disease_A = dic_miRNA.get(row['miRNA'], [])
        lst_disease_B = dic_gene.get(row['gene'], [])

        embs_A = np.array([dic_disease_emb.get(item, zero_vec) for item in lst_disease_A if item in dic_disease_emb])
        embs_B = np.array([dic_disease_emb.get(item, zero_vec) for item in lst_disease_B if item in dic_disease_emb])

        if len(embs_A) == 0 or len(embs_B) == 0:
            feature_list.append([0.0] * 12)
            continue

        stats_AB = calc_sim_stats(embs_A, embs_B)
        stats_AA = calc_sim_stats(embs_A, embs_A)
        stats_BB = calc_sim_stats(embs_B, embs_B)

        row_feats = list(stats_AB) + list(stats_AA) + list(stats_BB)
        feature_list.append(row_feats)

    X = np.array(feature_list)
    X = np.nan_to_num(X)
    return X

# ==========================================
# 4. Bagging PU Classifier Definition
# ==========================================

class BaggingPUClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=50, random_state=42):
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeClassifier()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators = []
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        pos_indices = np.where(y == 1)[0]
        unlabeled_indices = np.where(y == 0)[0]
        
        X_pos = X[pos_indices]
        X_unlabeled = X[unlabeled_indices]
        
        n_pos = len(X_pos)
        n_unlabeled_sample = n_pos 
        
        self.estimators = []
        np.random.seed(self.random_state)
        
        for i in range(self.n_estimators):
            sample_indices = np.random.choice(len(X_unlabeled), n_unlabeled_sample, replace=True)
            X_unlabeled_subset = X_unlabeled[sample_indices]
            
            X_train_bag = np.vstack((X_pos, X_unlabeled_subset))
            y_train_bag = np.hstack((np.ones(n_pos), np.zeros(n_unlabeled_sample)))
            
            clf = clone(self.base_estimator)
            if hasattr(clf, 'random_state'):
                 clf.set_params(random_state = self.random_state + i)
            
            clf.fit(X_train_bag, y_train_bag)
            self.estimators.append(clf)
            
        return self
    
    def predict_proba(self, X):
        sum_proba = np.zeros(len(X))
        for clf in self.estimators:
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(X)
                if prob.shape[1] == 2:
                    sum_proba += prob[:, 1]
                else:
                    sum_proba += prob[:, 0] if clf.classes_[0] == 1 else 0
            
        avg_proba = sum_proba / self.n_estimators
        return np.vstack((1 - avg_proba, avg_proba)).T
    
    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)

# ==========================================
# 5. Pipeline Logic (Model Definition & Evaluation)
# ==========================================

def get_models(config: dict) -> dict:
    params = config['model_params']
    seed = config['splitting']['random_state']

    models = {
        "PU_DecisionTree": BaggingPUClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=params['tree_depth']), 
            n_estimators=params['bagging_n_estimators'],
            random_state=seed
        ),
        "PU_RandomForest": BaggingPUClassifier(
            base_estimator=RandomForestClassifier(n_estimators=params['n_estimators_rf'], 
                                                  max_depth=params['tree_depth'], 
                                                  n_jobs=-1), 
            n_estimators=20, 
            random_state=seed
        ),
        "PU_XGBoost": BaggingPUClassifier(
            base_estimator=XGBClassifier(eval_metric='logloss', 
                                         max_depth=6, 
                                         n_jobs=1),
            n_estimators=30,
            random_state=seed
        ),
        "PU_LogisticReg": BaggingPUClassifier(
            base_estimator=LogisticRegression(solver='liblinear'),
            n_estimators=params['bagging_n_estimators'],
            random_state=seed
        ),
        "Naive_XGBoost": XGBClassifier(eval_metric='logloss', 
                                       max_depth=6, 
                                       n_estimators=params['n_estimators_xgb'],
                                       random_state=seed)
    }
    return models

def evaluate_single_model(model, X_eval, y_eval, model_name, dataset_name):
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    
    acc = accuracy_score(y_eval, y_pred)
    auc = roc_auc_score(y_eval, y_prob)
    f1 = f1_score(y_eval, y_pred)
    prec = precision_score(y_eval, y_pred, zero_division=0)
    rec = recall_score(y_eval, y_pred, zero_division=0)
    
    print(f"--- {model_name} on {dataset_name} Set ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    return acc, auc, f1

def split_data_by_group(X, y, groups, test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits data based on groups (miRNA IDs) to prevent biological leakage.
    Ensures that a specific miRNA (and all its related samples) exists 
    solely in Train, OR Test, OR Validation.
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    # Shuffle the unique groups (miRNAs)
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_groups)
    
    # Calculate cutoff counts for groups
    n_val_groups = int(n_groups * val_size)
    n_test_groups = int(n_groups * test_size)
    
    # Partition the groups
    val_groups_set = set(unique_groups[:n_val_groups])
    test_groups_set = set(unique_groups[n_val_groups : n_val_groups + n_test_groups])
    train_groups_set = set(unique_groups[n_val_groups + n_test_groups:])
    
    print(f"Splitting by miRNA Groups: Train={len(train_groups_set)}, Test={len(test_groups_set)}, Val={len(val_groups_set)}")
    
    # Create masking based on the original groups array
    # Using list comprehension for masking to ensure order preservation
    mask_train = np.array([g in train_groups_set for g in groups])
    mask_test = np.array([g in test_groups_set for g in groups])
    mask_ext = np.array([g in val_groups_set for g in groups])
    
    return (X[mask_train], X[mask_test], X[mask_ext],
            y[mask_train], y[mask_test], y[mask_ext])

def run_training_pipeline(X, y, groups, config):
    """
    Orchestrates the splitting (Grouped by miRNA), scaling, training, and evaluation.
    """
    # 1. Split Data by miRNA Group
    print("Splitting data by miRNA groups...")
    ext_val_ratio = config['splitting']['external_val_size']
    test_ratio = config['splitting']['test_size']
    rand_seed = config['splitting']['random_state']
    
    X_train, X_test, X_ext, y_train, y_test, y_ext = split_data_by_group(
        X, y, groups, 
        test_size=test_ratio, 
        val_size=ext_val_ratio, 
        random_state=rand_seed
    )

    # 2. Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_ext_scaled = scaler.transform(X_ext)
    
    print(f"Train samples: {X_train_scaled.shape[0]}, Test samples: {X_test_scaled.shape[0]}, Ext Val samples: {X_ext_scaled.shape[0]}")

    # 3. Model Training & Eval
    models_dict = get_models(config)
    results = []

    print("\n--- Starting Model Training ---")
    for name, model in models_dict.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Internal Evaluation
        evaluate_single_model(model, X_test_scaled, y_test, name, "Internal Test (Unseen miRNAs)")
        
        # External Evaluation
        acc, auc, f1 = evaluate_single_model(model, X_ext_scaled, y_ext, name, "External Validation (Unseen miRNAs)")
        
        results.append({
            "Model": name,
            "Ext_AUC": auc,
            "Ext_F1": f1
        })
    
    return results

# ==========================================
# 6. Main Execution
# ==========================================

def main():
    start_time = time.time()
    
    # 1. Load Config
    config = load_config('config/miRTarDS_PU_Learning.json')

    # 2. Load Raw Data
    print("Loading MTI data...")
    df_MTI = pd.read_csv(config['paths']['MTI_data'], sep='\t')
    if 'label' not in df_MTI.columns:
        print("Warning: 'label' column missing.")
        return

    # 3. Load Embeddings & Dicts
    print("Loading Embeddings...")
    dic_disease_embedding = load_embeddings(config['paths']['disease_embedding'])
    
    print("Loading Dictionaries...")
    lst_miRNA = df_MTI['miRNA'].to_list()
    lst_gene = df_MTI['gene'].to_list()
    dic_miRNA_disease, dic_gene_disease = load_dictionaries(config, lst_miRNA, lst_gene)

    # 4. Generate Features (X) and Labels (y)
    X = generate_features(df_MTI, dic_disease_embedding, dic_miRNA_disease, dic_gene_disease)
    y = df_MTI['label'].values
    
    # Extract Group Info (miRNA IDs) for splitting
    # Assumes df_MTI['miRNA'] contains IDs like 'hsa-miR-124'
    groups = df_MTI['miRNA'].values 
    
    print(f"Data Prep Complete. X: {X.shape}, y: {y.shape}")

    # 5. Run Training Pipeline (with Groups)
    results = run_training_pipeline(X, y, groups, config)

    # 6. Final Summary
    print("\n" + "="*40)
    print("FINAL SUMMARY (External Validation Set - Group Split)")
    print("="*40)
    summary_df = pd.DataFrame(results).sort_values(by="Ext_AUC", ascending=False)
    print(summary_df)

    # 7. Time check
    end_time = time.time()
    runtime = end_time - start_time
    hours, remainder = divmod(runtime, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'\nTotal Running Time: {int(hours)}h, {int(minutes)}m, {seconds:.2f}s')

if __name__ == '__main__':
    main()