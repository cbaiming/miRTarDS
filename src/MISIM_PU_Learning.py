import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
# 2. Feature Extraction (Single Feature: MISIM_value)
# ==========================================

def extract_misim_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts only the 'MISIM_value' column as the feature set.
    Handles NaNs by replacing them with 0.0 to ensure model stability.
    """
    print("Extracting MISIM_value feature...")
    
    if 'MISIM_value' not in df.columns:
        raise ValueError("Error: 'MISIM_value' column missing in the dataframe.")

    # Extract column, coerce errors to NaN, then fill with 0
    misim_series = pd.to_numeric(df['MISIM_value'], errors='coerce').fillna(0.0)
    
    # Reshape to (n_samples, 1) for sklearn compatibility
    X = misim_series.values.reshape(-1, 1)
    
    return X

# ==========================================
# 3. Bagging PU Classifier Definition
# ==========================================

class BaggingPUClassifier(BaseEstimator, ClassifierMixin):
    """
    Generic Bagging PU Learner.
    Undersamples the Unlabeled (negative) class to match Positive class size
    in each bag iteration.
    ref: Mordelet, F., & Vert, J. P. (2014). A bagging SVM to learn from positive and unlabeled examples.
    """
    def __init__(self, base_estimator=None, n_estimators=50, random_state=42):
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeClassifier()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators = []
        self.classes_ = None
        
    def fit(self, X, y):
        # Identify classes
        self.classes_ = np.unique(y)
        
        pos_indices = np.where(y == 1)[0]
        unlabeled_indices = np.where(y == 0)[0]
        
        X_pos = X[pos_indices]
        X_unlabeled = X[unlabeled_indices]
        
        n_pos = len(X_pos)
        # We undersample the unlabeled class to equal the size of positive class
        n_unlabeled_sample = n_pos  
        
        self.estimators = []
        np.random.seed(self.random_state)
        
        for i in range(self.n_estimators):
            # Bootstrap sample from Unlabeled set
            sample_indices = np.random.choice(len(X_unlabeled), n_unlabeled_sample, replace=True)
            X_unlabeled_subset = X_unlabeled[sample_indices]
            
            # Combine Positive and sampled Unlabeled
            X_train_bag = np.vstack((X_pos, X_unlabeled_subset))
            y_train_bag = np.hstack((np.ones(n_pos), np.zeros(n_unlabeled_sample)))
            
            # Clone and Train Base Estimator
            clf = clone(self.base_estimator)
            
            # Set random state for reproducibility if estimator supports it
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
                # Handle cases where model output shape varies (e.g. if single class in bag, though unlikely here)
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
# 4. Pipeline Logic
# ==========================================

def get_models(config: dict) -> dict:
    """
    Returns the dictionary of PU models using parameters from config.
    """
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
    """
    Evaluates a model and prints key metrics.
    """
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
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    
    return acc, auc, f1

def run_training_pipeline(X, y, config):
    """
    Orchestrates splitting, scaling, training, and evaluation.
    """
    # 1. Split Data (Stratified)
    print("Splitting data...")
    # External Validation Split
    X_main, X_ext, y_main, y_ext = train_test_split(
        X, y, 
        test_size=config['splitting']['external_val_size'], 
        random_state=config['splitting']['random_state'], 
        stratify=y
    )
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_main, y_main, 
        test_size=config['splitting']['test_size'], 
        random_state=config['splitting']['random_state'], 
        stratify=y_main
    )

    # 2. Scaling
    # Even with a single feature, scaling helps convergence for Logistic Regression/SVC
    print("Scaling feature (MISIM_value)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_ext_scaled = scaler.transform(X_ext)
    
    print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}, Ext Val shape: {X_ext_scaled.shape}")

    # 3. Model Training & Eval
    models_dict = get_models(config)
    results = []

    print("\n--- Starting Model Training (Feature: MISIM_value) ---")
    for name, model in models_dict.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Internal Evaluation
        evaluate_single_model(model, X_test_scaled, y_test, name, "Internal Test")
        
        # External Evaluation
        acc, auc, f1 = evaluate_single_model(model, X_ext_scaled, y_ext, name, "External Validation")
        
        results.append({
            "Model": name,
            "Ext_AUC": auc,
            "Ext_F1": f1
        })
    
    return results

# ==========================================
# 5. Main Execution
# ==========================================

def main():
    start_time = time.time()
    
    # 1. Load Config
    # Ensure this path points to your existing config file
    config = load_config('config/MISIM_PU_Learning.json')

    # 2. Load Raw Data
    print("Loading MTI data...")
    df_MTI = pd.read_csv(config['paths']['MTI_data'], sep='\t')
    
    # Validation
    if 'label' not in df_MTI.columns:
        print("Error: 'label' column missing.")
        return

    # 3. Feature Generation (Only uses MISIM_value)
    X = extract_misim_features(df_MTI)
    y = df_MTI['label'].values
    
    print(f"Data Prep Complete. X: {X.shape}, y: {y.shape}")

    # 4. Run Training Pipeline
    results = run_training_pipeline(X, y, config)

    # 5. Final Summary
    print("\n" + "="*40)
    print("FINAL SUMMARY (External Validation Set - Single Feature)")
    print("="*40)
    summary_df = pd.DataFrame(results).sort_values(by="Ext_AUC", ascending=False)
    print(summary_df)

    # 6. Time check
    end_time = time.time()
    runtime = end_time - start_time
    hours, remainder = divmod(runtime, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'\nTotal Running Time: {int(hours)}h, {int(minutes)}m, {seconds:.2f}s')

if __name__ == '__main__':
    main()