import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from utils.seed import set_seed
from utils.preprocessing import load_data, split_data, encode_for_tabpfn
from utils.perturbations import add_noise

from models.gbdt.lightgbm_model import get_lgbm
from models.gbdt.catboost_model import get_catboost
from models.tabpfn.tabpfn_model import get_tabpfn

from evaluation.metrics import classification_metrics
from evaluation.robustness import degradation

set_seed(42)

DATA_PATH = "data/raw/adult.csv"
TARGET = "income"
TASK = "classification"



X, y = load_data(DATA_PATH, TARGET)

num_cols = X.select_dtypes(exclude="object").columns

# Encode categorical features (SAME as baseline)
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Ensure target is pandas Series with aligned index
y = pd.Series(y, index=X.index)
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y.astype(str))
y = pd.Series(y, index=X.index)

X_train, X_test, y_train, y_test = split_data(X, y, TASK)



baseline = pd.read_csv("results/tables/baseline.csv")

noise_levels = [0.1, 0.3, 0.5]
results = []

for std in noise_levels:
    X_test_miss = add_noise(X_test, num_cols, std)

    # -------- LightGBM --------
    model = get_lgbm(TASK)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test_miss)[:, 1]
    pred = (prob > 0.5).astype(int)

    score = classification_metrics(y_test, pred, prob)["roc_auc"]
    base  = baseline[baseline.model=="LightGBM"]["roc_auc"].values[0]

    results.append({
        "model": "LightGBM",
        "noise_std": std,
        "roc_auc": score,
        "degradation": degradation(base, score)
    })

    # -------- CatBoost --------
    model = get_catboost(TASK)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test_miss)[:, 1]
    pred = (prob > 0.5).astype(int)

    score = classification_metrics(y_test, pred, prob)["roc_auc"]
    base  = baseline[baseline.model=="CatBoost"]["roc_auc"].values[0]

    results.append({
        "model": "CatBoost",
        "noise_std": std,
        "roc_auc": score,
        "degradation": degradation(base, score)
    })

    # -------- TabPFN --------

    # Use SAME numeric encoding as baseline
    X_train_pfn = X_train.copy()
    X_test_pfn  = X_test_miss.copy()

    MAX_TABPFN = 5000

    # SAME subsample as baseline logic
    if len(X_train_pfn) > MAX_TABPFN:
        X_train_pfn = X_train_pfn.sample(MAX_TABPFN, random_state=42)
        y_train_pfn = y_train.loc[X_train_pfn.index]
    else:
        y_train_pfn = y_train

    model = get_tabpfn()
    model.fit(X_train_pfn.values, y_train_pfn.values)

    prob = model.predict_proba(X_test_pfn.values)[:, 1]
    pred = (prob > 0.5).astype(int)

    score = classification_metrics(y_test, pred, prob)["roc_auc"]
    base  = baseline[baseline.model == "TabPFN"]["roc_auc"].values[0]

    results.append({
        "model": "TabPFN",
        "noise_std": std,
        "roc_auc": score,
        "degradation": degradation(base, score)
    })


df = pd.DataFrame(results)
df.to_csv("results/tables/noise.csv", index=False)
print(df)
