import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from utils.seed import set_seed
from utils.preprocessing import load_data

from models.gbdt.lightgbm_model import get_lgbm
from models.gbdt.catboost_model import get_catboost
from models.tabpfn.tabpfn_model import get_tabpfn

from evaluation.metrics import classification_metrics

set_seed(42)

DATA_PATH = "data/raw/adult.csv"
TARGET = "income"
TASK = "classification"

# ======================
# LOAD DATA
# ======================
X, y = load_data(DATA_PATH, TARGET)

# ======================
# ENCODE CATEGORICAL FEATURES (SAME AS BASELINE)
# ======================
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y.astype(str))
y = pd.Series(y, index=X.index)

# ======================
# COVARIATE SHIFT SPLIT
# ======================
# Train on younger population, test on older population
train_idx = X["age"] < 45
test_idx  = X["age"] >= 45

X_train = X.loc[train_idx]
y_train = y.loc[train_idx]

X_test  = X.loc[test_idx]
y_test  = y.loc[test_idx]

results = []

# ======================
# LightGBM
# ======================
model = get_lgbm(TASK)
model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
pred = (prob > 0.5).astype(int)

metrics = classification_metrics(y_test, pred, prob)
metrics["model"] = "LightGBM"
results.append(metrics)

# ======================
# CatBoost
# ======================
model = get_catboost(TASK)
model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
pred = (prob > 0.5).astype(int)

metrics = classification_metrics(y_test, pred, prob)
metrics["model"] = "CatBoost"
results.append(metrics)

# ======================
# TabPFN
# ======================
X_train_pfn = X_train.copy()
X_test_pfn  = X_test.copy()

MAX_TABPFN = 5000

if len(X_train_pfn) > MAX_TABPFN:
    X_train_pfn = X_train_pfn.sample(MAX_TABPFN, random_state=42)
    y_train_pfn = y_train.loc[X_train_pfn.index]
else:
    y_train_pfn = y_train

model = get_tabpfn()
model.fit(X_train_pfn.values, y_train_pfn.values)

prob = model.predict_proba(X_test_pfn.values)[:, 1]
pred = (prob > 0.5).astype(int)

metrics = classification_metrics(y_test, pred, prob)
metrics["model"] = "TabPFN"
results.append(metrics)

# ======================
# SAVE RESULTS
# ======================
df = pd.DataFrame(results)
df.to_csv("results/tables/shift.csv", index=False)

print(df)
