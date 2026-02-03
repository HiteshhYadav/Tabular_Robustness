import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from utils.seed import set_seed
from utils.preprocessing import load_data, split_data, encode_for_tabpfn

from models.gbdt.lightgbm_model import get_lgbm
from models.gbdt.catboost_model import get_catboost
from models.tabpfn.tabpfn_model import get_tabpfn

from evaluation.metrics import classification_metrics

set_seed(42)

# ======================
# CONFIG
# ======================
DATA_PATH = "data/raw/adult.csv"
TARGET = "income"
TASK = "classification"

# ======================
# LOAD DATA
# ======================
X, y = load_data(DATA_PATH, TARGET)

# Encode categorical features before splitting
from sklearn.preprocessing import LabelEncoder
encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y.astype(str))
y = pd.Series(y, index=X.index)

X_train, X_test, y_train, y_test = split_data(X, y, TASK)

results = []

# ======================
# LightGBM
# ======================
lgbm = get_lgbm(TASK)
lgbm.fit(X_train, y_train)
pred = lgbm.predict(X_test)
prob = lgbm.predict_proba(X_test)[:, 1]

metrics = classification_metrics(y_test, pred, prob)
metrics["model"] = "LightGBM"
results.append(metrics)

# ======================
# CatBoost
# ======================
cat = get_catboost(TASK)
cat.fit(X_train, y_train)
pred = cat.predict(X_test)
prob = cat.predict_proba(X_test)[:, 1]

metrics = classification_metrics(y_test, pred, prob)
metrics["model"] = "CatBoost"
results.append(metrics)

# ======================
# TabPFN (numeric only)
# ======================

X_train_pfn = X_train.copy()
X_test_pfn  = X_test.copy()

MAX_TABPFN = 5000

if len(X_train_pfn) > MAX_TABPFN:
    X_train_pfn = X_train_pfn.sample(MAX_TABPFN, random_state=42)
    y_train_pfn = y_train[X_train_pfn.index]
else:
    y_train_pfn = y_train

tabpfn = get_tabpfn()
tabpfn.fit(X_train_pfn.values, y_train_pfn.values)

prob = tabpfn.predict_proba(X_test_pfn.values)[:, 1]
pred = (prob > 0.5).astype(int)

metrics = classification_metrics(y_test, pred, prob)
metrics["model"] = "TabPFN"
results.append(metrics)



# ======================
# SAVE
# ======================
df = pd.DataFrame(results)
df.to_csv("results/tables/baseline.csv", index=False)

print(df)
