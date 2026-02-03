import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path, target):
    df = pd.read_csv(path)
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y

def split_data(X, y, task="classification"):
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if task=="classification" else None
    )

def encode_for_tabpfn(X):
    X_enc = X.copy()
    for col in X_enc.select_dtypes(include="object"):
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
    return X_enc
