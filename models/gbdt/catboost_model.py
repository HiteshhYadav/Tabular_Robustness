from catboost import CatBoostClassifier, CatBoostRegressor

def get_catboost(task="classification"):
    if task == "classification":
        return CatBoostClassifier(
            iterations=300, depth=8,
            learning_rate=0.05,
            random_seed=42, verbose=False
        )
    else:
        return CatBoostRegressor(
            iterations=300, depth=8,
            learning_rate=0.05,
            random_seed=42, verbose=False
        )
