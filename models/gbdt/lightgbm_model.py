import lightgbm as lgb

def get_lgbm(task="classification"):
    if task == "classification":
        return lgb.LGBMClassifier(
            n_estimators=300, max_depth=8,
            learning_rate=0.05, random_state=42
        )
    else:
        return lgb.LGBMRegressor(
            n_estimators=300, max_depth=8,
            learning_rate=0.05, random_state=42
        )

