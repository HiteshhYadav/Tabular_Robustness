import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models import FTTransformerConfig, TabTransformerConfig

from utils.seed import set_seed
from utils.preprocessing import load_data
from evaluation.metrics import classification_metrics

set_seed(42)

DATA_PATH = "data/raw/adult.csv"
TARGET = "income"

X, y = load_data(DATA_PATH, TARGET)
y = LabelEncoder().fit_transform(y.astype(str))

df = X.copy()
df[TARGET] = y

# Covariate shift
train_df = df[df["age"] < 45]
test_df  = df[df["age"] >= 45]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

data_config = DataConfig(
    target=[TARGET],
    categorical_cols=cat_cols,
    continuous_cols=num_cols
)

trainer_config = TrainerConfig(
    max_epochs=20,
    batch_size=1024,
    accelerator="cpu",
    devices=1,
    seed=42,
    trainer_kwargs={
        "enable_progress_bar": False,
        "enable_model_summary": False
    }
)

optimizer_config = OptimizerConfig()
results = []

for name, cfg in {
    "FT-Transformer": FTTransformerConfig(task="classification"),
    "TabTransformer": TabTransformerConfig(task="classification")
}.items():

    model = TabularModel(
        data_config=data_config,
        model_config=cfg,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config
    )

    model.fit(train=train_df)
    pred = model.predict(test_df)

    metrics = classification_metrics(
        test_df[TARGET],
        (pred["prediction"] > 0.5).astype(int),
        pred["prediction"]
    )
    metrics["model"] = name
    results.append(metrics)

pd.DataFrame(results).to_csv(
    "results/tables/transformers_shift.csv", index=False
)

print(pd.DataFrame(results))
