from pytorch_tabular.models import FTTransformerConfig

def get_ft_transformer(task):
    return FTTransformerConfig(
        task=task,
        learning_rate=1e-3,
        seed=42
    )
