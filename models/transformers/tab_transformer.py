from pytorch_tabular.models import TabTransformerConfig

def get_tab_transformer(task):
    return TabTransformerConfig(
        task=task,
        learning_rate=1e-3,
        seed=42
    )
