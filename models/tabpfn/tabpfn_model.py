from tabpfn import TabPFNClassifier
import os

def get_tabpfn():
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    return TabPFNClassifier(
        device="cpu",
        ignore_pretraining_limits=True
    )