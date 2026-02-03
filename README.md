# 📊 Tabular Robustness Benchmark

A systematic benchmark evaluating the robustness of tree-based, transformer-based, and foundation models for tabular data under missingness, noise, and distribution shift using real-world Kaggle-style datasets.

---

## 🔍 Overview

Recent advances in tabular machine learning have introduced transformer-based architectures and foundation models alongside traditional gradient-boosted decision trees (GBDTs). However, most prior work emphasizes predictive accuracy on curated datasets, leaving model reliability under real-world data corruptions largely unexplored.

This repository provides a reproducible experimental framework to analyze robustness and failure behavior of modern tabular models.

---

## 🎯 Research Objectives

- Establish baseline performance on clean datasets  
- Evaluate robustness under missing values, feature noise, and distribution shift  
- Quantify performance degradation across models  
- Compare reliability of GBDTs, transformers, and foundation models  

---

## 📂 Datasets

| Task | Dataset | Source |
|-----|--------|-------|
| Classification | Adult Income | UCI / Kaggle |
| Regression | House Prices | Kaggle |

---

## ⚙️ Experimental Scenarios

### Baseline  
Training and evaluation on clean data.

### Missingness  
Random feature removal at multiple rates (10%, 20%, 40%).

### Noise  
Gaussian noise injected into numerical features.

### Distribution Shift  
Covariate shift induced using subgroup-based splits.

---

## 📊 Evaluation Metrics

**Classification**
- ROC-AUC  
- F1-score  

**Regression**
- RMSE  
- R²  

**Robustness**
- Relative performance degradation

---

## 🧠 Models Implemented

- LightGBM  
- CatBoost  
- FT-Transformer  
- TabTransformer  
- TabPFN  


---

## 🚀 Running Experiments

Install dependencies:


pip install -r requirements.txt
Run baseline:

python experiments/baseline.py
Run robustness experiments:

python experiments/missingness.py
python experiments/noise.py
python experiments/shift.py
📈 Outputs
Saved in:

results/tables/
baseline.csv
missingness.csv
noise.csv
shift.csv

📚 Related Work
TabNet • FT-Transformer • TabTransformer • TabPFN

📌 Citation
Hitesh Yadav (2026)
Robustness of Tabular Machine Learning Models Under Real-World Data Corruptions
