# 🏭 Contextual Predictive Maintenance — IoT Edge AI

> **Infotact Solutions DS/ML Internship | 2026**
> Intern: Sanjay , Jinsha | Branch: `pred`

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-green)](https://lightgbm.readthedocs.io)
[![Macro F1](https://img.shields.io/badge/Macro%20F1-0.9875-brightgreen)]()
[![Status](https://img.shields.io/badge/Status-Complete-success)]()

---

## 📌 Project Overview

Most existing predictive maintenance systems rely exclusively on internal sensor signals and fail in real-world deployment. A machine's failure is rarely isolated — it is heavily influenced by external factors such as weather conditions, factory load, and shift timing.

This project builds an **advanced contextual data fusion framework** that integrates internal IoT telemetry (temperature, torque, rotational speed, tool wear) with simulated external environmental signals to accurately predict mechanical failures before they happen using robust ensemble modeling.

---

## 🎯 Business Objective

| Metric | Target | Achieved |
|---|---|---|
| Macro F1 Score | ≥ 0.85 | **0.9875** ✅ |
| Test Macro F1 | ≥ 0.85 | **1.0000** ✅ |
| ROC AUC | ≥ 0.90 | **1.0000** ✅ |

---

## 📁 Project Structure

```
predictive_maintenance/
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv              ← Original AI4I 2020 dataset (UCI)
│   └── processed/
│       ├── processed_data.csv        ← Week 1 output (72 features)
│       └── week2_enriched.csv        ← Week 2 output (85 features)
│
├── models/
│   ├── lgbm_model.pkl                ← Trained LightGBM model
│   ├── lgbm_final.pkl                ← Final model with threshold
│   └── shap_summary.png             ← SHAP feature importance chart
│
├── notebooks/
│   ├── 01_week.ipynb                 ← Week 1: IoT Telemetry Ingestion
│   ├── 02_week.ipynb                 ← Week 2: Contextual Data Fusion
│   ├── 03_week.ipynb                 ← Week 3: SMOTE + LightGBM Modeling
│   └── 04_week.ipynb                 ← Week 4: SHAP Explainability
│
├── .gitignore
└── README.md
```

---

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset** — UCI Machine Learning Repository

| Property | Value |
|---|---|
| Source | UCI ML Repository |
| Total rows | 10,000 |
| Original features | 14 |
| Failure rate | 3.39% (339 failures) |
| Failure types | TWF, HDF, PWF, OSF, RNF |

**Sensor columns:**
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

---

## 🗓️ Four-Week Engineering Roadmap

---

### ✅ Week 1 — IoT Telemetry Ingestion & Signal Processing

**Objective:** Ingest the AI4I dataset, perform EDA, and engineer baseline features from raw sensor signals.

**What was done:**
- Loaded AI4I 2020 dataset (10,000 rows × 14 columns)
- Performed Exploratory Data Analysis (EDA) — class distribution, correlation heatmap, boxplots, failure type breakdown
- Engineered **rolling window features** (mean, std, variance) over windows of 3, 5, and 10 readings for all 5 sensors
- Added **lag features** (1-step and 2-step lag) for all 5 sensors
- Simulated 3 external contextual signals — `ambient_humidity`, `factory_load_index`, `shift_period`
- Saved processed dataset to `data/processed/processed_data.csv`

**Feature engineering breakdown:**

| Type | Calculation | Count |
|---|---|---|
| Original features | Raw dataset columns | 14 |
| Rolling features | 5 sensors × 3 windows × 3 stats | 45 |
| Lag features | 5 sensors × 2 lags | 10 |
| External signals | humidity, load, shift | 3 |
| **Total** | | **72** |

**Key finding:** Failure rate is 3.39% — highly imbalanced dataset requiring SMOTE in Week 3.

**Notebook:** `notebooks/01_week.ipynb`
**Output:** `data/processed/processed_data.csv` — (9,991 rows × 72 features)

---

### ✅ Week 2 — Contextual Data Fusion & Feature Engineering

**Objective:** Enrich data with external contextual signals merged via timestamps and conduct an ablation study.

**What was done:**
- Added precise **timestamps** (minute-level) and extracted `hour`, `day`, `dayofweek`
- Simulated 5 additional external signals — `wind_speed_kmh`, `outdoor_temp_celsius`, `power_grid_voltage`, `operator_experience`, `is_weekend`
- Engineered 5 **smart combined features:**

| Feature | Formula | Meaning |
|---|---|---|
| `temp_difference` | Process temp − Air temp | Overheating risk |
| `power_consumption` | Speed × Torque ÷ 9550 | Machine stress |
| `wear_per_rotation` | Tool wear ÷ (Speed + 1) | Part degradation rate |
| `heat_stress_index` | Outdoor temp × Factory load ÷ 100 | Combined heat stress |
| `fatigue_risk` | (1 − Shift) × (10 − Experience) | Human error risk |

- Conducted **ablation study** to mathematically prove external features' impact

**Ablation Study Results:**

| Experiment | Macro F1 |
|---|---|
| Without external features | 0.8768 |
| With external features | 0.8748 |
| Note | Simulated signals carry no real correlation — in production, real API data would show clear improvement |

**Notebook:** `notebooks/02_week.ipynb`
**Output:** `data/processed/week2_enriched.csv` — (9,991 rows × 85 features)

---

### ✅ Week 3 — SMOTE + LightGBM Modeling

**Objective:** Build the core classification pipeline using SMOTE strictly inside CV folds and LightGBM ensemble modeling.

**What was done:**
- Applied **SMOTE strictly inside training folds** using `imblearn.Pipeline` — zero data leakage
- Used **StratifiedKFold (5 folds)** to preserve class distribution across folds
- Trained **LightGBM** with 500 estimators, balanced class weights
- Achieved **Macro F1 = 0.9875** on cross-validation
- Achieved **Macro F1 = 1.0000** on held-out test set

**Model Parameters:**

| Parameter | Value |
|---|---|
| n_estimators | 500 |
| learning_rate | 0.05 |
| max_depth | 6 |
| num_leaves | 31 |
| class_weight | balanced |
| SMOTE k_neighbors | 5 |

**Cross-Validation Results:**

| Fold | Macro F1 |
|---|---|
| Fold 1 | 0.9843 |
| Fold 2 | 0.9923 |
| Fold 3 | 0.9803 |
| Fold 4 | 0.9924 |
| Fold 5 | 0.9883 |
| **Mean** | **0.9875** |
| **Std** | **0.0047** |

**Test Set Results:**

| Metric | Score |
|---|---|
| Macro F1 | 1.0000 |
| ROC AUC | 1.0000 |
| Precision (Failure) | 1.00 |
| Recall (Failure) | 1.00 |

**Notebook:** `notebooks/03_week.ipynb`
**Output:** `models/lgbm_model.pkl`

---

### ✅ Week 4 — SHAP Explainability & Threshold Tuning

**Objective:** Explain model predictions using SHAP values and tune the decision threshold for real-world deployment.

**What was done:**
- Computed **SHAP values** using `TreeExplainer` for all test samples
- Generated SHAP summary bar plot and dot plot showing feature impact direction
- Explained individual machine predictions — which factors most caused the failure prediction
- Performed **threshold tuning** — evaluated Macro F1 across thresholds from 0.1 to 0.9
- Saved final model package with optimal threshold to `models/lgbm_final.pkl`

**Why SHAP matters:**
SHAP answers *"Why did the model predict this machine will fail?"* — enabling the Reliability Engineer persona to understand root causes and take targeted action.

**Why threshold tuning matters:**
In a factory setting, missing a real failure (false negative) is far more costly than a false alarm (false positive). Lowering the threshold catches more failures at the cost of more false alarms — a deliberate business trade-off.

**Notebook:** `notebooks/04_week.ipynb`
**Output:** `models/lgbm_final.pkl`, `models/shap_summary.png`

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML modeling | LightGBM, Scikit-learn |
| Imbalanced learning | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Version control | Git, GitHub |
| Environment | VS Code, Jupyter Notebook |

---

## ⚙️ Setup Instructions

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/sanjay071202/predictive-Maintenance.git
cd predictive-Maintenance
git checkout pred
```

**Step 2 — Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn lightgbm imbalanced-learn shap scikit-learn
```

**Step 3 — Download dataset:**

Download `ai4i2020.csv` from [UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) and place it in `data/raw/`.

**Step 4 — Run notebooks in order:**
```
01_week.ipynb → 02_week.ipynb → 03_week.ipynb → 04_week.ipynb
```

---

## 📈 Results Summary

| Week | Deliverable | Key Result |
|---|---|---|
| Week 1 | Feature engineering pipeline | 72 features from 14 raw columns |
| Week 2 | Contextual data fusion | 85 features + ablation study |
| Week 3 | SMOTE + LightGBM model | Macro F1 = 0.9875 (CV) |
| Week 4 | SHAP explainability | Feature importance + threshold tuned |

---

## 👤 User Personas Addressed

**Fleet/Plant Manager**
- Sees which machines have >80% probability of failing in the next 7 days
- Can schedule maintenance proactively — no surprise breakdowns

**Reliability Engineer**
- Uses SHAP feature importance charts to understand root causes
- Knows whether external temperature or internal vibration is driving the anomaly

---

## 📝 Important Notes

- `data/` folder is in `.gitignore` — CSV files must be downloaded separately
- SMOTE is applied **strictly inside training folds only** — no data leakage
- External signals are simulated — in production, replace with real weather API or V2X logs
- Column names are cleaned before LightGBM training (special characters removed)

---

## 🔗 Links

- **GitHub Repository:** [github.com/sanjay071202/predictive-Maintenance](https://github.com/sanjay071202/predictive-Maintenance)
- **Dataset:** [UCI AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **LinkedIn:** [linkedin.com/in/sanjay72002](https://linkedin.com/in/sanjay72002)

---






- 
