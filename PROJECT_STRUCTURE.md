# Diabetes Detection – Project Organization

## Repository Layout

```
diabetes-detection/
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies (TODO: Create)
├── setup.py                      # Package setup (optional/TODO)
│
├── diabetes_app.py               # CLI entrypoint (train/eval/predict) (TODO)
├── serve_api.py                  # FastAPI inference server (TODO)
├── app_dashboard.py              # Streamlit demo UI (TODO)
│
├── data/                         # Data I/O and validation
│   ├── __init__.py               # TODO
│   ├── downloader.py                # Load Pima/BRFSS datasets (TODO)
│
├── features/                     # Preprocessing & feature engineering
│   ├── __init__.py               # TODO
│   ├── preprocess.py             # Impute/scale/encode pipelines (TODO)
│   └── engineer.py               # Binning, interactions, flags (TODO)
│
├── models/                       # Model training & persistence
│   ├── __init__.py               # TODO
│   ├── train.py                  # Train with CV/SMOTE/weights (TODO)
│   ├── evaluate.py               # Metrics (ROC/PR/F1), curves (TODO)
│   ├── threshold.py              # Pick operating point (recall@precision) (TODO)
│   ├── calibrate.py              # Probability calibration (TODO)
│   └── registry.py               # Save/load artifacts (.pkl) (TODO)
│
├── explain/                      # Interpretability
│   ├── __init__.py               # TODO
│   ├── shap_utils.py             # SHAP summary/force plots (TODO)
│   └── reports.py                # Model & data cards (markdown) (TODO)
│
├── notebooks/                    # EDA & experiments
│   ├── 01_eda.ipynb              # Distributions, missingness, imbalance (TODO)
│   ├── 02_baseline.ipynb         # Logistic baseline (TODO)
│   └── 03_tree_boosting.ipynb    # XGBoost/LightGBM (TODO)
│
├── tests/                        # Test suites
│   ├── __init__.py               # TODO
│   ├── test_loaders.py           # Dataset loading & schema tests (TODO)
│   ├── test_preprocess.py        # Imputation/scaling leakage tests (TODO)
│   ├── test_train.py             # Reproducible training, CV shapes (TODO)
│   └── test_threshold.py         # Thresholding logic (TODO)
│
├── configs/                      # YAML configs for runs
│   ├── pima.yaml                 # Columns, metrics, params (TODO)
│   └── brfss.yaml                # Larger dataset config (TODO)
│
├── artifacts/                    # Saved models/plots (gitignored)
│   ├── models/                   # .pkl files
│   └── figures/                  # ROC/PR/Calibration/SHAP
│
└── utils/                        # Utilities
    ├── __init__.py               # TODO
    ├── io.py                     # Paths, saving, logging (TODO)
    └── metrics.py                # Custom metrics (PR-AUC, recall@prec) (TODO)
```

## Implementation Roadmap

### Phase 1: Data Foundation

- `data/loaders.py`: load Pima (`Outcome`) and BRFSS (`Diabetes_binary`) datasets, replace biologically impossible zeros (`Glucose`, `BP`, `BMI`, `Insulin`, `SkinThickness`) with `NaN`, and create stratified train/val/test splits.
- `data/validate.py`: generate reports on missingness, outliers, class balance, and feature ranges.

### Phase 2: Features & Pipelines

- `features/preprocess.py`: build preprocessing pipelines with median imputation, scaling (e.g., `StandardScaler`), and optional class balancing (class weights vs. SMOTE).
- `features/engineer.py`: add engineered features including age and BMI bins, `Glucose × BMI` interaction, and risk flag indicators.

### Phase 3: Modeling Core

- `models/train.py`: train baseline `LogisticRegression(class_weight='balanced')` and boosted models (XGBoost/LightGBM) with early stopping; log PR-AUC and ROC-AUC.
- `models/calibrate.py`: calibrate probabilities via `CalibratedClassifierCV` or isotonic/sigmoid methods on validation splits.
- `models/threshold.py`: select decision thresholds maximizing Recall at Precision ≥ 0.70 or optimizing `Fβ` (β = 2).
- `models/evaluate.py`: produce confusion matrix, ROC/PR curves, calibration curve, Brier score, and fairness slices (age/sex).

### Phase 4: Serving & Demo

- `serve_api.py`: FastAPI service exposing `/predict` endpoint returning probabilities, thresholded labels, and optional feature attributions.
- `app_dashboard.py`: Streamlit app with form inputs (Age, BMI, Glucose, etc.) displaying predicted risk, explanations, and disclaimers.

### Phase 5: Explainability & Reports

- `explain/shap_utils.py`: utilities for SHAP summary and force plots.
- `explain/reports.py`: generate markdown model and data cards for artifacts.

### Phase 6: Testing & CI

- Unit tests covering data loading, preprocessing leakage, training determinism, and threshold logic (`tests/`).
- Optional GitHub Actions workflow to run `pytest` and cache datasets.

## Dependencies

Add the following to `requirements.txt`:

```
pandas>=1.5
numpy>=1.21
scikit-learn>=1.2
imbalanced-learn>=0.11
xgboost>=2.0
lightgbm>=4.0
matplotlib>=3.6
seaborn>=0.12
shap>=0.44
fastapi>=0.110
uvicorn>=0.29
pyyaml>=6.0
pytest>=7.0
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`.
2. Download data:
   - Pima: https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv
   - BRFSS: Kaggle BRFSS 2015 dataset.
   - Place raw files under `data/raw/` (gitignored).
3. Run initial EDA: execute `notebooks/01_eda.ipynb`.
4. Train a model: `python diabetes_app.py train --config configs/pima.yaml`.
5. Evaluate & set threshold:
   - `python diabetes_app.py evaluate --config configs/pima.yaml`
   - `python diabetes_app.py pick-threshold --target-precision 0.70`
6. Serve API: `uvicorn serve_api:app --reload --port 8000`.
7. Launch Streamlit UI: `streamlit run app_dashboard.py`.

## Expected Outcomes

- Baseline logistic regression ROC-AUC ≈ 0.78–0.82 on Pima, with strong PR-AUC baseline.
- Boosted trees lift PR-AUC and improve recall at moderate precision.
- Calibrated probabilities enable better clinical decision thresholds.
- SHAP explainability highlights `Glucose`, `BMI`, and `Age` as key drivers.

## Testing Strategy

- **Unit tests** (`tests/`):
  - `test_loaders.py`: column integrity, no leakage across splits.
  - `test_preprocess.py`: zero-to-NaN conversion and train-only scaling.
  - `test_train.py`: cross-validation shapes and seed reproducibility.
  - `test_threshold.py`: monotonic threshold trade-offs and stability.
- **Integration tests**: end-to-end train → evaluate → serve on a Pima subset.
- **Performance & robustness**: ensure PR-AUC resilience to noise and class imbalance shifts.

## Nice-to-Haves

- Cost-sensitive evaluation where false negatives carry higher penalties.
- Drift monitoring for population shifts.
- Persisted model/data cards in `artifacts/` for documentation.
