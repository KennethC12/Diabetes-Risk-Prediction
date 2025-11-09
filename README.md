# Diabetes Detection System

A production-ready machine learning system for diabetes risk prediction using clinical and behavioral health indicators. Built with interpretability, calibration, and clinical deployment in mind.

## 🎯 Project Overview

This system predicts diabetes risk using two complementary datasets:
- **Pima Indians Diabetes Database**: Clinical measurements (glucose, BMI, insulin, etc.)
- **CDC BRFSS Dataset**: Large-scale behavioral risk factor surveillance data

The pipeline emphasizes:
- ✅ **Clinical safety**: Calibrated probabilities, configurable decision thresholds
- ✅ **Interpretability**: SHAP explanations for model predictions
- ✅ **Production-ready**: FastAPI service + Streamlit demo interface
- ✅ **Robustness**: Extensive testing, preprocessing leak prevention, stratified validation

## 📊 Key Features

- **Multiple model support**: Logistic Regression baseline + XGBoost/LightGBM boosted trees
- **Class imbalance handling**: SMOTE oversampling and class-weighted training
- **Threshold optimization**: Maximize recall at precision ≥ 0.70 or F-beta scoring
- **Probability calibration**: Post-hoc calibration for reliable risk estimates
- **Fairness evaluation**: Performance slices by age and sex subgroups
- **Full explainability**: SHAP force plots and feature importance visualizations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/diabetes-detection.git
cd diabetes-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
# Create data directory
mkdir -p data/raw

# Download Pima dataset (automatic via script)
python data/downloader.py --dataset pima

# Download BRFSS from Kaggle (requires Kaggle API credentials)
# Place kaggle.json in ~/.kaggle/
python data/downloader.py --dataset brfss
```

Alternative: Manually download datasets and place in `data/raw/`:
- Pima: https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv
- BRFSS: Kaggle BRFSS 2015 dataset

### Train Your First Model

```bash
# Train baseline logistic regression on Pima dataset
python diabetes_app.py train --config configs/pima.yaml

# Evaluate performance metrics
python diabetes_app.py evaluate --config configs/pima.yaml

# Pick optimal decision threshold
python diabetes_app.py pick-threshold --target-precision 0.70
```

### Launch Services

```bash
# Start FastAPI prediction server
uvicorn serve_api:app --reload --port 8000

# In another terminal, launch Streamlit UI
streamlit run app_dashboard.py
```

Access the dashboard at `http://localhost:8501`

## 📁 Repository Structure

```
diabetes-detection/
├── data/                         # Data loading and validation
│   ├── downloader.py            # Automated dataset downloads
│   └── __init__.py
│
├── features/                     # Feature engineering
│   ├── preprocess.py            # Imputation, scaling, encoding
│   ├── engineer.py              # Feature creation (bins, interactions)
│   └── __init__.py
│
├── models/                       # Model training & evaluation
│   ├── train.py                 # Training pipeline with CV
│   ├── evaluate.py              # Metrics, ROC/PR curves
│   ├── threshold.py             # Decision threshold optimization
│   ├── calibrate.py             # Probability calibration
│   ├── registry.py              # Model persistence
│   └── __init__.py
│
├── explain/                      # Model interpretability
│   ├── shap_utils.py            # SHAP visualizations
│   ├── reports.py               # Model/data cards
│   └── __init__.py
│
├── utils/                        # Shared utilities
│   ├── io.py                    # File I/O, logging
│   ├── metrics.py               # Custom metric functions
│   └── __init__.py
│
├── notebooks/                    # Exploratory analysis
│   ├── 01_eda.ipynb             # Data exploration
│   ├── 02_baseline.ipynb        # Baseline models
│   └── 03_tree_boosting.ipynb   # Advanced models
│
├── tests/                        # Test suite
│   ├── test_loaders.py
│   ├── test_preprocess.py
│   ├── test_train.py
│   └── test_threshold.py
│
├── configs/                      # Configuration files
│   ├── pima.yaml                # Pima dataset config
│   └── brfss.yaml               # BRFSS dataset config
│
├── artifacts/                    # Generated artifacts (gitignored)
│   ├── models/                  # Saved .pkl models
│   └── figures/                 # Plots and visualizations
│
├── diabetes_app.py              # CLI application
├── serve_api.py                 # FastAPI server
├── app_dashboard.py             # Streamlit UI
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 📈 Expected Performance

### Pima Indians Dataset (768 samples)
- **Logistic Regression**: ROC-AUC ≈ 0.78–0.82, PR-AUC ≈ 0.65–0.70
- **XGBoost/LightGBM**: ROC-AUC ≈ 0.82–0.86, PR-AUC ≈ 0.70–0.75
- **Key features**: Glucose, BMI, Age, Diabetes Pedigree Function

### BRFSS Dataset (200K+ samples)
- **Improved generalization** due to larger sample size
- **Better calibration** on held-out test sets
- **Behavioral features**: Physical activity, diet, healthcare access

## 🧪 Testing

Run the full test suite:

```bash
# All tests
pytest tests/ -v

# Specific test modules
pytest tests/test_loaders.py
pytest tests/test_preprocess.py
pytest tests/test_train.py

# With coverage report
pytest tests/ --cov=. --cov-report=html
```

## 🔍 Model Interpretability

Generate SHAP explanations:

```bash
# Generate SHAP summary plot
python -m explain.shap_utils --model artifacts/models/xgboost_pima.pkl \
                              --data data/processed/pima_test.csv \
                              --output artifacts/figures/shap_summary.png

# Generate model card
python -m explain.reports --model artifacts/models/xgboost_pima.pkl \
                          --output artifacts/model_card.md
```

## 📝 Development Workflow

### Phase 1: Data Foundation
1. Implement `data/downloader.py` for automated dataset retrieval
2. Create data validation reports (missingness, outliers, class balance)

### Phase 2: Feature Engineering
1. Build preprocessing pipelines (imputation, scaling)
2. Engineer features (BMI bins, glucose×BMI interactions)

### Phase 3: Model Training
1. Train baseline logistic regression
2. Implement XGBoost/LightGBM with early stopping
3. Add probability calibration

### Phase 4: Deployment
1. Create FastAPI prediction service
2. Build Streamlit demo interface

### Phase 5: Explainability
1. Generate SHAP visualizations
2. Create model and data cards

### Phase 6: Testing & CI
1. Write unit tests for all modules
2. Set up GitHub Actions for automated testing

## 📄 License

MIT License - see LICENSE file for details

## 📚 References

- Pima Indians Diabetes Database: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes)
- CDC BRFSS: [Behavioral Risk Factor Surveillance System](https://www.cdc.gov/brfss/)
- SHAP: [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)

---

**Built with ❤️ for responsible AI in healthcare**