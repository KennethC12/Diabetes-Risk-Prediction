# Diabetes Detection System

A production-ready deep learning system for diabetes risk prediction using NHANES clinical health data. Built with PyTorch neural networks, emphasizing interpretability, clinical deployment, and ethical AI practices.

## 🎯 Project Overview

This system predicts diabetes risk using the **National Health and Nutrition Examination Survey (NHANES)** dataset, which includes:
- Clinical measurements (HbA1c, glucose, BMI, blood pressure)
- Body composition metrics (weight, height, waist circumference)
- Lipid panels (total cholesterol, HDL, LDL, triglycerides)
- Demographic information (age, gender, race/ethnicity)

The pipeline emphasizes:
- ✅ **Deep Learning**: PyTorch neural networks with batch normalization and dropout
- ✅ **Clinical Safety**: sklearn metrics for comprehensive evaluation
- ✅ **Robustness**: Early stopping, learning rate scheduling, overfitting detection
- ✅ **Reproducibility**: Automatic Kaggle dataset download via kagglehub
- ✅ **Transparency**: Training history visualization and performance analysis

## 🏆 Model Performance

### PyTorch Neural Network (Best Model)
- **Architecture**: 16 → 128 → 64 → 32 → 1 (with BatchNorm + Dropout)
- **Test Accuracy**: 96.19%
- **ROC-AUC**: 0.9487
- **Precision**: 81.36%
- **Recall**: 65.31%
- **F1-Score**: 0.7245

**Comparison with sklearn baselines:**
- Outperforms Random Forest (ROC-AUC: 0.9471)
- Outperforms Gradient Boosting (ROC-AUC: 0.9456)
- Outperforms Logistic Regression (ROC-AUC: 0.9428)

**Key predictive features:**
1. HbA1c (LBXGH) - 42.8% importance
2. Age (RIDAGEYR) - 8.6%
3. Waist Circumference (BMXWAIST) - 5.6%
4. Total Cholesterol (LBXTC) - 5.5%
5. BMI (BMXBMI) - 4.6%

## 📊 Key Features

### Deep Learning Architecture
- **Multi-layer perceptron** with 3 hidden layers
- **Batch Normalization** for stable training
- **Dropout (30%)** for regularization
- **ReLU activation** for non-linearity
- **Sigmoid output** for probability estimates
- **12,993 trainable parameters**

### Training Optimization
- **Adam optimizer** with weight decay (L2 regularization)
- **Learning rate scheduling** (ReduceLROnPlateau)
- **Early stopping** with patience monitoring
- **Class imbalance handling** for 7.7% diabetes prevalence
- **Stratified train/val/test splits** (64%/16%/20%)

### Evaluation & Monitoring
- **sklearn metrics**: Accuracy, precision, recall, F1, ROC-AUC
- **Confusion matrix** visualization
- **ROC curves** with AUC scores
- **Training history** plots (loss and accuracy)
- **Overfitting analysis** with automated warnings

### Data Processing
- **Automatic Kaggle download** via kagglehub
- **Multi-file merging** (demographic, examination, laboratory, questionnaire)
- **Missing value imputation** (median strategy)
- **Feature scaling** (StandardScaler with mean=0, std=1)
- **Mini-batch training** (batch_size=64)

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

### Required Dependencies

```txt
# Core ML libraries
torch>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0

# Data & visualization
kagglehub>=0.2.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Optional (for extended features)
shap>=0.42.0  # For model interpretability
```

### Train Your First Model

```python
import kagglehub

# Download NHANES dataset (automatic)
path = kagglehub.dataset_download(
    "cdc/national-health-and-nutrition-examination-survey"
)

# Initialize model
from diabetes_detection import DiabetesDetectionModel

model = DiabetesDetectionModel(path)

# Full training pipeline
model.load_data()
model.explore_data()
model.preprocess_data()
model.create_data_loaders(batch_size=64)
model.build_model(hidden_sizes=[128, 64, 32], dropout_rate=0.3)
model.train_model(epochs=100, learning_rate=0.001, patience=15)

# Evaluate
results = model.evaluate_model()

# Visualizations
model.plot_training_history()
model.plot_roc_curve()
model.plot_confusion_matrix()
```

## 🧠 Model Architecture Explained

### Layer-by-Layer Breakdown

```python
DiabetesNN(
  Input: 16 features
    ↓
  Linear(16 → 128)        # 2,176 parameters
  BatchNorm1d(128)        # Normalize activations
  ReLU()                  # Non-linearity
  Dropout(30%)            # Regularization
    ↓
  Linear(128 → 64)        # 8,256 parameters
  BatchNorm1d(64)
  ReLU()
  Dropout(30%)
    ↓
  Linear(64 → 32)         # 2,080 parameters
  BatchNorm1d(32)
  ReLU()
  Dropout(30%)
    ↓
  Linear(32 → 1)          # 33 parameters
  Sigmoid()               # Output probability [0, 1]
)
```

**Total: 12,993 trainable parameters**

### Why This Architecture?

1. **Funnel shape** (16→128→64→32→1): Expands to capture complexity, then compresses to essentials
2. **Batch normalization**: Stabilizes training, allows higher learning rates
3. **ReLU activation**: Fast, avoids vanishing gradients
4. **30% dropout**: Prevents overfitting on medical data
5. **GPU-ready**: Automatically uses CUDA if available

## 📈 Training Results

### Typical Training Run

```
Epoch [1/100]   - Train Loss: 0.4100, Val Loss: 0.2704, Val Acc: 0.9485
Epoch [10/100]  - Train Loss: 0.1316, Val Loss: 0.1265, Val Acc: 0.9543
Learning rate reduced from 0.001000 to 0.000500
Epoch [20/100]  - Train Loss: 0.1202, Val Loss: 0.1270, Val Acc: 0.9537
Learning rate reduced from 0.000500 to 0.000250
Early stopping triggered at epoch 26

Training completed! Best validation loss: 0.1248

==================================================
OVERFITTING ANALYSIS
==================================================
Final Training Loss:   0.1202
Final Validation Loss: 0.1270
Loss Gap:              0.0068 (+5.65%)

✅ GOOD: No significant overfitting detected
✅ Training and validation accuracies are well-balanced
✅ Validation loss improved or stabilized (good generalization)
```

### Performance Metrics

```
Confusion Matrix:
[[1747   22]  ← 98.8% of healthy correctly identified
 [  51   96]]  ← 65.3% of diabetic correctly identified

Classification Report:
              precision    recall  f1-score   support
         0.0       0.97      0.99      0.98      1769
         1.0       0.81      0.65      0.72       147

    accuracy                           0.96      1916
```

## 🔍 Data Processing Pipeline

### 1. Automatic Data Download
```python
# Kagglehub handles authentication automatically
path = kagglehub.dataset_download(
    "cdc/national-health-and-nutrition-examination-survey"
)
# Downloads and caches locally for future runs
```

### 2. Multi-File Merging
```python
# Automatically loads and merges:
# - demographic.csv (age, gender, race)
# - examination.csv (BMI, BP, body measurements)
# - laboratory.csv (HbA1c, glucose, lipids)
# - questionnaire.csv (diabetes diagnosis)
# Merged on SEQN (sequence number)
```

### 3. Feature Selection
```python
# 16 carefully selected features:
# Demographics: RIDAGEYR, RIAGENDR, RIDRETH1
# Body: BMXBMI, BMXWT, BMXHT, BMXWAIST
# Blood Pressure: BPXSY1, BPXDI1, BPXSY2, BPXDI2
# Laboratory: LBXGH, LBXTC, LBDHDD, LBDLDL, LBXTR
```

### 4. Missing Value Imputation
```python
# Median imputation for robustness
imputer = SimpleImputer(strategy='median')
# Fits on training, transforms on validation/test
```

### 5. Feature Scaling
```python
# Standardization (mean=0, std=1)
scaler = StandardScaler()
# Critical for neural network convergence
```

### 6. Mini-Batch Loading
```python
# PyTorch DataLoaders with batch_size=64
# Shuffled training, sequential validation/test
# Automatic GPU transfer if available
```

## ⚠️ Ethical Considerations

### Use of Demographic Variables

This model includes **age, gender, and race/ethnicity** as features. Important notes:

1. **Race is a social construct**, not a biological category
2. Racial health disparities reflect **systemic inequalities**, not genetics
3. Race/ethnicity has **low feature importance** (~1-2%) compared to clinical markers (HbA1c: 42.8%)
4. Including race may help **identify underserved populations** for intervention
5. Alternative: Remove demographic features and rely solely on clinical measurements

### Recommendations for Deployment

- **Clinical oversight required** - This is a screening tool, not a diagnostic tool
- **Regular auditing** - Monitor for bias across demographic groups
- **Threshold tuning** - Adjust decision boundary based on clinical priorities (e.g., prioritize recall for early detection)
- **Explainability** - Provide feature importances and SHAP values for transparency
- **Human-in-the-loop** - Never replace healthcare professionals with automated decisions

## 🧪 Testing & Validation

### Overfitting Detection

The model includes automatic overfitting analysis:

```python
# Checks performed:
✅ Train-validation loss gap < 10%
✅ Train-validation accuracy gap < 5%
✅ Validation loss trend (improving vs. diverging)
✅ Visual inspection of training curves
```

### Cross-Validation (Future)

```python
# Planned: K-fold cross-validation
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Train 5 models, report mean ± std performance
```

## 🎨 Visualization Examples

### Training History
![Training curves showing loss and accuracy over epochs](artifacts/figures/training_history_example.png)

### ROC Curve
![ROC curve with AUC=0.9487](artifacts/figures/roc_curve_example.png)

### Confusion Matrix
![Heatmap showing prediction vs. actual](artifacts/figures/confusion_matrix_example.png)

## 🔧 Hyperparameter Tuning

### Quick Tuning Guide

```python
# More capacity (if underfitting)
model.build_model(hidden_sizes=[256, 128, 64], dropout_rate=0.2)

# More regularization (if overfitting)
model.build_model(hidden_sizes=[64, 32], dropout_rate=0.5)

# Faster training
model.train_model(learning_rate=0.01, batch_size=128)

# More thorough training
model.train_model(epochs=200, patience=30, learning_rate=0.0001)
```

### Grid Search (Advanced)

```python
# Iterate over hyperparameters
for hidden_sizes in [[64, 32], [128, 64, 32], [256, 128, 64]]:
    for dropout in [0.2, 0.3, 0.5]:
        for lr in [0.001, 0.0001]:
            # Train and compare validation ROC-AUC
            pass
```

## 📚 References

- **NHANES Dataset**: [CDC National Health and Nutrition Examination Survey](https://www.cdc.gov/nchs/nhanes/)
- **PyTorch Documentation**: [pytorch.org](https://pytorch.org/docs/stable/index.html)
- **Sklearn Metrics**: [scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
- **Diabetes Screening Guidelines**: [American Diabetes Association](https://diabetes.org/diabetes/risk-test)

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. **SHAP explanations** for model interpretability
2. **Threshold optimization** for clinical use cases
3. **Fairness metrics** across demographic subgroups
4. **FastAPI deployment** for production serving
5. **Streamlit dashboard** for interactive demo
6. **Unit tests** for all components
7. **CI/CD pipeline** with GitHub Actions

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- NHANES dataset provided by the CDC
- Built with PyTorch and scikit-learn
- Inspired by responsible AI practices in healthcare

---

**Built with ❤️ for ethical AI in healthcare**

*Note: This model is for research and educational purposes. Always consult healthcare professionals for medical decisions.*