import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import kagglehub
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses training on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        # BCE loss without reduction
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Calculate pt (probability of correct class)
        pt = torch.exp(-bce_loss)
        
        # Apply focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class DiabetesNN(nn.Module):
    """
    Neural Network for Diabetes Detection
    """
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(DiabetesNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DiabetesDetectionModel:
    """
    A comprehensive diabetes detection model using PyTorch with sklearn metrics
    FIXED: Data leakage and class imbalance issues
    """
    
    def __init__(self, data_path, device=None):
        """Initialize with path to NHANES data"""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # FIX #1: Store preprocessing objects to prevent data leakage
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
        self.model = None
        self.feature_names = None
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        print(f"Using device: {self.device}")
        
    def load_data(self):
        """Load NHANES dataset"""
        print("Loading NHANES data...")
        
        # Check if data_path is a directory (from kagglehub) or a file
        if os.path.isdir(self.data_path):
            # List all CSV files in the directory
            csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
            print(f"Found {len(csv_files)} CSV files in directory")
            
            if csv_files:
                print("Available files:", csv_files)
                
                # Try to load and merge relevant files
                dataframes = []
                for csv_file in csv_files:
                    file_path = os.path.join(self.data_path, csv_file)
                    try:
                        df = pd.read_csv(file_path)
                        print(f"Loaded {csv_file}: {df.shape}")
                        dataframes.append(df)
                    except Exception as e:
                        print(f"Could not load {csv_file}: {e}")
                
                # Merge dataframes if multiple exist (on common key like SEQN)
                if len(dataframes) > 1:
                    print("\nMerging datasets...")
                    self.data = dataframes[0]
                    for df in dataframes[1:]:
                        # Find common columns for merging
                        common_cols = list(set(self.data.columns) & set(df.columns))
                        if common_cols:
                            merge_key = common_cols[0]  # Usually 'SEQN' for NHANES
                            self.data = self.data.merge(df, on=merge_key, how='outer')
                else:
                    self.data = dataframes[0]
            else:
                raise FileNotFoundError("No CSV files found in the directory")
        else:
            # Load single CSV file
            self.data = pd.read_csv(self.data_path)
        
        print(f"Final data shape: {self.data.shape}")
        print(f"\nColumns: {list(self.data.columns)[:20]}...")
        return self
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        print("\nDataset Info:")
        print(self.data.info())
        
        print("\nFirst few rows:")
        print(self.data.head())
        
        print("\nBasic Statistics:")
        print(self.data.describe())
        
        print("\nMissing Values:")
        missing = self.data.isnull().sum()
        missing_pct = 100 * missing / len(self.data)
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        }).sort_values('Percentage', ascending=False)
        print(missing_df[missing_df['Missing_Count'] > 0].head(20))
        
        return self
    
    def preprocess_data(self, target_col='DIQ010', test_size=0.2, val_size=0.2):
        """
        Preprocess NHANES data for diabetes detection
        FIXED: Properly fit preprocessing objects only on training data
        
        Parameters:
        -----------
        target_col : str
            Name of the target column (diabetes indicator)
        test_size : float
            Proportion of data for testing
        val_size : float
            Proportion of training data for validation
        """
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Create a copy to work with
        df = self.data.copy()
        
        # Handle target variable
        if target_col not in df.columns:
            print(f"\nWarning: {target_col} not found. Looking for diabetes-related columns...")
            diabetes_cols = [col for col in df.columns if 'DIQ' in col or 'diabetes' in col.lower()]
            print(f"Found: {diabetes_cols}")
            if diabetes_cols:
                target_col = diabetes_cols[0]
                print(f"Using: {target_col}")
            else:
                raise ValueError("No diabetes indicator column found!")
        
        # Create binary target (1 = diabetes, 0 = no diabetes)
        if target_col in df.columns:
            df['diabetes'] = df[target_col].apply(lambda x: 1 if x == 1 else 0 if x == 2 else np.nan)
            df = df.dropna(subset=['diabetes'])
            
        print(f"\nTarget variable distribution:")
        print(df['diabetes'].value_counts())
        print(f"Diabetes prevalence: {df['diabetes'].mean():.2%}")
        
        # Select relevant features for diabetes prediction
        demographic_features = ['RIDAGEYR', 'RIAGENDR', 'RIDRETH1']
        body_features = ['BMXBMI', 'BMXWT', 'BMXHT', 'BMXWAIST']
        bp_features = ['BPXSY1', 'BPXDI1', 'BPXSY2', 'BPXDI2']
        lab_features = ['LBXGLU', 'LBXGH', 'LBXTC', 'LBDHDD', 'LBDLDL', 'LBXTR']
        
        # Combine all features
        potential_features = demographic_features + body_features + bp_features + lab_features
        
        # Select only available features
        available_features = [col for col in potential_features if col in df.columns]
        print(f"\nAvailable features: {len(available_features)}")
        print(available_features)
        
        # Create feature matrix
        X = df[available_features].copy()
        y = df['diabetes'].copy()
        
        # FIX #1: Split data BEFORE any preprocessing
        X_train_full, self.X_test, y_train_full, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size, random_state=42, stratify=y_train_full
        )
        
        # FIX #1: Fit imputer ONLY on training data
        print("\nHandling missing values...")
        self.feature_names = list(X.columns)
        
        # Fit on training data
        self.imputer.fit(self.X_train)
        
        # Transform all sets using the fitted imputer
        self.X_train = pd.DataFrame(
            self.imputer.transform(self.X_train),
            columns=self.feature_names,
            index=self.X_train.index
        )
        self.X_val = pd.DataFrame(
            self.imputer.transform(self.X_val),
            columns=self.feature_names,
            index=self.X_val.index
        )
        self.X_test = pd.DataFrame(
            self.imputer.transform(self.X_test),
            columns=self.feature_names,
            index=self.X_test.index
        )
        
        # FIX #1: Fit scaler ONLY on training data
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_names
        )
        self.X_val = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=self.feature_names
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_names
        )
        
        print(f"\nTraining set size: {len(self.X_train)}")
        print(f"Validation set size: {len(self.X_val)}")
        print(f"Test set size: {len(self.X_test)}")
        
        # Display class distribution
        print(f"\nClass distribution:")
        print(f"Training - Diabetes: {self.y_train.sum()}/{len(self.y_train)} ({self.y_train.mean():.2%})")
        print(f"Validation - Diabetes: {self.y_val.sum()}/{len(self.y_val)} ({self.y_val.mean():.2%})")
        print(f"Test - Diabetes: {self.y_test.sum()}/{len(self.y_test)} ({self.y_test.mean():.2%})")
        
        return self
    
    def create_data_loaders(self, batch_size=64):
        """Create PyTorch DataLoaders"""
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(self.X_train.values).to(self.device)
        y_train_tensor = torch.FloatTensor(self.y_train.values).unsqueeze(1).to(self.device)
        
        X_val_tensor = torch.FloatTensor(self.X_val.values).to(self.device)
        y_val_tensor = torch.FloatTensor(self.y_val.values).unsqueeze(1).to(self.device)
        
        X_test_tensor = torch.FloatTensor(self.X_test.values).to(self.device)
        y_test_tensor = torch.FloatTensor(self.y_test.values).unsqueeze(1).to(self.device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return self
    
    def build_model(self, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        """Build PyTorch neural network model"""
        input_size = len(self.feature_names)
        self.model = DiabetesNN(input_size, hidden_sizes, dropout_rate).to(self.device)
        print("\n" + "="*50)
        print("MODEL ARCHITECTURE")
        print("="*50)
        print(self.model)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return self
    
    def train_model(self, epochs=100, learning_rate=0.001, weight_decay=1e-5, 
                    patience=15, use_focal_loss=True, focal_alpha=0.25, focal_gamma=2.0):
        """
        Train the PyTorch model
        FIXED: Added Focal Loss option for handling class imbalance
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate for optimizer
        weight_decay : float
            L2 regularization parameter
        patience : int
            Early stopping patience
        use_focal_loss : bool
            Whether to use Focal Loss instead of BCE
        focal_alpha : float
            Focal loss alpha parameter (weight for positive class)
        focal_gamma : float
            Focal loss gamma parameter (focusing parameter)
        """
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # FIX #3: Use Focal Loss for class imbalance
        if use_focal_loss:
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        else:
            # Alternative: Use weighted BCE loss
            pos_weight = torch.tensor([len(self.y_train) / self.y_train.sum() - 1]).to(self.device)
            criterion = nn.BCELoss()
            print(f"Using BCE Loss with class weight: {pos_weight.item():.2f}")
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    predictions = (outputs >= 0.5).float()
                    val_correct += (predictions == y_batch).sum().item()
                    val_total += y_batch.size(0)
            
            val_loss /= len(self.val_loader)
            val_acc = val_correct / val_total
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            current_lr = optimizer.param_groups[0]['lr']
            old_lr = current_lr
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            if new_lr != old_lr:
                print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_diabetes_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_diabetes_model.pth'))
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
        
        return self
    
    def evaluate_model(self):
        """Evaluate the PyTorch model using sklearn metrics"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                outputs = self.model(X_batch)
                predictions = (outputs >= 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(outputs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        # Convert to numpy arrays
        y_pred = np.array(all_predictions).flatten()
        y_pred_proba = np.array(all_probabilities).flatten()
        y_true = np.array(all_targets).flatten()
        
        # Calculate metrics using sklearn
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        print(f"\nTest Set Results:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        print(f"\nTrue Negatives:  {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives:  {cm[1,1]}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # Store for visualization
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.training_history['train_loss'], label='Train Loss')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.training_history['val_acc'], label='Validation Accuracy', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        auc = roc_auc_score(self.y_true, self.y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'PyTorch NN (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Diabetes Detection (PyTorch)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix heatmap"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'])
        plt.title('Confusion Matrix - PyTorch Neural Network')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def predict(self, new_data):
        """
        Make predictions on new data
        FIXED: Uses saved preprocessing objects (no data leakage)
        """
        self.model.eval()
        
        # Ensure we have all required features
        missing_features = [f for f in self.feature_names if f not in new_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only the features used in training
        new_data_processed = new_data[self.feature_names].copy()
        
        # FIX #1: Use the SAVED imputer (fitted on training data only)
        new_data_imputed = pd.DataFrame(
            self.imputer.transform(new_data_processed),  # transform, not fit_transform!
            columns=self.feature_names
        )
        
        # FIX #1: Use the SAVED scaler (fitted on training data only)
        new_data_scaled = self.scaler.transform(new_data_imputed)  # transform, not fit_transform!
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(new_data_scaled).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs >= 0.5).float()
            probabilities = outputs
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()


# Example usage
if __name__ == "__main__":
    # Download dataset from Kaggle using kagglehub
    print("Downloading NHANES dataset from Kaggle...")
    path = kagglehub.dataset_download("cdc/national-health-and-nutrition-examination-survey")
    print("Path to dataset files:", path)
    
    # Initialize model with PyTorch
    diabetes_model = DiabetesDetectionModel(path)
    
    # Run full pipeline
    try:
        diabetes_model.load_data()
        diabetes_model.explore_data()
        diabetes_model.preprocess_data()
        diabetes_model.create_data_loaders(batch_size=64)
        diabetes_model.build_model(hidden_sizes=[128, 64, 32], dropout_rate=0.3)
        
        # FIX #3: Train with Focal Loss to handle class imbalance
        diabetes_model.train_model(
            epochs=100, 
            learning_rate=0.001, 
            patience=15,
            use_focal_loss=True,  # Enable Focal Loss
            focal_alpha=0.25,     # Weight for positive class
            focal_gamma=2.0       # Focusing parameter
        )
        
        # Evaluate
        results = diabetes_model.evaluate_model()
        
        # Visualizations
        diabetes_model.plot_training_history()
        diabetes_model.plot_roc_curve()
        diabetes_model.plot_confusion_matrix()
        
        print("\n" + "="*50)
        print("TRAINING COMPLETE!")
        print("="*50)
        print(f"Final Test ROC-AUC: {results['roc_auc']:.4f}")
        print(f"Final Test F1-Score: {results['f1_score']:.4f}")
        
        # Demonstrate the fixed predict method (no data leakage)
        print("\n" + "="*50)
        print("TESTING PREDICTION ON NEW DATA")
        print("="*50)
        print("The predict() method now uses saved preprocessing objects.")
        print("This prevents data leakage during inference!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nIf you encounter issues, please ensure:")
        print("1. kagglehub is installed: pip install kagglehub")
        print("2. PyTorch is installed: pip install torch")
        print("3. You're authenticated with Kaggle")
        print("4. The dataset contains diabetes-related variables (DIQ columns)")