import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom preprocessor to handle both numeric and categorical features.
    """
    def __init__(self, 
                 numeric_features: Optional[list] = None, 
                 categorical_features: Optional[list] = None):
        """
        Initialize the preprocessor with feature types.
        
        Args:
            numeric_features (list): List of numeric column names
            categorical_features (list): List of categorical column names
        """
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.label_encoders_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit method to prepare preprocessing transformations.
        
        Args:
            X (pd.DataFrame): Input dataframe
            y (pd.Series, optional): Target variable
        
        Returns:
            self: Fitted preprocessor
        """
        # Prepare label encoders for categorical features
        for col in self.categorical_features:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders_[col] = le
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the input data.
        
        Args:
            X (pd.DataFrame): Input dataframe
        
        Returns:
            np.ndarray: Transformed features
        """
        X_copy = X.copy()
        
        # Encode categorical features
        for col, le in self.label_encoders_.items():
            X_copy[col] = le.transform(X_copy[col].astype(str))
        
        return X_copy.values

class LoanDefaultModel:
    """
    Advanced Loan Default Prediction Model with multiple classifier support.
    """
    SUPPORTED_MODELS = {
        'logistic': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier
    }

    def __init__(self, 
                 model_type: str = 'logistic', 
                 random_state: int = 42):
        """
        Initialize the loan default model.
        
        Args:
            model_type (str): Type of model to use
            random_state (int): Random seed for reproducibility
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type. Choose from {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.preprocessor = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from various file types with error handling.
        
        Args:
            filepath (str): Path to the data file
        
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            file_extension = filepath.split('.')[-1].lower()
            
            if file_extension == 'xlsx':
                return pd.read_excel(filepath)
            elif file_extension == 'csv':
                return pd.read_csv(filepath)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def preprocess_data(self, 
                         data: pd.DataFrame, 
                         target_column: str = 'loan_status') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Comprehensive data preprocessing with flexible feature handling.
        
        Args:
            data (pd.DataFrame): Input dataframe
            target_column (str): Name of the target variable column
        
        Returns:
            Tuple of preprocessed features and target
        """
        # Drop unnecessary columns
        columns_to_drop = ['customer_id', 'transaction_date']
        data_cleaned = data.drop(columns=columns_to_drop, errors='ignore')

        # Identify feature types
        numeric_features = data_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = data_cleaned.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column from features
        if target_column in numeric_features:
            numeric_features.remove(target_column)
        if target_column in categorical_features:
            categorical_features.remove(target_column)

        # Prepare preprocessor
        self.preprocessor = DataPreprocessor(
            numeric_features=numeric_features, 
            categorical_features=categorical_features
        )

        # Separate features and target
        X = data_cleaned.drop(columns=[target_column], errors='ignore')
        y = data_cleaned[target_column] if target_column in data_cleaned.columns else None

        # Fit and transform
        X_transformed = self.preprocessor.fit_transform(X)

        return X_transformed, y

    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray) -> None:
        """
        Train the specified model with advanced configuration.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
        """
        model_class = self.SUPPORTED_MODELS[self.model_type]
        
        model_params = {
            'random_state': self.random_state
        }

        if self.model_type == 'logistic':
            model_params.update({
                'max_iter': 1000,  # Increased iterations
                'solver': 'liblinear'  # Good for smaller datasets
            })
        elif self.model_type == 'random_forest':
            model_params.update({
                'n_estimators': 200,  # Increased number of trees
                'max_depth': 10
            })
        elif self.model_type == 'gradient_boosting':
            model_params.update({
                'n_estimators': 200,
                'learning_rate': 0.1
            })

        self.model = model_class(**model_params)
        self.model.fit(X_train, y_train)

    def evaluate(self, 
                 X_test: np.ndarray, 
                 y_test: np.ndarray) -> Dict[str, Any]:
        """
        Advanced model evaluation with multiple metrics.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
        
        Returns:
            Dict of performance metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        y_pred = self.model.predict(X_test)
        
        # Comprehensive evaluation
        metrics = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, 
            X_test, 
            y_test, 
            cv=5, 
            scoring='accuracy'
        )
        metrics['cross_validation_scores'] = cv_scores

        # Pretty print results
        print("Model Evaluation Metrics:")
        print("Classification Report:\n", metrics['classification_report'])
        print("Confusion Matrix:\n", metrics['confusion_matrix'])
        print(f"Cross-Validation Scores: Mean = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}")

        return metrics

def main():
    """
    Main function to demonstrate model usage.
    """
    # Create model instances
    models = {
        'Logistic Regression': LoanDefaultModel(model_type='logistic'),
        'Random Forest': LoanDefaultModel(model_type='random_forest'),
        'Gradient Boosting': LoanDefaultModel(model_type='gradient_boosting')
    }

    # Load and preprocess data
    for name, model in models.items():
        print(f"\n{name} Model Training:")
        try:
            # Adjust paths as necessary
            train_data = model.load_data('train_data.xlsx')
            test_data = model.load_data('test_data.xlsx')

            X_train, y_train = model.preprocess_data(train_data)
            X_test, y_test = model.preprocess_data(test_data)

            # Train and evaluate
            model.train(X_train, y_train)
            model.evaluate(X_test, y_test)

        except Exception as e:
            print(f"Error in {name} model: {e}")

if __name__ == "__main__":
    main()