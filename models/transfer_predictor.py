"""Transfer prediction model using Random Forest (TabPFN fallback)."""
import pandas as pd
import numpy as np
# Using RandomForest instead of TabPFN due to CPU performance issues
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pickle
import os
from datetime import datetime
from typing import Dict, Tuple, Optional
from config import settings
from utils.data_extraction import extract_transfer_training_data, extract_transfer_prediction_candidates
from utils.feature_engineering import (
    prepare_transfer_features,
    calculate_recommended_quantity,
    calculate_ml_priority,
    assign_priority_label
)


class TransferPredictor:
    """ML model for predicting transfer success using TabPFN."""

    def __init__(self):
        """Initialize the transfer predictor."""
        self.model = None
        self.feature_columns = None
        self.model_version = None
        self.training_date = None
        self.metrics = {}

    def train(self, days_back: int = 90) -> Dict:
        """
        Train the transfer prediction model on historical data.

        Args:
            days_back: Number of days of historical data to use

        Returns:
            Dictionary with training metrics
        """

        print(f"Extracting training data (last {days_back} days)...")
        df = extract_transfer_training_data(days_back=days_back)

        if len(df) < settings.min_training_samples:
            raise ValueError(
                f"Insufficient training data: {len(df)} samples "
                f"(minimum {settings.min_training_samples} required)"
            )

        print(f"Preparing features from {len(df)} samples...")
        X, y = prepare_transfer_features(df)

        # Store feature columns for prediction
        self.feature_columns = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=settings.test_size,
            random_state=settings.random_state,
            stratify=y
        )

        print(f"Training Random Forest model on {len(X_train)} samples...")
        print(f"Class distribution: {y_train.value_counts().to_dict()}")

        # Initialize and train Random Forest (faster and more reliable than TabPFN on CPU)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=settings.random_state,
            n_jobs=-1  # Use all CPU cores
        )
        self.model.fit(X_train.values, y_train.values)

        # Evaluate on test set
        print("Evaluating model...")
        y_pred = self.model.predict(X_test.values)
        y_proba = self.model.predict_proba(X_test.values)[:, 1]

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'positive_class_ratio': y.mean(),
        }

        self.training_date = datetime.now().isoformat()
        self.model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print("\nTraining complete!")
        print(f"Model version: {self.model_version}")
        print(f"Accuracy: {self.metrics['accuracy']:.3f}")
        print(f"Precision: {self.metrics['precision']:.3f}")
        print(f"Recall: {self.metrics['recall']:.3f}")
        print(f"ROC-AUC: {self.metrics['roc_auc']:.3f}")

        return self.metrics

    def predict_transfers(self, limit: int = 100) -> pd.DataFrame:
        """
        Generate transfer recommendations using the trained model.

        Args:
            limit: Maximum number of recommendations to return

        Returns:
            DataFrame with predictions and recommendations
        """

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        print(f"Extracting transfer candidates...")
        df = extract_transfer_prediction_candidates(limit=limit)

        if len(df) == 0:
            print("No transfer candidates found.")
            return pd.DataFrame()

        print(f"Preparing features for {len(df)} candidates...")
        X, _ = prepare_transfer_features(df)

        # Ensure same feature columns as training
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns]

        print("Generating predictions...")
        predictions = self.model.predict(X.values)
        probabilities = self.model.predict_proba(X.values)[:, 1]

        # Calculate recommended quantities
        recommended_qty = calculate_recommended_quantity(df, probabilities)

        # Calculate ML priority scores
        priority_scores = calculate_ml_priority(
            probabilities,
            df['margin_percent'],
            df['to_store_daily_sales'] - df['from_store_daily_sales']
        )
        priority_labels = assign_priority_label(priority_scores)

        # Build results DataFrame
        results = df.copy()
        results['ml_prediction'] = predictions
        results['success_probability'] = probabilities
        results['recommended_qty'] = recommended_qty
        results['ml_priority_score'] = priority_scores
        results['ml_priority'] = priority_labels
        results['model_version'] = self.model_version

        # Filter out low-confidence predictions
        results = results[results['success_probability'] >= 0.5]

        # Sort by priority score
        results = results.sort_values('ml_priority_score', ascending=False)

        print(f"Generated {len(results)} recommendations (filtered by confidence >= 0.5)")

        return results.head(limit)

    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained model to disk.

        Args:
            filepath: Optional custom filepath

        Returns:
            Path where model was saved
        """

        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        if filepath is None:
            os.makedirs(settings.model_cache_dir, exist_ok=True)
            filepath = os.path.join(
                settings.model_cache_dir,
                f"transfer_predictor_{self.model_version}.pkl"
            )

        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_version': self.model_version,
            'training_date': self.training_date,
            'metrics': self.metrics
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {filepath}")
        return filepath

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to saved model file
        """

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.model_version = model_data['model_version']
        self.training_date = model_data['training_date']
        self.metrics = model_data['metrics']

        print(f"Model loaded: {self.model_version}")
        print(f"Training date: {self.training_date}")
        print(f"Metrics: {self.metrics}")

    @classmethod
    def load_latest(cls) -> 'TransferPredictor':
        """
        Load the most recently trained model.

        Returns:
            TransferPredictor instance with loaded model
        """

        model_dir = settings.model_cache_dir

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Find all model files
        model_files = [f for f in os.listdir(model_dir) if f.startswith('transfer_predictor_')]

        if not model_files:
            raise FileNotFoundError("No trained models found")

        # Get the latest model
        latest_model = max(model_files)
        filepath = os.path.join(model_dir, latest_model)

        predictor = cls()
        predictor.load_model(filepath)

        return predictor
