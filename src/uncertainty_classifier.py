import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import argparse
from typing import List, Dict, Tuple
import os

class UncertaintyFeatureExtractor:
    def __init__(self):
        # Define yes/no variations (case-insensitive)
        self.yes_variations = {
            'yes', 'YES', 'Yes', 'YEs', 'yES', 'yEs', 'yeS', 'YeS',
            'true', 'TRUE', 'True', 'correct', 'CORRECT', 'Correct',
            '1', 'positive', 'POSITIVE', 'Positive'
        }
        
        self.no_variations = {
            'no', 'NO', 'No', 'nO',
            'false', 'FALSE', 'False', 'incorrect', 'INCORRECT', 'Incorrect',
            '0', 'negative', 'NEGATIVE', 'Negative'
        }
    
    def extract_features(self, top_probs: List[Dict]) -> Dict:
        """Extract comprehensive features from token probabilities"""
        if not top_probs or len(top_probs) == 0:
            return self._get_empty_features()
        
        # Handle case where top_probs is a dict with string keys (e.g., {'0': [...], '1': [...]})
        if isinstance(top_probs, dict):
            # Convert dict to list by sorting keys and extracting values
            sorted_keys = sorted(top_probs.keys(), key=lambda x: int(x) if x.isdigit() else 0)
            top_probs = [top_probs[key] for key in sorted_keys if isinstance(top_probs[key], list)]
            # Flatten if nested (in case each key contains a list of steps)
            flattened = []
            for item in top_probs:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            top_probs = flattened
        
        features = {}
        
        # 1. Basic probability statistics
        all_probs = []
        all_tokens = []
        for step in top_probs:
            all_probs.extend(step['probabilities'])
            all_tokens.extend(step['tokens'])
        
        if all_probs:
            features['max_prob'] = max(all_probs)
            features['min_prob'] = min(all_probs)
            features['mean_prob'] = np.mean(all_probs)
            features['std_prob'] = np.std(all_probs)
            features['entropy_all'] = -np.sum([p * np.log(p + 1e-10) for p in all_probs if p > 0])
        
        # 2. Yes/No specific probabilities
        yes_probs = []
        no_probs = []
        
        for step in top_probs:
            step_yes_probs = []
            step_no_probs = []
            
            for token, prob in zip(step['tokens'], step['probabilities']):
                token_clean = token.strip().lower()
                if any(yes_var.lower() in token_clean for yes_var in self.yes_variations):
                    step_yes_probs.append(prob)
                    yes_probs.append(prob)
                elif any(no_var.lower() in token_clean for no_var in self.no_variations):
                    step_no_probs.append(prob)
                    no_probs.append(prob)
        
        # Yes probability features
        features['max_yes_prob'] = max(yes_probs) if yes_probs else 0.0
        features['min_yes_prob'] = min(yes_probs) if yes_probs else 0.0
        features['mean_yes_prob'] = np.mean(yes_probs) if yes_probs else 0.0
        features['sum_yes_prob'] = sum(yes_probs) if yes_probs else 0.0
        features['count_yes_tokens'] = len(yes_probs)
        
        # No probability features
        features['max_no_prob'] = max(no_probs) if no_probs else 0.0
        features['min_no_prob'] = min(no_probs) if no_probs else 0.0
        features['mean_no_prob'] = np.mean(no_probs) if no_probs else 0.0
        features['sum_no_prob'] = sum(no_probs) if no_probs else 0.0
        features['count_no_tokens'] = len(no_probs)
        
        # 3. Confidence and uncertainty measures
        features['yes_no_prob_diff'] = features['max_yes_prob'] - features['max_no_prob']
        features['yes_no_sum_diff'] = features['sum_yes_prob'] - features['sum_no_prob']
        features['confidence_ratio'] = (features['max_yes_prob'] / (features['max_no_prob'] + 1e-10))
        
        # 4. Per-step analysis
        step_entropies = []
        step_max_probs = []
        step_uncertainties = []
        
        for step in top_probs:
            probs = step['probabilities']
            if probs:
                # Entropy of this step
                entropy = -np.sum([p * np.log(p + 1e-10) for p in probs if p > 0])
                step_entropies.append(entropy)
                
                # Max probability of this step
                step_max_probs.append(max(probs))
                
                # Uncertainty (1 - max_prob)
                step_uncertainties.append(1 - max(probs))
        
        if step_entropies:
            features['mean_step_entropy'] = np.mean(step_entropies)
            features['max_step_entropy'] = max(step_entropies)
            features['min_step_entropy'] = min(step_entropies)
            features['std_step_entropy'] = np.std(step_entropies)
        
        if step_max_probs:
            features['mean_step_max_prob'] = np.mean(step_max_probs)
            features['min_step_max_prob'] = min(step_max_probs)
            features['std_step_max_prob'] = np.std(step_max_probs)
        
        if step_uncertainties:
            features['mean_step_uncertainty'] = np.mean(step_uncertainties)
            features['max_step_uncertainty'] = max(step_uncertainties)
            features['min_step_uncertainty'] = min(step_uncertainties)
        
        # 5. Sequence-level features
        features['num_generation_steps'] = len(top_probs)
        features['prob_variance'] = np.var(all_probs) if all_probs else 0.0
        
        # 6. Top token analysis
        if top_probs:
            first_step = top_probs[0]
            features['first_token_max_prob'] = max(first_step['probabilities']) if first_step['probabilities'] else 0.0
            features['first_token_entropy'] = -np.sum([p * np.log(p + 1e-10) for p in first_step['probabilities'] if p > 0])
            
            if len(top_probs) > 1:
                last_step = top_probs[-1]
                features['last_token_max_prob'] = max(last_step['probabilities']) if last_step['probabilities'] else 0.0
                features['last_token_entropy'] = -np.sum([p * np.log(p + 1e-10) for p in last_step['probabilities'] if p > 0])
        
        return features
    
    def _get_empty_features(self) -> Dict:
        """Return empty features for cases with no probability data"""
        return {
            'max_prob': 0.0, 'min_prob': 0.0, 'mean_prob': 0.0, 'std_prob': 0.0, 'entropy_all': 0.0,
            'max_yes_prob': 0.0, 'min_yes_prob': 0.0, 'mean_yes_prob': 0.0, 'sum_yes_prob': 0.0, 'count_yes_tokens': 0,
            'max_no_prob': 0.0, 'min_no_prob': 0.0, 'mean_no_prob': 0.0, 'sum_no_prob': 0.0, 'count_no_tokens': 0,
            'yes_no_prob_diff': 0.0, 'yes_no_sum_diff': 0.0, 'confidence_ratio': 1.0,
            'mean_step_entropy': 0.0, 'max_step_entropy': 0.0, 'min_step_entropy': 0.0, 'std_step_entropy': 0.0,
            'mean_step_max_prob': 0.0, 'min_step_max_prob': 0.0, 'std_step_max_prob': 0.0,
            'mean_step_uncertainty': 1.0, 'max_step_uncertainty': 1.0, 'min_step_uncertainty': 1.0,
            'num_generation_steps': 0, 'prob_variance': 0.0,
            'first_token_max_prob': 0.0, 'first_token_entropy': 0.0,
            'last_token_max_prob': 0.0, 'last_token_entropy': 0.0
        }

class UncertaintyClassifier:
    def __init__(self):
        self.feature_extractor = UncertaintyFeatureExtractor()
        self.model = None
        
    def prepare_training_data(self, results_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load results and prepare training data"""
        print(f"Loading results from {results_path}")
        
        df = pd.read_json(results_path)
        
        print(f"Loaded {len(df)} samples")
        print(f"Columns: {df.columns.tolist()}")
        
        # Show a sample of the data for debugging
        if len(df) > 0:
            print(f"Sample top_probs type: {type(df.iloc[0].get('top_probs', None))}")
            sample_top_probs = df.iloc[0].get('top_probs', None)
            if sample_top_probs:
                print(f"Sample top_probs preview: {str(sample_top_probs)[:200]}...")
        
        # Extract features and labels
        features_list = []
        labels = []
        
        for idx, row in df.iterrows():
            # Extract features from top_probs
            top_probs = row.get('top_probs', [])
            
            # Handle case where top_probs is stored as JSON string
            if isinstance(top_probs, str):
                try:
                    top_probs = json.loads(top_probs)
                except (json.JSONDecodeError, TypeError):
                    print(f"Warning: Could not parse top_probs for row {idx}")
                    top_probs = []
            
            features = self.feature_extractor.extract_features(top_probs)
            features_list.append(features)
            
            # Determine if prediction is correct
            predicted = int(row.get('chatbot_response_clean', 0))
            actual = row['label']  # Assuming 'label' column exists
            
            is_correct = 1 if predicted == actual else 0
            labels.append(is_correct)
            
            if idx < 5:  # Debug first few
                print(f"Sample {idx}: predicted={predicted}, actual={actual}, correct={is_correct}")
        
        feature_df = pd.DataFrame(features_list)
        labels_series = pd.Series(labels)
        
        print(f"Feature matrix shape: {feature_df.shape}")
        print(f"Label distribution: {labels_series.value_counts()}")
        
        # Check if we have enough data for training
        if len(feature_df) < 10:
            raise ValueError(f"Not enough samples for training: {len(feature_df)}. Need at least 10 samples.")
        
        # Check label distribution
        unique_labels = labels_series.value_counts()
        if len(unique_labels) < 2:
            raise ValueError(f"Need both positive and negative samples. Current distribution: {unique_labels.to_dict()}")
        
        return feature_df, labels_series
    
    def _extract_prediction(self, response: str) -> int:
        """Extract binary prediction from response"""
        if not response or pd.isna(response):
            return -1
        
        response_lower = str(response).lower().strip()
        
        # Check for yes variations
        for yes_var in self.feature_extractor.yes_variations:
            if yes_var.lower() in response_lower:
                return 1
        
        # Check for no variations
        for no_var in self.feature_extractor.no_variations:
            if no_var.lower() in response_lower:
                return 0
        
        return -1  # Unknown
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Train the XGBoost model"""
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train XGBoost model
        print("Training XGBoost model...")
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Feature importance
        print("\nTop 10 Feature Importances:")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(10))
        
        return X_test, y_test, y_pred, y_pred_proba
    
    def save_model(self, model_path: str):
        """Save the trained model"""
        if self.model:
            self.model.save_model(model_path)
            print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        print(f"Model loaded from {model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train uncertainty classifier')
    parser.add_argument('--results_path', type=str, required=True,
                       help='Path to JSON results file with top_probs')
    parser.add_argument('--model_output', type=str, default='uncertainty_model.json',
                       help='Path to save the trained model')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = UncertaintyClassifier()
    
    # Prepare training data
    X, y = classifier.prepare_training_data(args.results_path)
    
    # Train model
    classifier.train(X, y)
    
    # Save model
    classifier.save_model(args.model_output)

if __name__ == "__main__":
    main() 