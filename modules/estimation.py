import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            mean_squared_error, mean_absolute_error, r2_score, 
                            classification_report, confusion_matrix)
from modules.utils import setup_logging

logger = setup_logging()

class ModelEstimator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf
        self.task_type = None
        
    def fit_and_evaluate(self, df: pd.DataFrame, target_col: str, task: str = 'auto') -> Dict[str, Any]:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        if task == 'auto':
            self.task_type = self._detect_task_type(y)
        else:
            self.task_type = task
        
        logger.info(f"Task type: {self.task_type}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.get('test_size', 0.2),
            random_state=self.config.get('random_state', 42)
        )
        
        models_to_try = self._get_models()
        
        results = {
            'task_type': self.task_type,
            'target_column': target_col,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_count': X.shape[1],
            'models': {}
        }
        
        for name, model in models_to_try.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if self.task_type == 'classification':
                    metrics = self._calculate_classification_metrics(y_test, y_pred)
                    score = metrics['accuracy']
                else:
                    metrics = self._calculate_regression_metrics(y_test, y_pred)
                    score = metrics['r2_score']
                
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
                
                results['models'][name] = metrics
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = name
                    self.models[name] = model
                
                logger.info(f"{name}: Score={score:.4f}, CV={cv_scores.mean():.4f} ±{cv_scores.std():.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to train {name}: {str(e)}")
                continue
        
        results['best_model'] = self.best_model
        results['best_score'] = self.best_score
        
        if self.best_model and self.best_model in self.models:
            results['feature_importance'] = self._get_feature_importance(
                self.models[self.best_model], 
                X.columns.tolist()
            )
        
        return results
    
    def _detect_task_type(self, y: pd.Series) -> str:
        unique_values = y.nunique()
        
        if unique_values <= 20 or y.dtype == 'object' or y.dtype.name == 'category':
            return 'classification'
        else:
            return 'regression'
    
    def _get_models(self) -> Dict[str, Any]:
        if self.task_type == 'classification':
            return {
                'RandomForest': RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42,
                    n_jobs=-1
                ),
                'GradientBoosting': GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=42
                ),
                'LogisticRegression': LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            }
        else:
            return {
                'RandomForest': RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                ),
                'GradientBoosting': GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=42
                ),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42)
            }
    
    def _calculate_classification_metrics(self, y_true, y_pred) -> Dict[str, float]:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def _calculate_regression_metrics(self, y_true, y_pred) -> Dict[str, float]:
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }
    
    def _get_feature_importance(self, model, feature_names) -> Dict[str, float]:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]
            
            return {
                feature_names[i]: float(importances[i]) 
                for i in indices
            }
        elif hasattr(model, 'coef_'):
            coef = np.abs(model.coef_)
            if len(coef.shape) > 1:
                coef = coef.mean(axis=0)
            indices = np.argsort(coef)[::-1][:20]
            
            return {
                feature_names[i]: float(coef[i]) 
                for i in indices
            }
        else:
            return {}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_model is None or self.best_model not in self.models:
            raise ValueError("No model has been trained yet")
        
        return self.models[self.best_model].predict(X)