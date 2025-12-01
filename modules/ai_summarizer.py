import json
import requests
from typing import Dict, Any, Optional
from modules.utils import setup_logging

logger = setup_logging()

class AISummarizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_url = config.get('api_url', 'http://localhost:1234/v1/chat/completions')
        self.model = config.get('model', 'openai/gpt-oss-20b')
        
    def generate_summary(self, df_info: Dict[str, Any], metadata: Dict[str, Any], 
                        results: Optional[Dict[str, Any]] = None) -> str:
        
        prompt = self._build_prompt(df_info, metadata, results)
        
        try:
            response = self._call_llm(prompt)
            return response
        except Exception as e:
            logger.warning(f"AI summarization failed: {e}")
            return self._fallback_summary(df_info, metadata, results)
    
    def _build_prompt(self, df_info: Dict[str, Any], metadata: Dict[str, Any], 
                     results: Optional[Dict[str, Any]]) -> str:
        
        context = f"""Generate a comprehensive analysis summary for the following dataset processing results.

Dataset Information:
- Original dimensions: {metadata.get('original_shape', 'N/A')}
- Final dimensions: {metadata.get('final_shape', 'N/A')}
- Columns processed: {metadata.get('final_shape', [0, 0])[1]}
- Duplicates removed: {metadata.get('duplicates_removed', 0)}
"""

        if metadata.get('missing_values'):
            context += f"\nMissing Values Handled:\n"
            for col, count in list(metadata['missing_values'].items())[:5]:
                context += f"- {col}: {count} values\n"
        
        if results:
            context += f"\nMachine Learning Results:\n"
            context += f"- Task Type: {results.get('task_type', 'N/A')}\n"
            context += f"- Target Column: {results.get('target_column', 'N/A')}\n"
            context += f"- Training Samples: {results.get('train_size', 'N/A')}\n"
            context += f"- Test Samples: {results.get('test_size', 'N/A')}\n"
            context += f"- Best Model: {results.get('best_model', 'N/A')}\n"
            context += f"- Best Score: {results.get('best_score', 'N/A')}\n"
            
            if results.get('models'):
                context += "\nModel Performance Comparison:\n"
                for model_name, metrics in results['models'].items():
                    if results['task_type'] == 'classification':
                        context += f"- {model_name}: Accuracy={metrics.get('accuracy', 0):.4f}, "
                        context += f"Precision={metrics.get('precision', 0):.4f}, "
                        context += f"Recall={metrics.get('recall', 0):.4f}, "
                        context += f"F1={metrics.get('f1_score', 0):.4f}\n"
                    else:
                        context += f"- {model_name}: R²={metrics.get('r2_score', 0):.4f}, "
                        context += f"RMSE={metrics.get('rmse', 0):.2f}, "
                        context += f"MAE={metrics.get('mae', 0):.2f}\n"
            
            if results.get('feature_importance'):
                context += f"\nTop 5 Most Important Features:\n"
                for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:5], 1):
                    context += f"{i}. {feature}: {importance:.4f}\n"
        
        context += """\n\nBased on this information, provide:
1. A concise executive summary of the data processing pipeline
2. Key insights about data quality and preprocessing steps
3. Analysis of model performance and recommendations
4. Actionable insights and next steps

Keep the summary professional, data-driven, and under 500 words."""
        
        return context
    
    def _call_llm(self, prompt: str) -> str:
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a data science expert providing insights on machine learning pipeline results.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.7,
            'max_tokens': 1000
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"API call failed with status {response.status_code}")
    
    def _fallback_summary(self, df_info: Dict[str, Any], metadata: Dict[str, Any], 
                         results: Optional[Dict[str, Any]]) -> str:
        
        summary_parts = []
        
        summary_parts.append("EXECUTIVE SUMMARY")
        summary_parts.append("=" * 60)
        summary_parts.append("")
        
        summary_parts.append("Data Processing Overview:")
        summary_parts.append(f"The pipeline processed a dataset with {metadata.get('original_shape', [0, 0])[0]} rows and {metadata.get('original_shape', [0, 0])[1]} columns.")
        summary_parts.append(f"After preprocessing, the final dataset contains {metadata.get('final_shape', [0, 0])[0]} rows and {metadata.get('final_shape', [0, 0])[1]} features.")
        
        if metadata.get('duplicates_removed', 0) > 0:
            summary_parts.append(f"Removed {metadata.get('duplicates_removed')} duplicate records.")
        
        summary_parts.append("")
        
        if metadata.get('missing_values'):
            summary_parts.append("Data Quality:")
            summary_parts.append(f"Handled missing values in {len(metadata['missing_values'])} columns using imputation techniques.")
        
        summary_parts.append("")
        
        if results:
            summary_parts.append("Machine Learning Results:")
            summary_parts.append(f"Task: {results.get('task_type', 'N/A').title()}")
            summary_parts.append(f"Best Model: {results.get('best_model', 'N/A')}")
            summary_parts.append(f"Performance Score: {results.get('best_score', 0):.4f}")
            
            if results.get('models'):
                summary_parts.append("")
                summary_parts.append("Model Comparison:")
                for model_name, metrics in results['models'].items():
                    if results['task_type'] == 'classification':
                        summary_parts.append(f"  {model_name}: Accuracy={metrics.get('accuracy', 0):.4f}")
                    else:
                        summary_parts.append(f"  {model_name}: R²={metrics.get('r2_score', 0):.4f}")
        
        summary_parts.append("")
        summary_parts.append("Recommendations:")
        summary_parts.append("- Review feature importance to understand key predictors")
        summary_parts.append("- Consider cross-validation results for model stability")
        summary_parts.append("- Monitor for data drift in production deployment")
        
        return "\n".join(summary_parts)
    
    def generate_predictions_summary(self, predictions: Any, actual: Any, 
                                    task_type: str) -> str:
        
        prompt = f"""Analyze these prediction results for a {task_type} task.

Sample predictions (first 10): {predictions[:10].tolist() if hasattr(predictions, 'tolist') else predictions[:10]}
Sample actual values (first 10): {actual[:10].tolist() if hasattr(actual, 'tolist') else actual[:10]}

Provide a brief analysis of prediction quality and patterns."""
        
        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"Predictions summary failed: {e}")
            return "Predictions generated successfully. Check detailed metrics in the report."