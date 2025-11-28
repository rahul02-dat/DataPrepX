import argparse
from pathlib import Path
from modules.utils import setup_logging, load_config
from modules.preprocess import DataPreprocessor
from modules.estimation import ModelEstimator
from modules.explainability import ExplainabilityAnalyzer
from modules.report_gen import ReportGenerator

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(
        description='DataPrepX v1.1 - AI-Enabled Data Processing and ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/loan_approval.csv --target loan_approved
  python main.py --input data/housing.csv --target price --task regression --tune
  python main.py --input data/data.xlsx --target outcome --report-format both --explain
        """
    )
    
    parser.add_argument('--input', '-i', required=True, type=str,
                       help='Input data file path (CSV or XLSX)')
    parser.add_argument('--target', '-t', required=True, type=str,
                       help='Target column name for prediction')
    parser.add_argument('--task', choices=['auto', 'classification', 'regression'],
                       default='auto', help='Task type (default: auto-detect)')
    parser.add_argument('--config', '-c', type=str, default='config/default_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='Output directory for reports')
    parser.add_argument('--report-format', choices=['pdf', 'docx', 'both'],
                       default='pdf', help='Report format')
    parser.add_argument('--tune', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--explain', action='store_true',
                       help='Generate explainability analysis (SHAP/LIME)')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Enable parallel model training')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(10)
    
    logger.info("=" * 70)
    logger.info("DataPrepX v1.1 - Starting Pipeline")
    logger.info("=" * 70)
    
    try:
        config = load_config(args.config)
        
        if args.tune:
            config['estimation']['enable_hyperparameter_tuning'] = True
        
        config['estimation']['parallel_training'] = args.parallel
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Step 1/5: Loading data from {args.input}")
        preprocessor = DataPreprocessor(config['preprocessing'])
        df_clean, metadata = preprocessor.process(args.input)
        logger.info(f"✓ Data preprocessed: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        
        logger.info(f"Step 2/5: Training ML models (Task: {args.task})")
        estimator = ModelEstimator(config['estimation'])
        results = estimator.fit_and_evaluate(df_clean, args.target, args.task)
        logger.info(f"✓ Best Model: {results['best_model']} (Score: {results['best_score']:.4f})")
        
        explainability_results = None
        if args.explain:
            logger.info("Step 3/5: Generating explainability analysis")
            explainer = ExplainabilityAnalyzer(config.get('explainability', {}))
            explainability_results = explainer.analyze(
                estimator.models[results['best_model']],
                df_clean.drop(columns=[args.target]),
                df_clean[args.target],
                results['task_type']
            )
            logger.info("✓ Explainability analysis complete")
        else:
            logger.info("Step 3/5: Skipping explainability (use --explain to enable)")
        
        logger.info("Step 4/5: Generating reports")
        report_gen = ReportGenerator(config['report'])
        report_paths = report_gen.generate(
            df_clean, metadata, results, explainability_results,
            output_dir, args.report_format
        )
        
        logger.info("Step 5/5: Pipeline complete!")
        logger.info("=" * 70)
        logger.info("Generated Reports:")
        for path in report_paths:
            logger.info(f"  📄 {path}")
        logger.info("=" * 70)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    exit(main())