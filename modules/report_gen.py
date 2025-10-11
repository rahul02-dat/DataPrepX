import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from modules.utils import setup_logging, get_timestamp

logger = setup_logging()
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

class ReportGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.charts_dir = None
        
    def generate(self, df: pd.DataFrame, metadata: Dict[str, Any], 
                results: Dict[str, Any], output_dir: Path, 
                report_format: str = 'pdf') -> List[Path]:
        
        self.charts_dir = output_dir / 'charts'
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating visualizations...")
        chart_paths = self._generate_charts(df, metadata, results)
        
        report_paths = []
        timestamp = get_timestamp()
        
        if report_format in ['pdf', 'both']:
            pdf_path = output_dir / f'DataPrepX_Report_{timestamp}.pdf'
            self._generate_pdf_report(df, metadata, results, chart_paths, pdf_path)
            report_paths.append(pdf_path)
            logger.info(f"PDF report generated: {pdf_path}")
        
        if report_format in ['docx', 'both']:
            docx_path = output_dir / f'DataPrepX_Report_{timestamp}.docx'
            self._generate_docx_report(df, metadata, results, chart_paths, docx_path)
            report_paths.append(docx_path)
            logger.info(f"DOCX report generated: {docx_path}")
        
        return report_paths
    
    def _generate_charts(self, df: pd.DataFrame, metadata: Dict[str, Any], 
                        results: Dict[str, Any]) -> Dict[str, Path]:
        chart_paths = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            fig, ax = plt.subplots(figsize=(12, 8))
            df[numeric_cols[:10]].boxplot(ax=ax, rot=45)
            ax.set_title('Feature Distributions', fontsize=14, fontweight='bold')
            ax.set_ylabel('Value')
            plt.tight_layout()
            path = self.charts_dir / 'boxplot.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_paths['boxplot'] = path
        
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            path = self.charts_dir / 'correlation.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_paths['correlation'] = path
        
        if numeric_cols:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:4]):
                axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[i].set_title(f'Distribution: {col}', fontweight='bold')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
            
            for i in range(len(numeric_cols[:4]), 4):
                axes[i].axis('off')
            
            plt.tight_layout()
            path = self.charts_dir / 'distributions.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_paths['distributions'] = path
        
        if results and 'feature_importance' in results:
            importance = results['feature_importance']
            if importance:
                features = list(importance.keys())[:15]
                values = [importance[f] for f in features]
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(features, values, color='steelblue')
                ax.set_xlabel('Importance', fontweight='bold')
                ax.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
                ax.invert_yaxis()
                plt.tight_layout()
                path = self.charts_dir / 'feature_importance.png'
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_paths['feature_importance'] = path
        
        if results and 'models' in results:
            model_names = list(results['models'].keys())
            if results['task_type'] == 'classification':
                scores = [results['models'][m].get('accuracy', 0) for m in model_names]
                metric = 'Accuracy'
            else:
                scores = [results['models'][m].get('r2_score', 0) for m in model_names]
                metric = 'R² Score'
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(model_names, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_ylabel(metric, fontweight='bold')
            ax.set_title(f'Model Performance Comparison ({metric})', fontsize=14, fontweight='bold')
            ax.set_ylim([0, max(scores) * 1.1])
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            path = self.charts_dir / 'model_comparison.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_paths['model_comparison'] = path
        
        return chart_paths
    
    def _generate_pdf_report(self, df: pd.DataFrame, metadata: Dict[str, Any],
                            results: Dict[str, Any], chart_paths: Dict[str, Path],
                            output_path: Path):
        
        doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=1
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        story.append(Paragraph("DataPrepX Analysis Report", title_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 30))
        
        story.append(Paragraph("1. Executive Summary", heading_style))
        summary_text = f"""
        This report provides a comprehensive analysis of the dataset processed through DataPrepX.
        The dataset contains {metadata['final_shape'][0]:,} rows and {metadata['final_shape'][1]} features
        after preprocessing and feature engineering.
        """
        story.append(Paragraph(summary_text, styles['BodyText']))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("2. Data Overview", heading_style))
        
        data_table_data = [
            ['Metric', 'Value'],
            ['Original Rows', f"{metadata['original_shape'][0]:,}"],
            ['Original Columns', str(metadata['original_shape'][1])],
            ['Final Rows', f"{metadata['final_shape'][0]:,}"],
            ['Final Columns', str(metadata['final_shape'][1])],
            ['Duplicates Removed', str(metadata.get('duplicates_removed', 0))]
        ]
        
        data_table = Table(data_table_data, colWidths=[3*inch, 3*inch])
        data_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(data_table)
        story.append(Spacer(1, 20))
        
        if metadata.get('missing_values'):
            story.append(Paragraph("3. Missing Values Handled", heading_style))
            missing_text = "The following columns had missing values that were imputed:<br/>"
            for col, count in list(metadata['missing_values'].items())[:10]:
                missing_text += f"• {col}: {count} missing values<br/>"
            story.append(Paragraph(missing_text, styles['BodyText']))
            story.append(Spacer(1, 20))
        
        story.append(PageBreak())
        
        story.append(Paragraph("4. Data Visualizations", heading_style))
        
        if 'distributions' in chart_paths:
            story.append(Paragraph("4.1 Feature Distributions", styles['Heading3']))
            story.append(Image(str(chart_paths['distributions']), width=6*inch, height=4*inch))
            story.append(Spacer(1, 20))
        
        if 'correlation' in chart_paths:
            story.append(Paragraph("4.2 Correlation Analysis", styles['Heading3']))
            story.append(Image(str(chart_paths['correlation']), width=6*inch, height=5*inch))
            story.append(Spacer(1, 20))
        
        story.append(PageBreak())
        
        if results:
            story.append(Paragraph("5. Machine Learning Results", heading_style))
            
            ml_summary = f"""
            <b>Task Type:</b> {results['task_type'].title()}<br/>
            <b>Target Column:</b> {results['target_column']}<br/>
            <b>Training Samples:</b> {results['train_size']:,}<br/>
            <b>Test Samples:</b> {results['test_size']:,}<br/>
            <b>Best Model:</b> {results['best_model']}<br/>
            <b>Best Score:</b> {results['best_score']:.4f}
            """
            story.append(Paragraph(ml_summary, styles['BodyText']))
            story.append(Spacer(1, 20))
            
            if 'model_comparison' in chart_paths:
                story.append(Paragraph("5.1 Model Performance", styles['Heading3']))
                story.append(Image(str(chart_paths['model_comparison']), width=6*inch, height=4*inch))
                story.append(Spacer(1, 20))
            
            if 'feature_importance' in chart_paths:
                story.append(Paragraph("5.2 Feature Importance", styles['Heading3']))
                story.append(Image(str(chart_paths['feature_importance']), width=6*inch, height=5*inch))
                story.append(Spacer(1, 20))
        
        story.append(PageBreak())
        story.append(Paragraph("6. Conclusion", heading_style))
        conclusion = f"""
        The DataPrepX pipeline successfully processed the dataset, applying comprehensive
        preprocessing, feature engineering, and machine learning modeling.
        {f"The best performing model was {results['best_model']} with a score of {results['best_score']:.4f}." if results else ""}
        All generated visualizations and detailed metrics are included in this report.
        """
        story.append(Paragraph(conclusion, styles['BodyText']))
        
        doc.build(story)
    
    def _generate_docx_report(self, df: pd.DataFrame, metadata: Dict[str, Any],
                             results: Dict[str, Any], chart_paths: Dict[str, Path],
                             output_path: Path):
        
        doc = Document()
        
        title = doc.add_heading('DataPrepX Analysis Report', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        doc.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        doc.add_paragraph()
        
        doc.add_heading('1. Executive Summary', 1)
        doc.add_paragraph(
            f"This report provides a comprehensive analysis of the dataset processed through DataPrepX. "
            f"The dataset contains {metadata['final_shape'][0]:,} rows and {metadata['final_shape'][1]} features "
            f"after preprocessing and feature engineering."
        )
        
        doc.add_heading('2. Data Overview', 1)
        table = doc.add_table(rows=6, cols=2)
        table.style = 'Light Grid Accent 1'
        
        table.rows[0].cells[0].text = 'Metric'
        table.rows[0].cells[1].text = 'Value'
        table.rows[1].cells[0].text = 'Original Rows'
        table.rows[1].cells[1].text = f"{metadata['original_shape'][0]:,}"
        table.rows[2].cells[0].text = 'Original Columns'
        table.rows[2].cells[1].text = str(metadata['original_shape'][1])
        table.rows[3].cells[0].text = 'Final Rows'
        table.rows[3].cells[1].text = f"{metadata['final_shape'][0]:,}"
        table.rows[4].cells[0].text = 'Final Columns'
        table.rows[4].cells[1].text = str(metadata['final_shape'][1])
        table.rows[5].cells[0].text = 'Duplicates Removed'
        table.rows[5].cells[1].text = str(metadata.get('duplicates_removed', 0))
        
        doc.add_paragraph()
        
        if metadata.get('missing_values'):
            doc.add_heading('3. Missing Values Handled', 1)
            p = doc.add_paragraph("The following columns had missing values that were imputed:")
            for col, count in list(metadata['missing_values'].items())[:10]:
                doc.add_paragraph(f"{col}: {count} missing values", style='List Bullet')
        
        doc.add_page_break()
        
        doc.add_heading('4. Data Visualizations', 1)
        
        if 'distributions' in chart_paths:
            doc.add_heading('4.1 Feature Distributions', 2)
            doc.add_picture(str(chart_paths['distributions']), width=Inches(6))
            doc.add_paragraph()
        
        if 'correlation' in chart_paths:
            doc.add_heading('4.2 Correlation Analysis', 2)
            doc.add_picture(str(chart_paths['correlation']), width=Inches(6))
            doc.add_paragraph()
        
        doc.add_page_break()
        
        if results:
            doc.add_heading('5. Machine Learning Results', 1)
            
            doc.add_paragraph(f"Task Type: {results['task_type'].title()}")
            doc.add_paragraph(f"Target Column: {results['target_column']}")
            doc.add_paragraph(f"Training Samples: {results['train_size']:,}")
            doc.add_paragraph(f"Test Samples: {results['test_size']:,}")
            doc.add_paragraph(f"Best Model: {results['best_model']}")
            doc.add_paragraph(f"Best Score: {results['best_score']:.4f}")
            doc.add_paragraph()
            
            if 'model_comparison' in chart_paths:
                doc.add_heading('5.1 Model Performance', 2)
                doc.add_picture(str(chart_paths['model_comparison']), width=Inches(6))
                doc.add_paragraph()
            
            if 'feature_importance' in chart_paths:
                doc.add_heading('5.2 Feature Importance', 2)
                doc.add_picture(str(chart_paths['feature_importance']), width=Inches(6))
                doc.add_paragraph()
        
        doc.add_page_break()
        
        doc.add_heading('6. Conclusion', 1)
        doc.add_paragraph(
            f"The DataPrepX pipeline successfully processed the dataset, applying comprehensive "
            f"preprocessing, feature engineering, and machine learning modeling. "
            f"{f'The best performing model was {results[\"best_model\"]} with a score of {results[\"best_score\"]:.4f}.' if results else ' ''} "
            f"All generated visualizations and detailed metrics are included in this report."
        )
        
        doc.save(str(output_path))