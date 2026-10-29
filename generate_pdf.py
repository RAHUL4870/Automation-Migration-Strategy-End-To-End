"""
Script to generate a comprehensive PDF documentation: PROJECT_CODE_AND_WORKFLOW_EXPLANATION.pdf
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch

pdf_filename = 'PROJECT_CODE_AND_WORKFLOW_EXPLANATION.pdf'
doc = SimpleDocTemplate(
    pdf_filename,
    pagesize=letter,
    rightMargin=40,
    leftMargin=40,
    topMargin=40,
    bottomMargin=40
)

styles = getSampleStyleSheet()

# Custom Styles
title_style = ParagraphStyle(
    'CoverTitle',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=20,
    leading=24,
    textColor=colors.HexColor('#1e3a8a'),
    spaceAfter=8
)

subtitle_style = ParagraphStyle(
    'CoverSubtitle',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=11,
    leading=15,
    textColor=colors.HexColor('#475569'),
    spaceAfter=14
)

h1_style = ParagraphStyle(
    'SectionH1',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=13,
    leading=17,
    textColor=colors.HexColor('#0f172a'),
    spaceBefore=12,
    spaceAfter=6
)

h2_style = ParagraphStyle(
    'SectionH2',
    parent=styles['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=10.5,
    leading=14,
    textColor=colors.HexColor('#2563eb'),
    spaceBefore=8,
    spaceAfter=4
)

body_style = ParagraphStyle(
    'BodyTextCustom',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9,
    leading=13,
    textColor=colors.HexColor('#1e293b'),
    spaceAfter=6
)

code_style = ParagraphStyle(
    'CodeSnippet',
    parent=styles['Normal'],
    fontName='Courier',
    fontSize=8,
    leading=11,
    textColor=colors.HexColor('#0f172a'),
    backColor=colors.HexColor('#f1f5f9'),
    borderPadding=5,
    spaceBefore=4,
    spaceAfter=6
)

qa_q_style = ParagraphStyle(
    'QAQuestion',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=9.5,
    leading=13,
    textColor=colors.HexColor('#1e3a8a'),
    spaceBefore=6,
    spaceAfter=3
)

qa_a_style = ParagraphStyle(
    'QAAnswer',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=8.5,
    leading=12,
    textColor=colors.HexColor('#334155'),
    spaceAfter=6
)

elements = []

# Title & Subtitle
elements.append(Paragraph('Enterprise Automation Migration Strategy', title_style))
elements.append(Paragraph('<b>End-to-End Line-by-Line ML Code, Preprocessing, Mathematical Foundations and Workflow Architecture</b><br/><font size=8.5 color="#64748b">Academic Defense and Comprehensive Project Documentation</font>', subtitle_style))
elements.append(Spacer(1, 6))

# 1. Project Architecture
elements.append(Paragraph('1. Project Architecture and End-to-End Workflow', h1_style))
elements.append(Paragraph('This platform provides a data-driven, multi-output machine learning system to evaluate industrial automation readiness across 6,000 enterprise records in 5 major industrial sectors (Automotive, Electronics, Food, Pharma, and Textile). The system simultaneously executes 6 dedicated regression pipelines (forecasting Cost Savings, Productivity Gains, Timeline, Profit Increase, Demand, and Risk Score) and 1 multi-class classification pipeline (recommending strategic phase transitions: Stay vs. Move to Semi/Flexible/Full Automation).', body_style))

# Winning Models Table
table_data = [
    ['Target Variable', 'Type', 'Winning Algorithm', 'Key Metric Score'],
    ['Labor_Cost_Savings_Pct', 'Regression', 'Polynomial Reg (Deg 2)', 'R2 = 0.9307 (93.1%)'],
    ['Productivity_Improvement_Pct', 'Regression', 'Polynomial Reg (Deg 2)', 'R2 = 0.8726 (87.3%)'],
    ['Next_Recommended_Phase', 'Classification', 'Support Vector Machine (SVC)', 'Accuracy = 92.50%'],
    ['Expected_Annual_Profit_Increase_Pct', 'Regression', 'Polynomial Reg (Deg 2)', 'R2 = 0.8073 (80.7%)'],
    ['Customer_Demand_Forecast_Pct', 'Regression', 'Linear Regression', 'R2 = 0.7327 (73.3%)'],
    ['Risk_Score', 'Regression', 'Polynomial Reg (Deg 2)', 'R2 = 0.6546 (65.5%)'],
    ['Time_to_Next_Migration_Months', 'Regression', 'Polynomial Reg (Deg 2)', 'R2 = 0.5803 (58.0%)']
]

t = Table(table_data, colWidths=[1.8*inch, 1.0*inch, 2.3*inch, 1.8*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,0), 8),
    ('BOTTOMPADDING', (0,0), (-1,0), 5),
    ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8fafc')),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
    ('FONTSIZE', (0,1), (-1,-1), 7.5),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')])
]))
elements.append(t)
elements.append(Spacer(1, 10))

# 2. Section 2: Data Preprocessing & Pipeline Architecture
elements.append(Paragraph('2. Feature Engineering and Preprocessing Pipeline', h1_style))
elements.append(Paragraph('<b>Why We Used ColumnTransformer with StandardScaler and OneHotEncoder:</b>', h2_style))
elements.append(Paragraph('• <b>StandardScaler:</b> Normalizes 16 numerical features by calculating z-scores (z = (x - mean) / std). This prevents features with large scales (e.g. Monthly Production Volume up to 35,000 units) from dominating distance metrics in Support Vector Machines (SVM) and K-Nearest Neighbors (KNN).<br/>'
                          '• <b>OneHotEncoder(handle_unknown="ignore"):</b> Converts 3 categorical features (Industry, Product_Lifecycle_Stage, Current_Automation_Level) into orthogonal binary indicator vectors. Avoids assigning misleading ordinal rankings (such as Automotive=0, Textile=4).<br/>'
                          '• <b>Pipeline and Data Leakage Prevention:</b> Encapsulating transformers inside a Scikit-Learn Pipeline guarantees that scaler statistics are fit solely on training folds, eliminating data leakage.', body_style))

# Code Block
elements.append(Paragraph(
'# Preprocessor Definition<br/>'
'preprocessor = ColumnTransformer(transformers=[<br/>'
'&nbsp;&nbsp;("num", StandardScaler(), numerical_features),<br/>'
'&nbsp;&nbsp;("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)<br/>'
'])', code_style))

# 3. Section 3: Mathematical Foundations & Algorithms
elements.append(Paragraph('3. Mathematical Foundations and Algorithm Evaluation', h1_style))
elements.append(Paragraph('<b>Regression Algorithms Evaluated (5 Algorithms):</b>', h2_style))
elements.append(Paragraph('1. <b>Linear Regression (OLS):</b> Minimizes sum of squared residuals via closed-form normal equation. Achieved R2 = 0.7327 on Demand Forecast.<br/>'
                          '2. <b>Decision Tree Regressor:</b> Recursively splits feature space to maximize MSE reduction.<br/>'
                          '3. <b>Polynomial Regression (Degree 2) [WINNER]:</b> Expands features into quadratic cross-terms (x_i * x_j). Captures compounding industrial synergies (e.g. Volume * Labor Cost), reaching <b>R2 = 0.9307 (93.1%)</b> on Labor Cost Savings.<br/>'
                          '4. <b>Support Vector Regressor (SVR):</b> Constructs an epsilon-insensitive tube using RBF kernel mapping.<br/>'
                          '5. <b>K-Nearest Neighbors (KNN):</b> Instance-based inverse distance-weighted averaging across 5 neighbors.', body_style))

elements.append(Paragraph('<b>Classification Algorithms Evaluated (5 Algorithms):</b>', h2_style))
elements.append(Paragraph('1. <b>Support Vector Classifier (SVC) [WINNER]:</b> Maximizes classification margin in infinite-dimensional Hilbert space via RBF kernel. Achieved <b>92.50% Accuracy</b>.<br/>'
                          '2. <b>Logistic Regression:</b> Multinomial softmax cross-entropy minimization (92.33% Accuracy).<br/>'
                          '3. <b>Decision Tree Classifier:</b> Gini impurity recursive partitioning (88.75% Accuracy).<br/>'
                          '4. <b>Gaussian Naive Bayes:</b> Bayes conditional independence probability estimator (87.00% Accuracy).<br/>'
                          '5. <b>KNN Classifier:</b> Majority voting across nearest neighbors (74.75% Accuracy).', body_style))

elements.append(PageBreak())

# 4. Section 4: Hyperparameter Optimization
elements.append(Paragraph('4. Hyperparameter Optimization and Tuning', h1_style))
elements.append(Paragraph('<b>GridSearchCV and RandomizedSearchCV Implementation:</b>', h2_style))
elements.append(Paragraph('• <b>GridSearchCV on SVR:</b> Exhaustively evaluated regularization parameter C in [1.0, 10.0, 50.0, 100.0] and epsilon in [0.01, 0.1, 0.2, 0.5] with 5-fold cross-validation, improving test R2 from 0.880 to <b>0.9169</b>.<br/>'
                          '• <b>RandomizedSearchCV on Decision Trees:</b> Sampled 30 iterations from distributions of max_depth (4 to 25), min_samples_split (2 to 20), and min_samples_leaf (1 to 10), boosting Decision Tree R2 from 0.777 to <b>0.8124</b>.<br/>'
                          '• <b>GridSearchCV on SVC:</b> Optimized C=10.0 and RBF kernel parameter gamma="scale", securing <b>92.50% Accuracy</b>.', body_style))

# 5. Section 5: Feature Importance Analysis
elements.append(Paragraph('5. Feature Importance and Key Operational Drivers', h1_style))
elements.append(Paragraph('Using <b>Permutation Feature Importance</b> (measuring R2 reduction when feature columns are randomly shuffled), the top drivers for automation decisions are:<br/>'
                          '1. <b>Current_Automation_Level:</b> Accounts for >60% of variance between manual and automated operations.<br/>'
                          '2. <b>Baseline ROI (%):</b> Multiplier for capital allocation feasibility.<br/>'
                          '3. <b>Monthly_Production_Volume:</b> Critical scale driver for payback acceleration.<br/>'
                          '4. <b>Cycle_Time_Min:</b> Major lever for takt time compression.<br/>'
                          '5. <b>Safety_Incidents_Annual and Defect_Rate_Pct:</b> Primary determinants for the Risk Score model.', body_style))

elements.append(Spacer(1, 6))

# 6. Section 6: Viva Voce Defense Questions & Answers
elements.append(Paragraph('6. Viva Voce / Project Defense Q and A Guide', h1_style))

qas = [
    ('Q1: Why did Polynomial Regression outperform Linear Regression and Decision Trees on Labor Cost Savings?',
     'Answer: Manufacturing automation benefits exhibit non-linear compounding interactions (e.g. high volume combined with high manual labor cost creates quadratic savings, not simple additive linear savings). Polynomial Regression (Degree 2) explicitly includes cross-product terms (x_i * x_j), enabling the model to capture non-linear manufacturing dynamics while retaining mathematical convex stability, reaching R2 = 93.1%.'),
    
    ('Q2: Why was ColumnTransformer inside a Pipeline chosen for preprocessing?',
     'Answer: Preprocessing before data splitting causes Data Leakage, where test set distribution parameters (mean and variance) contaminate training folds. A Scikit-Learn Pipeline fits transformations strictly on training folds and transforms validation/test sets without leakage.'),
    
    ('Q3: Why was One-Hot Encoding used instead of Label Encoding for categorical features?',
     'Answer: Label Encoding assigns arbitrary integers (0, 1, 2, 3, 4) to categories like Industry, which misleads linear and distance-based estimators into assuming false mathematical magnitude relationships (e.g. Textile=4 is greater than Automotive=0). One-Hot Encoding creates orthogonal binary indicator features.'),
    
    ('Q4: How does Support Vector Classifier achieve 92.5% accuracy on Next_Recommended_Phase?',
     'Answer: The Radial Basis Function (RBF) kernel projects the input feature space into an infinite-dimensional Reproducing Kernel Hilbert Space (RKHS), where non-linear decision boundaries between automation phases become linearly separable. Coupled with One-vs-One (OvO) multi-class voting, it achieves optimal margin separation.')
]

for q, a in qas:
    elements.append(Paragraph(f'<b>{q}</b>', qa_q_style))
    elements.append(Paragraph(a, qa_a_style))

doc.build(elements)
print('SUCCESS: PROJECT_CODE_AND_WORKFLOW_EXPLANATION.pdf has been generated successfully!')
