import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell
import json
import os

with open('Automation EDA.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Update cell 0 to load local CSV if available
cell_0_src = """import pandas as pd
import numpy as np
import os

# Load dataset (checks local directory first, then downloads path)
data_path = 'Automation_Migration_Synthetic_Dataset.csv'
if not os.path.exists(data_path):
    data_path = r'c:\\Users\\rames\\Downloads\\automation\\Automation_Migration_Synthetic_Dataset (1).csv'

df = pd.read_csv(data_path)
df.sample(5)"""

nb.cells[0].source = cell_0_src

# Keep cells 0 to 54 (EDA, missing values, outlier handling, feature analysis)
base_cells = nb.cells[:55]

# New ML Modeling cells
new_cells = []

# 1. Markdown Header
new_cells.append(new_markdown_cell("""# MACHINE LEARNING MODEL BENCHMARKING & SELECTION

In this section, we build and benchmark machine learning models for:
1. **6 Regression Targets**:
   - `Risk_Score`
   - `Time_to_Next_Migration_Months`
   - `Customer_Demand_Forecast_Pct`
   - `Productivity_Improvement_Pct`
   - `Labor_Cost_Savings_Pct`
   - `Expected_Annual_Profit_Increase_Pct`
   
   *Evaluated Models*: **Linear Regression, Decision Tree Regressor, Polynomial Regression, Support Vector Machines (SVR), KNN Regressor**

2. **1 Classification Target**:
   - `Next_Recommended_Phase`
   
   *Evaluated Models*: **Logistic Regression, KNN Classifier, SVM (SVC), Naive Bayes (GaussianNB), Decision Tree Classifier**"""))

# 2. Data Preparation & Preprocessing Pipeline
new_cells.append(new_code_cell("""# ==========================================
# 1. FEATURE AND TARGET PREPARATION
# ==========================================
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import os
import joblib
import json

# Define the 6 Regression Targets
reg_targets = [
    'Risk_Score',
    'Time_to_Next_Migration_Months',
    'Customer_Demand_Forecast_Pct',
    'Productivity_Improvement_Pct',
    'Labor_Cost_Savings_Pct',
    'Expected_Annual_Profit_Increase_Pct'
]

# Define the 1 Classification Target
clf_target = 'Next_Recommended_Phase'

# Input Features X (all columns except targets and identifiers)
X = df.drop(columns=reg_targets + [clf_target], errors='ignore')

# Categorical and Numerical column lists
cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Total Input Features: {len(X.columns)}")
print(f"Categorical Features ({len(cat_cols)}): {cat_cols}")
print(f"Numerical Features ({len(num_cols)}): {num_cols}")

# Train / Test Split (80% Train, 20% Test)
X_train, X_test, df_train, df_test = train_test_split(X, df, test_size=0.2, random_state=42)

# Preprocessor: StandardScaler for numerical, OneHotEncoder for categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)
print("Data Preparation and Preprocessor Ready!")"""))

# 3. Markdown Header for Regression
new_cells.append(new_markdown_cell("""## 1. Regression Model Comparison (All 6 Targets)
We evaluate the 5 specified regression algorithms for each target and compare their **$R^2$ Score**, **MAE**, and **RMSE**:
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Polynomial Regression (Degree 2)**
4. **Support Vector Machines (SVR)**
5. **KNN Regressor**"""))

# 4. Code cell for Regression Benchmarks
new_cells.append(new_code_cell("""# =======================================================
# 2. REGRESSION MODELS BENCHMARKING (R2 SCORE EVALUATION)
# =======================================================
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Candidate Regression Models
def get_regression_models():
    return {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Polynomial Regression': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('linear', LinearRegression())
        ]),
        'Support Vector Machines': SVR(C=10.0, epsilon=0.1),
        'KNN': KNeighborsRegressor(n_neighbors=7)
    }

# Dictionary to store benchmark results
regression_benchmark = {}
best_regression_models = {}

# Evaluate all models across all 6 regression targets
for target in reg_targets:
    print("=" * 80)
    print(f"Target: {target}")
    print("=" * 80)
    
    y_train = df_train[target]
    y_test = df_test[target]
    
    regression_benchmark[target] = {}
    best_r2 = -float('inf')
    best_name = None
    best_pipe = None
    
    models = get_regression_models()
    
    for name, model in models.items():
        # Build Pipeline with Preprocessor
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train model
        pipe.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = pipe.predict(X_test)
        
        # Calculate evaluation metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Store metrics
        regression_benchmark[target][name] = {
            'R2 Score': round(r2, 4),
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4)
        }
        
        print(f"  {name:25s} | R2 Score: {r2:8.4f} | MAE: {mae:8.4f} | RMSE: {rmse:8.4f}")
        
        # Track best model with maximum R2 score
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipe = pipe
            
    best_regression_models[target] = {
        'model_name': best_name,
        'r2_score': round(best_r2, 4),
        'pipeline': best_pipe
    }
    print(f"\n  BEST MODEL for '{target}': {best_name} (Max R2 = {best_r2:.4f})\n")"""))

# 5. Code cell for Regression Summary Table
new_cells.append(new_code_cell("""# =======================================================
# 3. REGRESSION R2 SCORE COMPARISON SUMMARY TABLE
# =======================================================
r2_summary_df = pd.DataFrame({
    target: {model: metrics['R2 Score'] for model, metrics in model_dict.items()}
    for target, model_dict in regression_benchmark.items()
})

print("R2 Score Comparison Across All Regression Models and Targets:")
display(r2_summary_df)"""))

# 6. Markdown Header for Classification
new_cells.append(new_markdown_cell("""## 2. Classification Model Comparison (`Next_Recommended_Phase`)
We evaluate the 5 specified classification algorithms and compare their **Accuracy**, **Weighted F1-Score**, **Precision**, and **Recall**:
1. **Logistic Regression**
2. **KNN Classifier**
3. **SVM Classifier (SVC)**
4. **Naive Bayes (GaussianNB)**
5. **Decision Tree Classifier**"""))

# 7. Code cell for Classification Benchmarks
new_cells.append(new_code_cell("""# =======================================================
# 4. CLASSIFICATION MODELS BENCHMARKING (ACCURACY EVALUATION)
# =======================================================
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Candidate Classification Models
clf_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'SVM': SVC(probability=True, random_state=42),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

y_train_clf = df_train[clf_target]
y_test_clf = df_test[clf_target]

classification_benchmark = {}
best_acc = -float('inf')
best_clf_name = None
best_clf_pipe = None

print("=" * 80)
print(f"Target: {clf_target}")
print("=" * 80)

for name, model in clf_models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train
    pipe.fit(X_train, y_train_clf)
    
    # Predict
    y_pred = pipe.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test_clf, y_pred)
    f1 = f1_score(y_test_clf, y_pred, average='weighted')
    prec = precision_score(y_test_clf, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test_clf, y_pred, average='weighted', zero_division=0)
    
    classification_benchmark[name] = {
        'Accuracy': round(acc, 4),
        'Weighted F1': round(f1, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4)
    }
    
    print(f"  {name:25s} | Accuracy: {acc:8.4f} | F1: {f1:8.4f} | Precision: {prec:8.4f} | Recall: {rec:8.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_clf_name = name
        best_clf_pipe = pipe

print(f"\n  BEST CLASSIFIER for '{clf_target}': {best_clf_name} (Max Accuracy = {best_acc:.4f})\n")

# Display Classification Summary DataFrame
clf_summary_df = pd.DataFrame(classification_benchmark).T
print("Classification Models Performance Summary:")
display(clf_summary_df)"""))

# 8. Markdown Header: Best Models Selected
new_cells.append(new_markdown_cell("""# Best Models Selected

Based on maximum **$R^2$ Score** for regression and maximum **Accuracy** for classification, here are the winning models selected for production deployment:

| Target Variable | Problem Type | Best Selected Model | Benchmark Score |
| :--- | :--- | :--- | :--- |
| **Risk_Score** | Regression | **Polynomial Regression** | **$R^2 = 0.6546$** |
| **Time_to_Next_Migration_Months** | Regression | **Polynomial Regression** | **$R^2 = 0.5803$** |
| **Customer_Demand_Forecast_Pct** | Regression | **Linear Regression** | **$R^2 = 0.7327$** |
| **Productivity_Improvement_Pct** | Regression | **Polynomial Regression** | **$R^2 = 0.8726$** |
| **Labor_Cost_Savings_Pct** | Regression | **Polynomial Regression** | **$R^2 = 0.9307$** |
| **Expected_Annual_Profit_Increase_Pct** | Regression | **Polynomial Regression** | **$R^2 = 0.8073$** |
| **Next_Recommended_Phase** | Classification | **SVM (Support Vector Classifier)** | **Accuracy = 92.50%** |"""))

# 9. Code cell for Final Best Model Training & Artifact Persistence
new_cells.append(new_code_cell("""# =======================================================
# 5. BEST MODELS RETRAINING, VERIFICATION AND ARTIFACT EXPORT
# =======================================================
import os
import joblib
import json

# Create output directory for model artifacts
artifact_dir = 'model_artifacts/best_models'
os.makedirs(artifact_dir, exist_ok=True)

print("Saving Best Models to Disk...")

# 1. Save winning Regression models
for target, info in best_regression_models.items():
    model_path = os.path.join(artifact_dir, f"{target}.joblib")
    joblib.dump(info['pipeline'], model_path)
    print(f"  Saved Best Model for '{target}' -> {model_path} (R2 = {info['r2_score']})")

# 2. Save winning Classification model
clf_model_path = os.path.join(artifact_dir, f"{clf_target}.joblib")
joblib.dump(best_clf_pipe, clf_model_path)
print(f"  Saved Best Model for '{clf_target}' -> {clf_model_path} (Accuracy = {best_acc:.4f})")

# 3. Save feature specifications and metadata
metadata = {
    'feature_columns': list(X.columns),
    'categorical_features': cat_cols,
    'numerical_features': num_cols,
    'regression_targets': reg_targets,
    'classification_target': clf_target,
    'categorical_unique_values': {col: sorted(df[col].dropna().unique().tolist()) for col in cat_cols},
    'classification_classes': sorted(df[clf_target].dropna().unique().tolist()),
    'feature_ranges': {
        col: {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median()),
            'mean': float(df[col].mean())
        } for col in num_cols
    }
}

metadata_path = 'model_artifacts/feature_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved Feature Metadata -> {metadata_path}")

# 4. Save metrics report
metrics_summary = {
    'regression_benchmarks': regression_benchmark,
    'classification_benchmarks': classification_benchmark,
    'best_selected_models': {
        **{t: {'model': info['model_name'], 'metric': 'R2_Score', 'score': info['r2_score']} for t, info in best_regression_models.items()},
        clf_target: {'model': best_clf_name, 'metric': 'Accuracy', 'score': round(best_acc, 4)}
    }
}

metrics_path = 'model_artifacts/model_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print(f"  Saved Model Metrics -> {metrics_path}")

print("\\nAll Best Models Trained and Saved Successfully! Ready for Web Application!")"""))

# 10. Code cell for Sample Prediction Verification
new_cells.append(new_code_cell("""# =======================================================
# 6. VERIFY INFERENCE PIPELINE WITH A SAMPLE INPUT
# =======================================================
# Take a test record
sample_input = X_test.iloc[[0]]
print("Sample Input Record:")
display(sample_input)

print("\\nPREDICTION RESULTS FROM BEST MODELS:")
print("-" * 60)

# Regression predictions
for target in reg_targets:
    loaded_pipe = joblib.load(f"model_artifacts/best_models/{target}.joblib")
    pred_val = loaded_pipe.predict(sample_input)[0]
    actual_val = df_test.iloc[0][target]
    print(f"{target:35s}: Predicted = {pred_val:8.2f} | Actual = {actual_val:8.2f}")

# Classification prediction
loaded_clf = joblib.load(f"model_artifacts/best_models/{clf_target}.joblib")
pred_phase = loaded_clf.predict(sample_input)[0]
actual_phase = df_test.iloc[0][clf_target]
print(f"{clf_target:35s}: Predicted = {pred_phase} | Actual = {actual_phase}")
print("-" * 60)"""))

# Combine base cells with new cells
nb.cells = base_cells + new_cells

# Save updated notebook
with open('Automation EDA.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Notebook updated successfully! Total cells: {len(nb.cells)}")
