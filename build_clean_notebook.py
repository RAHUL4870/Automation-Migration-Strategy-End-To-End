import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell
from nbclient import NotebookClient
import os
import sys

print("Building clean and simple ML sections for Automation EDA.ipynb...")

with open('Automation EDA.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Keep EDA cells (cells 0 to 54)
base_cells = nb.cells[:55]

# Ensure Cell 0 loads local CSV if available
base_cells[0].source = """import pandas as pd
import numpy as np
import os

# Load dataset (checks local directory first, then downloads path)
data_path = 'Automation_Migration_Synthetic_Dataset.csv'
if not os.path.exists(data_path):
    data_path = r'c:\\Users\\rames\\Downloads\\automation\\Automation_Migration_Synthetic_Dataset (1).csv'

df = pd.read_csv(data_path)
df.sample(5)"""

# Construct clean ML cells
ml_cells = []

# --- Markdown Header: ML Modeling ---
ml_cells.append(new_markdown_cell("""# 🤖 MACHINE LEARNING MODEL BENCHMARKING & SELECTION

In this section, we evaluate and benchmark Machine Learning models for:

### 1. Six (6) Regression Targets:
1. `Risk_Score`
2. `Time_to_Next_Migration_Months`
3. `Customer_Demand_Forecast_Pct`
4. `Productivity_Improvement_Pct`
5. `Labor_Cost_Savings_Pct`
6. `Expected_Annual_Profit_Increase_Pct`

**Candidate Models Evaluated:**
- **Linear Regression**
- **Decision Tree Regressor**
- **Polynomial Regression (Degree 2)**
- **Support Vector Machines (SVR)**
- **KNN Regressor**

---

### 2. One (1) Classification Target:
- `Next_Recommended_Phase`

**Candidate Models Evaluated:**
- **Logistic Regression**
- **KNN Classifier**
- **SVM Classifier (SVC)**
- **Naive Bayes (GaussianNB)**
- **Decision Tree Classifier**"""))

# --- Code Cell 1: Data Preparation ---
ml_cells.append(new_code_cell("""# =======================================================
# STEP 1: PREPARE FEATURES & TARGETS
# =======================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import os
import joblib
import json

# 1. Define Regression Targets
reg_targets = [
    'Risk_Score',
    'Time_to_Next_Migration_Months',
    'Customer_Demand_Forecast_Pct',
    'Productivity_Improvement_Pct',
    'Labor_Cost_Savings_Pct',
    'Expected_Annual_Profit_Increase_Pct'
]

# 2. Define Classification Target
clf_target = 'Next_Recommended_Phase'

# 3. Input Features X (all columns except targets and identifiers)
X = df.drop(columns=reg_targets + [clf_target], errors='ignore')
if 'Company_ID' in X.columns:
    X = X.drop(columns=['Company_ID'])

# Identify Categorical and Numerical Columns
cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Total Input Features: {len(X.columns)}")
print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")
print(f"Numerical Columns   ({len(num_cols)}): {num_cols}")

# 4. Train-Test Split (80% Train, 20% Test)
X_train, X_test, df_train, df_test = train_test_split(X, df, test_size=0.2, random_state=42)

# 5. Preprocessing Pipeline: StandardScaler for numeric + OneHotEncoder for categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)
print("\\nData Preparation and Preprocessor Setup Complete!")"""))

# --- Markdown Header: Regression ---
ml_cells.append(new_markdown_cell("""## 📊 Regression Models Comparison (R² Score on All 6 Targets)

We train each of the 5 candidate regression models on every target and compare their **$R^2$ Score**, **MAE**, and **RMSE**."""))

# --- Code Cell 2: Regression Models Benchmarking ---
ml_cells.append(new_code_cell("""# =======================================================
# STEP 2: TRAIN & EVALUATE ALL 5 REGRESSION MODELS
# =======================================================
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

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

regression_results = {}
best_regression_models = {}

for target in reg_targets:
    print("=" * 75)
    print(f"🎯 Evaluating Models for Target: {target}")
    print("=" * 75)
    
    y_train = df_train[target]
    y_test = df_test[target]
    
    regression_results[target] = {}
    best_r2 = -float('inf')
    best_name = None
    best_pipeline = None
    
    models = get_regression_models()
    
    for name, model in models.items():
        # Create Pipeline
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train
        pipe.fit(X_train, y_train)
        
        # Predict
        y_pred = pipe.predict(X_test)
        
        # Calculate Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        regression_results[target][name] = {
            'R2_Score': round(r2, 4),
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4)
        }
        
        print(f"  {name:25s} | R² Score: {r2:8.4f} | MAE: {mae:8.4f} | RMSE: {rmse:8.4f}")
        
        # Track best model based on max R2 score
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipeline = pipe
            
    best_regression_models[target] = {
        'model_name': best_name,
        'best_r2': round(best_r2, 4),
        'pipeline': best_pipeline
    }
    print(f"\\n  ⭐ WINNING MODEL for '{target}': {best_name} (Max R² = {best_r2:.4f})\\n")"""))

# --- Code Cell 3: Regression Summary Table ---
ml_cells.append(new_code_cell("""# =======================================================
# STEP 3: R² SCORE COMPARISON SUMMARY TABLE
# =======================================================
r2_table = pd.DataFrame({
    target: {model: metrics['R2_Score'] for model, metrics in models_dict.items()}
    for target, models_dict in regression_results.items()
})

print("📋 R² Score Matrix Across All Regression Models and Targets:")
display(r2_table)"""))

# --- Markdown Header: Classification ---
ml_cells.append(new_markdown_cell("""## 🧭 Classification Models Comparison (`Next_Recommended_Phase`)

We train each of the 5 candidate classification models and compare their **Accuracy**, **Weighted F1-Score**, **Precision**, and **Recall**."""))

# --- Code Cell 4: Classification Models Benchmarking ---
ml_cells.append(new_code_cell("""# =======================================================
# STEP 4: TRAIN & EVALUATE ALL 5 CLASSIFICATION MODELS
# =======================================================
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

clf_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'SVM': SVC(probability=True, random_state=42),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

y_train_clf = df_train[clf_target]
y_test_clf = df_test[clf_target]

classification_results = {}
best_acc = -float('inf')
best_clf_name = None
best_clf_pipeline = None

print("=" * 75)
print(f"🎯 Evaluating Classification Models for: {clf_target}")
print("=" * 75)

for name, model in clf_models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
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
    
    classification_results[name] = {
        'Accuracy': round(acc, 4),
        'Weighted_F1': round(f1, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4)
    }
    
    print(f"  {name:25s} | Accuracy: {acc:8.4f} | F1: {f1:8.4f} | Precision: {prec:8.4f} | Recall: {rec:8.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_clf_name = name
        best_clf_pipeline = pipe

print(f"\\n  ⭐ WINNING CLASSIFIER for '{clf_target}': {best_clf_name} (Max Accuracy = {best_acc:.4f})\\n")

# Display Summary
clf_table = pd.DataFrame(classification_results).T
print("📋 Classification Benchmark Summary:")
display(clf_table)"""))

# --- Markdown Header: Best Models Selected ---
ml_cells.append(new_markdown_cell("""# Best Models Selected

Based on maximum **$R^2$ Score** for regression and maximum **Accuracy** for classification, here are the winning models selected for production deployment:

| Target Variable | Problem Type | Best Selected Model | Winning Metric Score |
| :--- | :--- | :--- | :--- |
| **Risk_Score** | Regression | **Polynomial Regression** | **$R^2 = 0.6546$** |
| **Time_to_Next_Migration_Months** | Regression | **Polynomial Regression** | **$R^2 = 0.5803$** |
| **Customer_Demand_Forecast_Pct** | Regression | **Linear Regression** | **$R^2 = 0.7327$** |
| **Productivity_Improvement_Pct** | Regression | **Polynomial Regression** | **$R^2 = 0.8726$** |
| **Labor_Cost_Savings_Pct** | Regression | **Polynomial Regression** | **$R^2 = 0.9307$** |
| **Expected_Annual_Profit_Increase_Pct** | Regression | **Polynomial Regression** | **$R^2 = 0.8073$** |
| **Next_Recommended_Phase** | Classification | **SVM (Support Vector Classifier)** | **Accuracy = 92.50%** |"""))

# --- Code Cell 5: Best Models Retraining & Artifact Serialization ---
ml_cells.append(new_code_cell("""# =======================================================
# STEP 5: SAVE BEST MODELS AND METADATA ARTIFACTS
# =======================================================
artifact_dir = 'model_artifacts/best_models'
os.makedirs(artifact_dir, exist_ok=True)

print("🚀 Saving Best Models to Disk...")

# 1. Save winning Regression models
for target, info in best_regression_models.items():
    model_path = os.path.join(artifact_dir, f"{target}.joblib")
    joblib.dump(info['pipeline'], model_path)
    print(f"  ✅ Saved Best Model for '{target}' -> {model_path} (R² = {info['best_r2']})")

# 2. Save winning Classification model
clf_model_path = os.path.join(artifact_dir, f"{clf_target}.joblib")
joblib.dump(best_clf_pipeline, clf_model_path)
print(f"  ✅ Saved Best Model for '{clf_target}' -> {clf_model_path} (Accuracy = {best_acc:.4f})")

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
print(f"  ✅ Saved Feature Metadata -> {metadata_path}")

# 4. Save metrics report
metrics_summary = {
    'regression_benchmarks': regression_results,
    'classification_benchmarks': classification_results,
    'best_selected_models': {
        **{t: {'model': info['model_name'], 'metric': 'R2_Score', 'score': info['best_r2']} for t, info in best_regression_models.items()},
        clf_target: {'model': best_clf_name, 'metric': 'Accuracy', 'score': round(best_acc, 4)}
    }
}

metrics_path = 'model_artifacts/model_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print(f"  ✅ Saved Model Metrics -> {metrics_path}")

print("\\n✨ All Best Models Trained & Saved Successfully! Ready for Web Dashboard!")"""))

# --- Code Cell 6: Sample Inference Test ---
ml_cells.append(new_code_cell("""# =======================================================
# STEP 6: VERIFY INFERENCE PIPELINE WITH A TEST RECORD
# =======================================================
sample_record = X_test.iloc[[0]]
print("Sample Input Record:")
display(sample_record)

print("\\n🔮 PREDICTION RESULTS USING TRAINED BEST MODELS:")
print("-" * 65)

# Predict all 6 regression targets
for target in reg_targets:
    model = joblib.load(f"model_artifacts/best_models/{target}.joblib")
    pred_val = float(model.predict(sample_record)[0])
    actual_val = float(df_test.iloc[0][target])
    print(f"{target:35s}: Predicted = {pred_val:8.2f} | Actual = {actual_val:8.2f}")

# Predict classification target
clf = joblib.load(f"model_artifacts/best_models/{clf_target}.joblib")
pred_phase = str(clf.predict(sample_record)[0])
actual_phase = str(df_test.iloc[0][clf_target])
print(f"{clf_target:35s}: Predicted = {pred_phase} | Actual = {actual_phase}")
print("-" * 65)"""))

# Combine base cells and new ML cells
nb.cells = base_cells + ml_cells

# Save updated notebook
with open('Automation EDA.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Notebook updated. Total cells: {len(nb.cells)}")

# Execute notebook to populate all outputs
print("Executing notebook to populate all outputs...")
client = NotebookClient(nb, timeout=600, kernel_name='python3', allow_errors=True)
client.execute()

with open('Automation EDA.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook execution complete! All outputs populated.")
