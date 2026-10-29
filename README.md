# 🚀 Enterprise Automation Migration Strategy & Decision Intelligence Platform

An end-to-end Machine Learning and Decision Intelligence platform designed for evaluating industrial automation feasibility, forecasting ROI & Labor Cost Savings (achieving up to **93.1% $R^2$**), assessing operational risks, and recommending phase transitions across enterprise workflows.

---

## 📌 Final Year Project Highlights

- **Multi-Target Predictive Analytics**: 6 dedicated regression pipelines forecasting Cost Savings, Productivity Gains, Timeline, Demand, Risk Index, and Profit Increase.
- **Phase Transition Recommendation**: Multi-class classification model recommending optimal automation tiers (**92.5% Accuracy**).
- **Comprehensive Algorithm Benchmarking**: Strict evaluation across Linear Regression, Decision Trees, Polynomial Regression, Support Vector Machines (SVR/SVC), and KNN.
- **Hyperparameter Optimization**: Systematic tuning using **GridSearchCV** & **RandomizedSearchCV**.
- **Feature Importance Analysis**: Extraction and visualization of key operational drivers.
- **Interactive Enterprise Web Dashboard**: Fully featured dashboard matching enterprise design standards with real-time Chart.js interactive visualizations, process recommendations table, risk donut chart, risk vs benefit scatter plot, and interactive ML parameter simulation.

---

## 📊 Machine Learning Model Benchmarks

### 1. Best Selected Models (Winning Scores)

| Target Feature | Problem Type | Best Selected Algorithm | Optimization Method | Winning Metric Score |
| :--- | :--- | :--- | :--- | :--- |
| **Labor_Cost_Savings_Pct** | Regression | **Polynomial Regression (Degree 2)** | Polynomial Features + Linear Estimator | **$R^2 = 0.9307$ (93.1%)** \| MAE: 3.59% |
| **Productivity_Improvement_Pct** | Regression | **Polynomial Regression (Degree 2)** | Polynomial Features + Linear Estimator | **$R^2 = 0.8726$ (87.3%)** \| MAE: 4.75% |
| **Next_Recommended_Phase** | Classification | **Support Vector Classifier (SVC)** | GridSearchCV / RBF Kernel ($C=10$) | **Accuracy = 92.50% (92.5%)** \| F1: 0.9234 |
| **Expected_Annual_Profit_Increase_Pct** | Regression | **Polynomial Regression (Degree 2)** | Polynomial Features + Linear Estimator | **$R^2 = 0.8073$ (80.7%)** \| MAE: 5.22% |
| **Customer_Demand_Forecast_Pct** | Regression | **Linear Regression** | Standard Scaled Feature Pipeline | **$R^2 = 0.7327$ (73.3%)** \| MAE: 4.29% |
| **Risk_Score** | Regression | **Polynomial Regression (Degree 2)** | Polynomial Features + Linear Estimator | **$R^2 = 0.6546$ (65.5%)** \| MAE: 7.31 |
| **Time_to_Next_Migration_Months** | Regression | **Polynomial Regression (Degree 2)** | Polynomial Features + Linear Estimator | **$R^2 = 0.5803$ (58.0%)** \| MAE: 9.47 mos |

---

### 2. Complete 5-Model Regression Matrix ($R^2$ Scores)

| Regression Target | Linear Regression | Decision Tree | Polynomial Reg (Deg 2) | Support Vector Machines (SVR) | KNN Regressor |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Labor_Cost_Savings_Pct** | 0.8608 | 0.8580 | **0.9307 ⭐** | 0.9139 | 0.7890 |
| **Productivity_Improvement_Pct** | 0.7978 | 0.7773 | **0.8726 ⭐** | 0.8472 | 0.7119 |
| **Expected_Annual_Profit_Increase_Pct** | 0.7524 | 0.6355 | **0.8073 ⭐** | 0.7893 | 0.6846 |
| **Customer_Demand_Forecast_Pct** | **0.7327 ⭐** | 0.4079 | 0.7146 | 0.6849 | 0.5106 |
| **Risk_Score** | 0.6263 | 0.2860 | **0.6546 ⭐** | 0.6302 | 0.4733 |
| **Time_to_Next_Migration_Months** | 0.5087 | 0.2156 | **0.5803 ⭐** | 0.4963 | 0.3126 |

---

### 3. Classification Models Benchmark (`Next_Recommended_Phase`)

| Algorithm | Accuracy | Weighted F1 | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **SVM (Support Vector Classifier) ⭐** | **92.50%** | **0.9234** | **0.9237** | **0.9250** |
| **Logistic Regression** | 92.33% | 0.9222 | 0.9226 | 0.9233 |
| **Decision Tree Classifier** | 88.75% | 0.8866 | 0.8866 | 0.8875 |
| **Naive Bayes (GaussianNB)** | 87.00% | 0.8673 | 0.8716 | 0.8700 |
| **KNN Classifier** | 74.75% | 0.7326 | 0.7451 | 0.7475 |

---

## 🏗️ Repository Architecture

```
Automation Migration Strategy/
├── app.py                             # Flask Web Server & Real-Time REST API Backend
├── streamlit_app.py                   # Interactive Streamlit Cloud Dashboard
├── predictor.py                       # ML Inference Engine loading 7 Best Models
├── Automation EDA.ipynb               # Full EDA, Model Benchmarks & Tuning Notebook
├── requirements.txt                   # Project Dependencies
├── templates/
│   └── index.html                     # Dashboard UI (matching design specifications)
├── static/
│   ├── css/style.css                  # Custom styling & clean layout
│   ├── js/app.js                      # Chart.js visualizations & real-time inference
│   └── images/                        # Generated 3D banners & roadmap infographics
└── model_artifacts/                   # Serialized Trained Best Models
    ├── best_models/                   # .joblib files for all 7 targets
    ├── feature_metadata.json          # Input feature definitions & ranges
    └── model_metrics.json             # Benchmark scores & parameter metadata
```

---

## 🛠️ How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application (Recommended)
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

### 3. Launch the Streamlit Dashboard (Alternative)
```bash
streamlit run streamlit_app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser.

---

## 🌐 Live Web Deployment Guide

### Option A: Render / Railway / PythonAnywhere (Flask Dashboard)
1. Push this repository to GitHub.
2. In [Render.com](https://render.com) or [Railway.app](https://railway.app), create a new **Web Service** connected to your repo.
3. Set **Start Command**: `gunicorn app:app` or `python app.py`.
4. Your website will be live with a free HTTPS URL!

### Option B: Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
2. Select your repository, set **Main file path** to `streamlit_app.py`.
3. Click **Deploy!**

---

Developed for Enterprise Automation Strategy & Decision Intelligence.
