"""
Predictor Module for Enterprise Automation Migration Strategy
Loads the 7 trained Best ML Models (6 Regression + 1 Classification)
and provides inference, scenario analysis, and feature validation APIs.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "best_models")

# Load Feature Metadata
with open(os.path.join(ARTIFACTS_DIR, "feature_metadata.json"), "r", encoding="utf-8") as f:
    FEATURE_METADATA = json.load(f)

# Load Model Metrics
with open(os.path.join(ARTIFACTS_DIR, "model_metrics.json"), "r", encoding="utf-8") as f:
    MODEL_METRICS = json.load(f)

FEATURE_COLUMNS = FEATURE_METADATA["feature_columns"]
CATEGORICAL_FEATURES = FEATURE_METADATA["categorical_features"]
NUMERICAL_FEATURES = FEATURE_METADATA["numerical_features"]
REGRESSION_TARGETS = FEATURE_METADATA["regression_targets"]
CLASSIFICATION_TARGET = FEATURE_METADATA["classification_target"]

# Lazy-loaded model dictionary
_MODELS: Dict[str, Any] = {}

def get_models() -> Dict[str, Any]:
    """Load and cache the trained ML models."""
    global _MODELS
    if not _MODELS:
        for target in REGRESSION_TARGETS:
            model_path = os.path.join(MODELS_DIR, f"{target}.joblib")
            _MODELS[target] = joblib.load(model_path)
        
        clf_path = os.path.join(MODELS_DIR, f"{CLASSIFICATION_TARGET}.joblib")
        _MODELS[CLASSIFICATION_TARGET] = joblib.load(clf_path)
    return _MODELS

# Default baseline presets for quick experimentation
INDUSTRY_PRESETS = {
    "Automotive Assembly (High Vol / Semi-Auto)": {
        "Industry": "Automotive",
        "Product_Lifecycle_Stage": "Maturity",
        "Current_Automation_Level": "Semi-Automation",
        "Monthly_Production_Volume": 15000.0,
        "Demand_Growth_Pct": 8.5,
        "Order_Backlog_Units": 2400.0,
        "Labor_Cost_Monthly_USD_k": 78.5,
        "Machine_Cost_Estimate_USD_k": 450.0,
        "Capital_Available_USD_k": 600.0,
        "Defect_Rate_Pct": 2.4,
        "Cycle_Time_Min": 12.0,
        "ROI_Pct": 35.0,
        "Payback_Period_Years": 3.8,
        "Product_Variety_SKUs": 45,
        "Employee_Turnover_Pct": 12.0,
        "Annual_Maintenance_Cost_USD_k": 28.0,
        "Machine_Utilization_Pct": 82.5,
        "Overtime_Hours_Monthly": 65.0,
        "Safety_Incidents_Annual": 2.0
    },
    "Electronics PCB (High Variety / Manual)": {
        "Industry": "Electronics",
        "Product_Lifecycle_Stage": "Growth",
        "Current_Automation_Level": "Continue Manual",
        "Monthly_Production_Volume": 8500.0,
        "Demand_Growth_Pct": 22.0,
        "Order_Backlog_Units": 1800.0,
        "Labor_Cost_Monthly_USD_k": 55.0,
        "Machine_Cost_Estimate_USD_k": 280.0,
        "Capital_Available_USD_k": 320.0,
        "Defect_Rate_Pct": 4.8,
        "Cycle_Time_Min": 8.5,
        "ROI_Pct": 48.0,
        "Payback_Period_Years": 2.6,
        "Product_Variety_SKUs": 120,
        "Employee_Turnover_Pct": 18.5,
        "Annual_Maintenance_Cost_USD_k": 15.0,
        "Machine_Utilization_Pct": 65.0,
        "Overtime_Hours_Monthly": 85.0,
        "Safety_Incidents_Annual": 4.0
    },
    "Pharma Sterile Packaging (Strict Quality / Semi-Auto)": {
        "Industry": "Pharma",
        "Product_Lifecycle_Stage": "Growth",
        "Current_Automation_Level": "Semi-Automation",
        "Monthly_Production_Volume": 6200.0,
        "Demand_Growth_Pct": 16.5,
        "Order_Backlog_Units": 950.0,
        "Labor_Cost_Monthly_USD_k": 62.0,
        "Machine_Cost_Estimate_USD_k": 520.0,
        "Capital_Available_USD_k": 750.0,
        "Defect_Rate_Pct": 0.8,
        "Cycle_Time_Min": 16.0,
        "ROI_Pct": 42.0,
        "Payback_Period_Years": 3.2,
        "Product_Variety_SKUs": 25,
        "Employee_Turnover_Pct": 8.0,
        "Annual_Maintenance_Cost_USD_k": 36.0,
        "Machine_Utilization_Pct": 88.0,
        "Overtime_Hours_Monthly": 30.0,
        "Safety_Incidents_Annual": 1.0
    },
    "Food & Beverage Packaging (Continuous Flow)": {
        "Industry": "Food",
        "Product_Lifecycle_Stage": "Maturity",
        "Current_Automation_Level": "Flexible Automation",
        "Monthly_Production_Volume": 22000.0,
        "Demand_Growth_Pct": 5.0,
        "Order_Backlog_Units": 3100.0,
        "Labor_Cost_Monthly_USD_k": 42.0,
        "Machine_Cost_Estimate_USD_k": 360.0,
        "Capital_Available_USD_k": 400.0,
        "Defect_Rate_Pct": 1.9,
        "Cycle_Time_Min": 5.2,
        "ROI_Pct": 29.0,
        "Payback_Period_Years": 4.1,
        "Product_Variety_SKUs": 60,
        "Employee_Turnover_Pct": 14.0,
        "Annual_Maintenance_Cost_USD_k": 24.0,
        "Machine_Utilization_Pct": 92.0,
        "Overtime_Hours_Monthly": 45.0,
        "Safety_Incidents_Annual": 2.0
    },
    "Textile Weaving Mill (Labor Intensive / Manual)": {
        "Industry": "Textile",
        "Product_Lifecycle_Stage": "Maturity",
        "Current_Automation_Level": "Continue Manual",
        "Monthly_Production_Volume": 4500.0,
        "Demand_Growth_Pct": 3.2,
        "Order_Backlog_Units": 1100.0,
        "Labor_Cost_Monthly_USD_k": 38.0,
        "Machine_Cost_Estimate_USD_k": 180.0,
        "Capital_Available_USD_k": 150.0,
        "Defect_Rate_Pct": 6.2,
        "Cycle_Time_Min": 22.0,
        "ROI_Pct": 18.5,
        "Payback_Period_Years": 5.5,
        "Product_Variety_SKUs": 85,
        "Employee_Turnover_Pct": 22.0,
        "Annual_Maintenance_Cost_USD_k": 12.0,
        "Machine_Utilization_Pct": 58.0,
        "Overtime_Hours_Monthly": 75.0,
        "Safety_Incidents_Annual": 5.0
    }
}

def get_default_input() -> Dict[str, Any]:
    """Return median default input values."""
    defaults = {}
    defaults["Industry"] = "Electronics"
    defaults["Product_Lifecycle_Stage"] = "Maturity"
    defaults["Current_Automation_Level"] = "Semi-Automation"
    for col, stats in FEATURE_METADATA["feature_ranges"].items():
        defaults[col] = float(stats["median"])
    return defaults

def validate_and_prepare_input(input_data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
    """Validate and convert input dictionary/DataFrame into aligned feature DataFrame."""
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Fill missing features with defaults
    defaults = get_default_input()
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = defaults.get(col, 0.0)
            
    # Reorder columns to exact training order
    df = df[FEATURE_COLUMNS]
    
    # Convert data types
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(defaults[col])
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
        
    return df

def predict_migration_strategy(input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    """
    Run inference across all 7 best trained models simultaneously.
    Returns:
        - predictions: dict of all 6 regression predictions and 1 classification prediction
        - phase_probabilities: class probability distribution (if supported)
        - risk_category: Low / Moderate / High / Critical classification
        - executive_insights: automated strategic actionable takeaway
    """
    df_input = validate_and_prepare_input(input_data)
    models = get_models()
    
    predictions = {}
    
    # 1. Regression Predictions
    for target in REGRESSION_TARGETS:
        val = float(models[target].predict(df_input)[0])
        predictions[target] = round(val, 2)
        
    # 2. Classification Prediction
    clf_model = models[CLASSIFICATION_TARGET]
    recommended_phase = str(clf_model.predict(df_input)[0])
    predictions[CLASSIFICATION_TARGET] = recommended_phase
    
    # Class probabilities if available
    phase_probs = {}
    if hasattr(clf_model, "predict_proba"):
        probs = clf_model.predict_proba(df_input)[0]
        classes = clf_model.classes_
        for cls_name, prob in zip(classes, probs):
            phase_probs[cls_name] = round(float(prob) * 100, 1)
            
    # Strategic Risk Categorization
    risk_score = predictions["Risk_Score"]
    if risk_score < 30:
        risk_level = "Low Risk"
        risk_badge = "success"
    elif risk_score < 55:
        risk_level = "Moderate Risk"
        risk_badge = "warning"
    elif risk_score < 75:
        risk_level = "High Risk"
        risk_badge = "danger"
    else:
        risk_level = "Critical Risk"
        risk_badge = "critical"
        
    # Generate Strategic Insights
    insights = []
    if predictions["Labor_Cost_Savings_Pct"] > 30:
        insights.append(f"High labor efficiency opportunity with projected {predictions['Labor_Cost_Savings_Pct']}% cost reduction.")
    if predictions["Productivity_Improvement_Pct"] > 25:
        insights.append(f"Strong throughput boost: {predictions['Productivity_Improvement_Pct']}% productivity gain expected.")
    if predictions["Time_to_Next_Migration_Months"] < 24:
        insights.append(f"Accelerated roadmap: Transition recommended within {predictions['Time_to_Next_Migration_Months']} months.")
    else:
        insights.append(f"Stabilized timeline: Estimated transition readiness in {predictions['Time_to_Next_Migration_Months']} months.")
        
    return {
        "predictions": predictions,
        "phase_probabilities": phase_probs,
        "risk_level": risk_level,
        "risk_badge": risk_badge,
        "insights": insights,
        "model_metadata": MODEL_METRICS["best_selected_models"]
    }
