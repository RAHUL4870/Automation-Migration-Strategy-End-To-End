"""
Enterprise Automation Migration Strategy - Streamlit Interactive Web Application
Modern Decision Intelligence Dashboard powered by 7 Scikit-Learn Best ML Models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import predictor

# Page Configuration
st.set_page_config(
    page_title="Automation Strategy AI | Industry 4.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #060913;
        color: #f8fafc;
    }
    
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 27, 75, 0.8) 100%);
        border: 1px solid rgba(0, 229, 255, 0.3);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #00e5ff 0%, #3b82f6 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
    }
    
    .phase-card {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.2) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px solid #a855f7;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: 0 0 25px rgba(168, 85, 247, 0.25);
    }
    
    .kpi-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .kpi-val {
        font-family: 'Outfit', sans-serif;
        font-size: 26px;
        font-weight: 800;
        color: #00e5ff;
        margin-top: 4px;
    }
    
    .kpi-lbl {
        font-size: 12px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="hero-container">
    <h1 style="font-size: 36px; margin-bottom: 10px;">
        🤖 Enterprise <span class="gradient-text">Automation Migration Strategy</span>
    </h1>
    <p style="font-size: 16px; color: #cbd5e1; max-width: 900px; margin-bottom: 15px;">
        Predictive Decision Intelligence platform evaluating enterprise automation feasibility, forecasting ROI, labor cost savings up to <strong>93.1% R²</strong>, risk index, and phase transitions powered by 7 Scikit-Learn Best ML Models.
    </p>
    <div style="display: flex; gap: 15px; flex-wrap: wrap;">
        <span style="background: rgba(0,229,255,0.15); border: 1px solid #00e5ff; padding: 4px 12px; border-radius: 20px; font-size: 13px; color: #00e5ff; font-weight: 600;">⚡ 7 Best Models Active</span>
        <span style="background: rgba(16,185,129,0.15); border: 1px solid #10b981; padding: 4px 12px; border-radius: 20px; font-size: 13px; color: #10b981; font-weight: 600;">🎯 92.5% Phase Accuracy</span>
        <span style="background: rgba(168,85,247,0.15); border: 1px solid #a855f7; padding: 4px 12px; border-radius: 20px; font-size: 13px; color: #a855f7; font-weight: 600;">📈 93.1% Max R² Score</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar: Presets & Input Controls
st.sidebar.header("🎛️ Factory Parameter Console")

# Preset selector
presets = predictor.INDUSTRY_PRESETS
selected_preset_name = st.sidebar.selectbox(
    "Load Industry Archetype Preset:",
    list(presets.keys()),
    index=0
)
preset_data = presets[selected_preset_name]

# Expandable input sections
with st.sidebar.expander("🏢 1. Company & Industry Profile", expanded=True):
    industry = st.selectbox("Industry Sector", ["Automotive", "Electronics", "Food", "Pharma", "Textile"], 
                            index=["Automotive", "Electronics", "Food", "Pharma", "Textile"].index(preset_data["Industry"]))
    stage = st.selectbox("Product Lifecycle Stage", ["Introduction", "Growth", "Maturity", "Decline"],
                         index=["Introduction", "Growth", "Maturity", "Decline"].index(preset_data["Product_Lifecycle_Stage"]))
    current_auto = st.selectbox("Current Automation Level", ["Continue Manual", "Semi-Automation", "Flexible Automation", "Full Automation"],
                                index=["Continue Manual", "Semi-Automation", "Flexible Automation", "Full Automation"].index(preset_data["Current_Automation_Level"]))

with st.sidebar.expander("⚙️ 2. Operations & Capacity", expanded=False):
    vol = st.slider("Monthly Volume (Units)", 800, 35000, int(preset_data["Monthly_Production_Volume"]), step=100)
    backlog = st.slider("Order Backlog (Units)", 70, 11000, int(preset_data["Order_Backlog_Units"]), step=50)
    defect = st.slider("Defect Rate (%)", 0.1, 22.0, float(preset_data["Defect_Rate_Pct"]), step=0.1)
    cycle_time = st.slider("Cycle Time (Min)", 1.0, 43.0, float(preset_data["Cycle_Time_Min"]), step=0.5)
    skus = st.slider("Product Variety (SKUs)", 2, 315, int(preset_data["Product_Variety_SKUs"]), step=1)
    util = st.slider("Machine Utilization (%)", 20.0, 99.0, float(preset_data["Machine_Utilization_Pct"]), step=0.5)

with st.sidebar.expander("💰 3. Financial & Investment", expanded=False):
    labor_cost = st.slider("Monthly Labor Cost ($k)", 5.0, 135.0, float(preset_data["Labor_Cost_Monthly_USD_k"]), step=1.0)
    machine_cost = st.slider("Machine Cost Estimate ($k)", 20.0, 1000.0, float(preset_data["Machine_Cost_Estimate_USD_k"]), step=5.0)
    capital = st.slider("Capital Available ($k)", 10.0, 1200.0, float(preset_data["Capital_Available_USD_k"]), step=5.0)
    roi_input = st.slider("Baseline ROI (%)", -19.0, 100.0, float(preset_data["ROI_Pct"]), step=1.0)
    payback = st.slider("Payback Target (Years)", 0.2, 15.0, float(preset_data["Payback_Period_Years"]), step=0.1)
    maint_cost = st.slider("Annual Maintenance ($k)", 1.0, 100.0, float(preset_data["Annual_Maintenance_Cost_USD_k"]), step=1.0)

with st.sidebar.expander("👥 4. Workforce & Risk Factors", expanded=False):
    overtime = st.slider("Monthly Overtime (Hours)", 0.0, 290.0, float(preset_data["Overtime_Hours_Monthly"]), step=1.0)
    turnover = st.slider("Turnover Rate (%)", 1.0, 44.0, float(preset_data["Employee_Turnover_Pct"]), step=0.5)
    safety = st.slider("Annual Safety Incidents", 0, 12, int(preset_data["Safety_Incidents_Annual"]), step=1)
    demand_growth = st.slider("Demand Growth (%)", -8.0, 45.0, float(preset_data["Demand_Growth_Pct"]), step=0.5)

# Assemble Input Payload
input_payload = {
    "Industry": industry,
    "Product_Lifecycle_Stage": stage,
    "Current_Automation_Level": current_auto,
    "Monthly_Production_Volume": float(vol),
    "Demand_Growth_Pct": float(demand_growth),
    "Order_Backlog_Units": float(backlog),
    "Labor_Cost_Monthly_USD_k": float(labor_cost),
    "Machine_Cost_Estimate_USD_k": float(machine_cost),
    "Capital_Available_USD_k": float(capital),
    "Defect_Rate_Pct": float(defect),
    "Cycle_Time_Min": float(cycle_time),
    "ROI_Pct": float(roi_input),
    "Payback_Period_Years": float(payback),
    "Product_Variety_SKUs": int(skus),
    "Employee_Turnover_Pct": float(turnover),
    "Annual_Maintenance_Cost_USD_k": float(maint_cost),
    "Machine_Utilization_Pct": float(util),
    "Overtime_Hours_Monthly": float(overtime),
    "Safety_Incidents_Annual": float(safety)
}

# Run Inference
result = predictor.predict_migration_strategy(input_payload)
preds = result["predictions"]

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Real-Time Predictions & Analytics", "📊 ML Benchmarks & R² Leaderboard", "🧭 Migration Roadmap Guide"])

with tab1:
    # Phase Recommendation Banner
    rec_phase = preds["Next_Recommended_Phase"]
    st.markdown(f"""
    <div class="phase-card">
        <span style="font-size: 13px; color: #e9d5ff; font-weight: 700; text-transform: uppercase;">🎯 AI Recommended Strategic Phase</span>
        <h2 style="font-size: 32px; font-weight: 800; color: #ffffff; margin-top: 5px; margin-bottom: 8px;">
            {rec_phase}
        </h2>
        <p style="font-size: 15px; color: #cbd5e1; margin-bottom: 0;">
            Evaluated against trained Support Vector Classifier (92.5% Accuracy) and 6 specialized regression estimators.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 6 Core KPI Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-lbl">🛡️ Migration Risk Score</div>
            <div class="kpi-val" style="color: {'#10b981' if preds['Risk_Score'] < 30 else '#f59e0b' if preds['Risk_Score'] < 55 else '#ef4444'};">
                {preds['Risk_Score']:.1f} / 100
            </div>
            <span style="font-size: 12px; color: #64748b;">Polynomial Reg ($R^2$: 0.655)</span>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-lbl">⏱️ Transition Timeline</div>
            <div class="kpi-val">{preds['Time_to_Next_Migration_Months']:.1f} Mos</div>
            <span style="font-size: 12px; color: #64748b;">Polynomial Reg ($R^2$: 0.580)</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-lbl">💵 Labor Cost Savings</div>
            <div class="kpi-val" style="color: #10b981;">{preds['Labor_Cost_Savings_Pct']:+.1f}%</div>
            <span style="font-size: 12px; color: #64748b;">Polynomial Reg ($R^2$: 0.931)</span>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-lbl">⚡ Productivity Gain</div>
            <div class="kpi-val" style="color: #a855f7;">{preds['Productivity_Improvement_Pct']:+.1f}%</div>
            <span style="font-size: 12px; color: #64748b;">Polynomial Reg ($R^2$: 0.873)</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-lbl">📈 Annual Profit Increase</div>
            <div class="kpi-val" style="color: #f59e0b;">{preds['Expected_Annual_Profit_Increase_Pct']:+.1f}%</div>
            <span style="font-size: 12px; color: #64748b;">Polynomial Reg ($R^2$: 0.807)</span>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-lbl">👥 Customer Demand Forecast</div>
            <div class="kpi-val" style="color: #3b82f6;">{preds['Customer_Demand_Forecast_Pct']:+.1f}%</div>
            <span style="font-size: 12px; color: #64748b;">Linear Regression ($R^2$: 0.733)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Interactive Visualizations
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("🕸️ Facility Capability Radar Profile")
        radar_categories = ['Volume Capacity', 'Capital Budget', 'Labor Savings', 'Productivity', 'Safety Score', 'Risk Readiness']
        norm_vol = min(100, (vol / 25000) * 100)
        norm_cap = min(100, (capital / 800) * 100)
        norm_sav = min(100, max(0, preds['Labor_Cost_Savings_Pct'] * 2))
        norm_prd = min(100, max(0, preds['Productivity_Improvement_Pct'] * 2))
        norm_saf = min(100, max(0, 100 - safety * 10))
        norm_rsk = min(100, max(0, 100 - preds['Risk_Score']))
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[norm_vol, norm_cap, norm_sav, norm_prd, norm_saf, norm_rsk],
            theta=radar_categories,
            fill='toself',
            name='Current Facility Profile',
            line_color='#00e5ff',
            fillcolor='rgba(0, 229, 255, 0.25)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[75, 70, 80, 75, 90, 80],
            theta=radar_categories,
            fill='toself',
            name='Industry 4.0 Benchmark',
            line_color='#a855f7',
            fillcolor='rgba(168, 85, 247, 0.15)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#94a3b8"),
                angularaxis=dict(color="#cbd5e1")
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            margin=dict(l=40, r=40, t=20, b=20),
            height=320
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_chart2:
        st.subheader("📊 Forecast Impact Metric Distribution")
        metrics_df = pd.DataFrame({
            "Metric": ["Labor Savings", "Productivity Gain", "Profit Increase", "Demand Forecast", "Risk Score"],
            "Forecast Value (%)": [
                preds["Labor_Cost_Savings_Pct"],
                preds["Productivity_Improvement_Pct"],
                preds["Expected_Annual_Profit_Increase_Pct"],
                preds["Customer_Demand_Forecast_Pct"],
                preds["Risk_Score"]
            ],
            "Color": ["#10b981", "#a855f7", "#f59e0b", "#3b82f6", "#ef4444"]
        })
        fig_bar = px.bar(
            metrics_df,
            x="Metric",
            y="Forecast Value (%)",
            color="Metric",
            color_discrete_sequence=["#10b981", "#a855f7", "#f59e0b", "#3b82f6", "#ef4444"],
            text_auto='.1f'
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            height=320,
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.header("🏆 Best Machine Learning Models & Leaderboards")
    
    # Selected Best Models
    best_models_data = [
        {"Target Variable": "Labor_Cost_Savings_Pct", "Problem Type": "Regression", "Winning Algorithm": "Polynomial Regression (Degree 2)", "Metric": "R² Score", "Score": "0.9307 (93.1%)"},
        {"Target Variable": "Productivity_Improvement_Pct", "Problem Type": "Regression", "Winning Algorithm": "Polynomial Regression (Degree 2)", "Metric": "R² Score", "Score": "0.8726 (87.3%)"},
        {"Target Variable": "Next_Recommended_Phase", "Problem Type": "Classification", "Winning Algorithm": "Support Vector Machine (SVC)", "Metric": "Accuracy", "Score": "0.9250 (92.5%)"},
        {"Target Variable": "Expected_Annual_Profit_Increase_Pct", "Problem Type": "Regression", "Winning Algorithm": "Polynomial Regression (Degree 2)", "Metric": "R² Score", "Score": "0.8073 (80.7%)"},
        {"Target Variable": "Customer_Demand_Forecast_Pct", "Problem Type": "Regression", "Winning Algorithm": "Linear Regression", "Metric": "R² Score", "Score": "0.7327 (73.3%)"},
        {"Target Variable": "Risk_Score", "Problem Type": "Regression", "Winning Algorithm": "Polynomial Regression (Degree 2)", "Metric": "R² Score", "Score": "0.6546 (65.5%)"},
        {"Target Variable": "Time_to_Next_Migration_Months", "Problem Type": "Regression", "Winning Algorithm": "Polynomial Regression (Degree 2)", "Metric": "R² Score", "Score": "0.5803 (58.0%)"}
    ]
    st.dataframe(pd.DataFrame(best_models_data), use_container_width=True, hide_index=True)
    
    st.markdown("### 📋 Complete 5-Model Regression Benchmarks ($R^2$ Matrix)")
    reg_matrix_data = {
        "Regression Target": ["Labor_Cost_Savings_Pct", "Productivity_Improvement_Pct", "Expected_Annual_Profit_Increase_Pct", "Customer_Demand_Forecast_Pct", "Risk_Score", "Time_to_Next_Migration_Months"],
        "Linear Regression": [0.8608, 0.7978, 0.7524, 0.7327, 0.6263, 0.5087],
        "Decision Tree": [0.8580, 0.7773, 0.6355, 0.4079, 0.2860, 0.2156],
        "Polynomial Reg (Deg 2)": [0.9307, 0.8726, 0.8073, 0.7146, 0.6546, 0.5803],
        "SVR (SVM)": [0.9139, 0.8472, 0.7893, 0.6849, 0.6302, 0.4963],
        "KNN Regressor": [0.7890, 0.7119, 0.6846, 0.5106, 0.4733, 0.3126]
    }
    st.dataframe(pd.DataFrame(reg_matrix_data), use_container_width=True, hide_index=True)
    
    st.markdown("### 🧭 Classification Models Benchmarking (`Next_Recommended_Phase`)")
    clf_matrix_data = {
        "Classification Algorithm": ["SVM (Support Vector Classifier)", "Logistic Regression", "Decision Tree Classifier", "Naive Bayes (GaussianNB)", "KNN Classifier"],
        "Accuracy": ["92.50%", "92.33%", "88.75%", "87.00%", "74.75%"],
        "Weighted F1": [0.9234, 0.9222, 0.8866, 0.8673, 0.7326],
        "Precision": [0.9237, 0.9226, 0.8866, 0.8716, 0.7451],
        "Recall": [0.9250, 0.9233, 0.8875, 0.8700, 0.7475]
    }
    st.dataframe(pd.DataFrame(clf_matrix_data), use_container_width=True, hide_index=True)

with tab3:
    st.header("🧭 Industry 4.0 Migration Roadmap")
    if os.path.exists("static/images/roadmap_infographic.jpg"):
        st.image("static/images/roadmap_infographic.jpg", caption="Strategic Automation Evolution Roadmap (Levels 1-3)", use_container_width=True)
    
    st.markdown("""
    ### Strategic Migration Framework
    - **Level 1 (Manual Workstations)**: High manual touchpoint, batch flow, labor-dependent throughput. Best suited when capital budget is constrained or SKU variety is extreme.
    - **Level 2 (Semi-Automated Robotics)**: Human-robot collaboration, automated material handling, cobot assembly cells. Maximizes cost efficiency with minimal upfront risk.
    - **Level 3 (Autonomous Smart Factory / Flexible Automation)**: End-to-end connected IoT cells, reconfigurable production lines, dynamic AI scheduling. Delivers peak productivity and cost savings.
    """)
