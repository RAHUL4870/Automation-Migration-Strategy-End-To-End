/**
 * Enterprise Automation Migration Strategy - Dashboard Interactive JavaScript
 * Real-time 7-Model Prediction Engine, Real Dataset Visualizations & Feature Importance.
 */

document.addEventListener('DOMContentLoaded', () => {

    // Chart Instances
    let realReadinessChart = null;
    let realRiskChart = null;
    let featureImportanceChart = null;
    let realScatterChart = null;

    // DOM Elements - Prediction Engine
    const form = document.getElementById('predictionEngineForm');
    const computeBtn = document.getElementById('btn-run-prediction');
    const presetBtns = document.querySelectorAll('.btn-preset');

    // Industry Presets Store
    let presetsData = {};

    // 1. Fetch Presets & Real Dataset Stats
    async function initDashboardData() {
        try {
            // Load Presets
            const resPresets = await fetch('/api/presets');
            const dataPresets = await resPresets.json();
            if (dataPresets.status === 'success') {
                presetsData = dataPresets.presets;
            }

            // Load Dataset Stats for Charts
            const resStats = await fetch('/api/dataset-stats');
            const dataStats = await resStats.json();
            if (dataStats.status === 'success') {
                initChartsWithRealData(dataStats.stats);
            }
        } catch (err) {
            console.error('Error initializing dashboard data:', err);
        }
    }

    // 2. Preset Click Listener
    presetBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            presetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const presetKey = btn.dataset.preset;
            if (presetsData[presetKey]) {
                applyPreset(presetsData[presetKey]);
            }
        });
    });

    function applyPreset(p) {
        if (p.Industry) document.getElementById('Industry').value = p.Industry;
        if (p.Product_Lifecycle_Stage) document.getElementById('Product_Lifecycle_Stage').value = p.Product_Lifecycle_Stage;
        if (p.Current_Automation_Level) document.getElementById('Current_Automation_Level').value = p.Current_Automation_Level;
        if (p.Monthly_Production_Volume) document.getElementById('Monthly_Production_Volume').value = p.Monthly_Production_Volume;
        if (p.Labor_Cost_Monthly_USD_k) document.getElementById('Labor_Cost_Monthly_USD_k').value = p.Labor_Cost_Monthly_USD_k;
        if (p.Machine_Cost_Estimate_USD_k) document.getElementById('Machine_Cost_Estimate_USD_k').value = p.Machine_Cost_Estimate_USD_k;
        if (p.Capital_Available_USD_k) document.getElementById('Capital_Available_USD_k').value = p.Capital_Available_USD_k;
        if (p.Defect_Rate_Pct) document.getElementById('Defect_Rate_Pct').value = p.Defect_Rate_Pct;
        if (p.Cycle_Time_Min) document.getElementById('Cycle_Time_Min').value = p.Cycle_Time_Min;
        if (p.Machine_Utilization_Pct) document.getElementById('Machine_Utilization_Pct').value = p.Machine_Utilization_Pct;

        runPrediction();
    }

    // 3. Extract Form Payload
    function getFormPayload() {
        return {
            Industry: document.getElementById('Industry').value,
            Product_Lifecycle_Stage: document.getElementById('Product_Lifecycle_Stage').value,
            Current_Automation_Level: document.getElementById('Current_Automation_Level').value,
            Monthly_Production_Volume: parseFloat(document.getElementById('Monthly_Production_Volume').value) || 12000,
            Labor_Cost_Monthly_USD_k: parseFloat(document.getElementById('Labor_Cost_Monthly_USD_k').value) || 55.0,
            Machine_Cost_Estimate_USD_k: parseFloat(document.getElementById('Machine_Cost_Estimate_USD_k').value) || 320.0,
            Capital_Available_USD_k: parseFloat(document.getElementById('Capital_Available_USD_k').value) || 400.0,
            Defect_Rate_Pct: parseFloat(document.getElementById('Defect_Rate_Pct').value) || 2.5,
            Cycle_Time_Min: parseFloat(document.getElementById('Cycle_Time_Min').value) || 12.0,
            Machine_Utilization_Pct: parseFloat(document.getElementById('Machine_Utilization_Pct').value) || 78.0,
            ROI_Pct: 32.0,
            Payback_Period_Years: 3.6,
            Product_Variety_SKUs: 45,
            Employee_Turnover_Pct: 12.0,
            Annual_Maintenance_Cost_USD_k: 22.0,
            Overtime_Hours_Monthly: 45.0,
            Safety_Incidents_Annual: 2.0,
            Demand_Growth_Pct: 12.0,
            Order_Backlog_Units: 1800.0
        };
    }

    // 4. Run Real-Time Prediction API
    async function runPrediction() {
        const payload = getFormPayload();
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const json = await res.json();
            if (json.status === 'success') {
                updatePredictionUI(json.data);
            }
        } catch (err) {
            console.error('Prediction API Error:', err);
        }
    }

    // Debounced automatic prediction on input change
    let debounceTimer = null;
    const formInputs = form.querySelectorAll('input, select');
    formInputs.forEach(input => {
        input.addEventListener('input', () => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(runPrediction, 200);
        });
    });

    if (computeBtn) {
        computeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            runPrediction();
        });
    }

    // 5. Update Prediction Outputs
    function updatePredictionUI(data) {
        const p = data.predictions;

        // Phase Box
        document.getElementById('phaseHeroTitle').textContent = p.Next_Recommended_Phase;
        const riskBadge = document.getElementById('phaseRiskBadge');
        riskBadge.textContent = `Risk Level: ${data.risk_level}`;
        
        // 4 Core KPI Values
        document.getElementById('predSavingsVal').textContent = `${p.Labor_Cost_Savings_Pct > 0 ? '+' : ''}${p.Labor_Cost_Savings_Pct.toFixed(1)}%`;
        document.getElementById('predProdVal').textContent = `${p.Productivity_Improvement_Pct > 0 ? '+' : ''}${p.Productivity_Improvement_Pct.toFixed(1)}%`;
        document.getElementById('predProfitVal').textContent = `${p.Expected_Annual_Profit_Increase_Pct > 0 ? '+' : ''}${p.Expected_Annual_Profit_Increase_Pct.toFixed(1)}%`;
        document.getElementById('predRiskVal').textContent = `${p.Risk_Score.toFixed(1)} / 100`;

        // Secondary predictions
        document.getElementById('predTimelineVal').textContent = `${p.Time_to_Next_Migration_Months.toFixed(1)} Months`;
        document.getElementById('predDemandVal').textContent = `${p.Customer_Demand_Forecast_Pct > 0 ? '+' : ''}${p.Customer_Demand_Forecast_Pct.toFixed(1)}%`;
    }

    // =========================================================================
    // 6. INITIALIZE CHARTS WITH REAL DATASET METRICS
    // =========================================================================
    function initChartsWithRealData(stats) {
        
        // --- CHART 1: Real Industry Readiness Stacked Bar ---
        const ctxBar = document.getElementById('realReadinessBarChart').getContext('2d');
        const indData = stats.industry_readiness || {
            industries: ['Automotive', 'Electronics', 'Food', 'Pharma', 'Textile'],
            recommended: [645, 680, 520, 410, 379],
            stay: [828, 783, 674, 522, 559]
        };

        realReadinessChart = new Chart(ctxBar, {
            type: 'bar',
            data: {
                labels: indData.industries,
                datasets: [
                    {
                        label: 'Recommended to Migrate',
                        data: indData.recommended,
                        backgroundColor: '#10b981',
                        borderRadius: 0,
                        barPercentage: 0.55
                    },
                    {
                        label: 'Retain Current Level',
                        data: indData.stay,
                        backgroundColor: '#cbd5e1',
                        borderRadius: { topLeft: 4, topRight: 4 },
                        barPercentage: 0.55
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(c) {
                                return ` ${c.dataset.label}: ${c.raw.toLocaleString()} facilities`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        stacked: true,
                        grid: { display: false },
                        ticks: { color: '#64748b', font: { size: 11, family: "'Plus Jakarta Sans', sans-serif" } }
                    },
                    y: {
                        stacked: true,
                        beginAtZero: true,
                        grid: { color: '#f1f5f9' },
                        ticks: { color: '#94a3b8', font: { size: 10.5 } },
                        title: {
                            display: true,
                            text: 'Number of Enterprises (Total: 6,000)',
                            color: '#94a3b8',
                            font: { size: 10 }
                        }
                    }
                }
            }
        });

        // --- CHART 2: Real Risk Score Donut Chart ---
        const ctxDonut = document.getElementById('realRiskDonutChart').getContext('2d');
        const rd = stats.risk_distribution || { low_pct: 31.4, med_pct: 51.8, high_pct: 14.6, crit_pct: 2.2 };

        realRiskChart = new Chart(ctxDonut, {
            type: 'doughnut',
            data: {
                labels: ['Low Risk (0–20)', 'Moderate (21–40)', 'High Risk (41–60)', 'Critical (> 60)'],
                datasets: [{
                    data: [rd.low_pct, rd.med_pct, rd.high_pct, rd.crit_pct],
                    backgroundColor: ['#10b981', '#f59e0b', '#f97316', '#ef4444'],
                    borderWidth: 2,
                    borderColor: '#ffffff',
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '72%',
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(c) {
                                return ` ${c.label}: ${c.raw}%`;
                            }
                        }
                    }
                }
            }
        });

        // --- CHART 3: Feature Importance (Real Extracted Feature Drivers) ---
        const ctxFI = document.getElementById('featureImportanceChart').getContext('2d');
        const fiLabels = [
            'Current Level: Continue Manual',
            'Baseline ROI (%)',
            'Current Level: Full Automation',
            'Product Variety (SKUs)',
            'Monthly Labor Cost ($k)',
            'Monthly Production Volume',
            'Cycle Time (Min)',
            'Machine Cost ($k)',
            'Machine Utilization (%)',
            'Defect Rate (%)'
        ];
        const fiScores = [0.637, 0.222, 0.064, 0.020, 0.014, 0.012, 0.010, 0.009, 0.007, 0.005];

        featureImportanceChart = new Chart(ctxFI, {
            type: 'bar',
            data: {
                labels: fiLabels,
                datasets: [{
                    label: 'Feature Importance Weight',
                    data: fiScores,
                    backgroundColor: [
                        '#2563eb', '#3b82f6', '#60a5fa', '#10b981', '#34d399',
                        '#8b5cf6', '#a78bfa', '#f59e0b', '#fbbf24', '#f97316'
                    ],
                    borderRadius: 4,
                    barPercentage: 0.65
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(c) {
                                return ` Importance: ${(c.raw * 100).toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 0.7,
                        grid: { color: '#f1f5f9' },
                        ticks: {
                            callback: function(v) { return (v * 100).toFixed(0) + '%'; },
                            color: '#94a3b8',
                            font: { size: 10 }
                        },
                        title: { display: true, text: 'Relative Feature Impact (%)', color: '#94a3b8', font: { size: 10 } }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#0f172a', font: { size: 11, family: "'Plus Jakarta Sans', sans-serif" } }
                    }
                }
            }
        });

        // --- CHART 4: Real Dataset Scatter (ROI vs. Risk Score) ---
        const ctxScatter = document.getElementById('realDatasetScatterChart').getContext('2d');
        const rawPoints = stats.scatter_sample || [];

        const lowRiskPoints = [];
        const medRiskPoints = [];
        const highRiskPoints = [];

        rawPoints.forEach(pt => {
            const item = { x: pt.Risk_Score, y: pt.ROI_Pct, label: `${pt.Industry} (${pt.Next_Recommended_Phase})` };
            if (pt.Risk_Score <= 25) lowRiskPoints.push(item);
            else if (pt.Risk_Score <= 45) medRiskPoints.push(item);
            else highRiskPoints.push(item);
        });

        realScatterChart = new Chart(ctxScatter, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Low Risk (0–25)',
                        data: lowRiskPoints,
                        backgroundColor: '#10b981',
                        borderColor: '#059669',
                        pointRadius: 5,
                        pointHoverRadius: 8
                    },
                    {
                        label: 'Medium Risk (26–45)',
                        data: medRiskPoints,
                        backgroundColor: '#f59e0b',
                        borderColor: '#d97706',
                        pointRadius: 5,
                        pointHoverRadius: 8
                    },
                    {
                        label: 'High Risk (> 45)',
                        data: highRiskPoints,
                        backgroundColor: '#ef4444',
                        borderColor: '#dc2626',
                        pointRadius: 6,
                        pointHoverRadius: 9
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: { boxWidth: 8, font: { size: 11 } }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(c) {
                                const pt = c.raw;
                                return ` ${pt.label}: Risk Score ${pt.x.toFixed(1)}, ROI ${pt.y.toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        min: 0,
                        max: 85,
                        grid: { color: '#f1f5f9' },
                        ticks: { color: '#94a3b8', font: { size: 10 } },
                        title: { display: true, text: 'Enterprise Risk Score (0–100)', color: '#94a3b8', font: { size: 10 } }
                    },
                    y: {
                        min: -20,
                        max: 100,
                        grid: { color: '#f1f5f9' },
                        ticks: { color: '#94a3b8', font: { size: 10 } },
                        title: { display: true, text: 'Baseline ROI (%)', color: '#94a3b8', font: { size: 10 } }
                    }
                }
            }
        });
    }

    // Initialize Application
    initDashboardData().then(() => {
        runPrediction();
    });
});
