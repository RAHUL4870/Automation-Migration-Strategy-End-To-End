"""
Enterprise Automation Migration Strategy - Web Application Server
Flask Web Server providing REST APIs and rich interactive dashboard UI.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import predictor

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/")
def index():
    """Render main interactive dashboard."""
    metadata = predictor.FEATURE_METADATA
    metrics = predictor.MODEL_METRICS
    presets = predictor.INDUSTRY_PRESETS
    
    # Load real dataset stats
    stats_path = os.path.join(app.static_folder, "data", "dataset_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            dataset_stats = json.load(f)
    else:
        dataset_stats = {}
        
    return render_template("index.html", metadata=metadata, metrics=metrics, presets=presets, stats=dataset_stats)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict all 7 migration targets given input parameters."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input payload provided"}), 400
        
        result = predictor.predict_migration_strategy(data)
        return jsonify({
            "status": "success",
            "data": result
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/presets", methods=["GET"])
def api_presets():
    """Return available industry presets."""
    return jsonify({
        "status": "success",
        "presets": predictor.INDUSTRY_PRESETS
    })

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return comprehensive ML benchmark scores."""
    return jsonify({
        "status": "success",
        "metrics": predictor.MODEL_METRICS,
        "metadata": predictor.FEATURE_METADATA
    })

@app.route("/api/dataset-stats", methods=["GET"])
def api_dataset_stats():
    """Return actual dataset distribution and summary statistics."""
    stats_path = os.path.join(app.static_folder, "data", "dataset_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return jsonify({"status": "success", "stats": stats})
    return jsonify({"status": "error", "message": "Stats file not found"}), 404

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Run What-If scenario simulation on baseline inputs."""
    try:
        payload = request.get_json(force=True)
        baseline = payload.get("baseline", predictor.get_default_input())
        deltas = payload.get("deltas", {})
        
        modified = dict(baseline)
        if "capital_change_pct" in deltas:
            factor = 1.0 + (float(deltas["capital_change_pct"]) / 100.0)
            modified["Capital_Available_USD_k"] = max(10.0, float(baseline["Capital_Available_USD_k"]) * factor)
            
        if "defect_reduction_pct" in deltas:
            factor = 1.0 - (float(deltas["defect_reduction_pct"]) / 100.0)
            modified["Defect_Rate_Pct"] = max(0.1, float(baseline["Defect_Rate_Pct"]) * factor)
            
        if "volume_growth_pct" in deltas:
            factor = 1.0 + (float(deltas["volume_growth_pct"]) / 100.0)
            modified["Monthly_Production_Volume"] = max(500.0, float(baseline["Monthly_Production_Volume"]) * factor)
            
        base_res = predictor.predict_migration_strategy(baseline)
        sim_res = predictor.predict_migration_strategy(modified)
        
        return jsonify({
            "status": "success",
            "baseline_predictions": base_res["predictions"],
            "simulated_predictions": sim_res["predictions"],
            "simulated_risk_level": sim_res["risk_level"],
            "modified_inputs": modified
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "models_loaded": len(predictor.get_models())})

if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Automation Migration Strategy Dashboard running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
