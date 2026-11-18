import os
import pickle

import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

with open(os.path.join(BASE, "pcos_model.pkl"), "rb") as f:
    models = pickle.load(f)

explainers = {k: shap.TreeExplainer(v["model"]) for k, v in models.items()}
FOLLICLE = ["follicle_l", "follicle_r"]


def pcos_contributions(sv):
    """Per-feature contributions toward the PCOS class (handles old and new shap formats)."""
    if isinstance(sv, list):          # older shap: [class 0, class 1]
        arr = np.asarray(sv[1])
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:             # newer shap: (samples, features, classes)
            arr = arr[:, :, 1]
    return arr[0]


@app.route("/")
def home():
    return render_template("index.html", page="home")


@app.route("/learn")
def learn():
    return render_template("learn.html", page="learn")


@app.route("/screening")
def screening():
    return render_template("screening.html", page="screening")


@app.route("/result")
def result():
    return render_template("result.html", page="result")


@app.route("/doctors")
def doctors():
    return render_template("doctors.html", page="doctors")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    basic = models["basic"]["features"]
    missing = [f for f in basic if f not in data]
    if missing:
        return jsonify(error="Missing fields: " + ", ".join(missing)), 400

    key = "advanced" if all(data.get(f) not in (None, "") for f in FOLLICLE) else "basic"
    m = models[key]
    try:
        row = {f: float(data[f]) for f in m["features"]}
    except (TypeError, ValueError):
        return jsonify(error="All values must be numbers"), 400

    X = pd.DataFrame([row])[m["features"]]
    prob = float(m["model"].predict_proba(X)[0][1])
    level = "Low" if prob < 0.30 else "Moderate" if prob < 0.60 else "High"

    contrib = pcos_contributions(explainers[key].shap_values(X, check_additivity=False))
    explain = sorted(
        [{"feature": f, "impact": round(float(c) * 100, 1), "value": row[f]}
         for f, c in zip(m["features"], contrib)],
        key=lambda t: -abs(t["impact"]),
    )[:6]

    return jsonify(mode=key, risk_percent=round(prob * 100, 1),
                   risk_level=level, explain=explain)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)