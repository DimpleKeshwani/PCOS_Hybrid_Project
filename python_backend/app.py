import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

with open(os.path.join(BASE, "pcos_model.pkl"), "rb") as f:
    models = pickle.load(f)

FOLLICLE = ["follicle_l", "follicle_r"]


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

    prob = float(m["model"].predict_proba(pd.DataFrame([row])[m["features"]])[0][1])
    level = "Low" if prob < 0.30 else "Moderate" if prob < 0.60 else "High"
    imp = sorted(zip(m["features"], m["model"].feature_importances_), key=lambda t: -t[1])[:6]
    return jsonify(
        mode=key, risk_percent=round(prob * 100, 1), risk_level=level,
        importance=[{"feature": f, "value": round(float(v) * 100, 1)} for f, v in imp],
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)