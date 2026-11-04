import re

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

df = pd.read_csv("pcos_data.csv", encoding="utf-8-sig")
df.columns = [re.sub(r"\s+", "", str(c)).lower() for c in df.columns]


def col(prefix):
    m = [c for c in df.columns if c.startswith(prefix)]
    if not m:
        raise SystemExit(f"Column starting with '{prefix}' not found")
    return m[0]


cols = {
    "age": col("age("), "weight": col("weight(kg"), "height": col("height(cm"),
    "cycle": col("cycle(r/i"), "cycle_length": col("cyclelength"),
    "weight_gain": col("weightgain"), "hair_growth": col("hairgrowth"),
    "skin_darkening": col("skindarkening"), "pimples": col("pimples"),
    "hair_loss": col("hairloss"), "fast_food": col("fastfood"),
    "exercise": col("reg.exercise"), "waist": col("waist(inch"),
    "hip": col("hip(inch"),
    "follicle_l": col("follicleno.(l"), "follicle_r": col("follicleno.(r"),
    "target": col("pcos("),
}

data = df[list(cols.values())].copy()
data.columns = list(cols.keys())
data = data.apply(pd.to_numeric, errors="coerce").dropna()
print("Rows used:", len(data))

base = ["age", "weight", "height", "cycle", "cycle_length",
        "weight_gain", "hair_growth", "skin_darkening", "pimples"]
sets = {
    "A: current 9 features": base,
    "B: + hair loss, fast food, exercise": base + ["hair_loss", "fast_food", "exercise"],
    "C: B + waist, hip": base + ["hair_loss", "fast_food", "exercise", "waist", "hip"],
    "D: C + follicle counts (needs ultrasound)": base + [
        "hair_loss", "fast_food", "exercise", "waist", "hip",
        "follicle_l", "follicle_r"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=300, random_state=42,
                               class_weight="balanced")

for name, feats in sets.items():
    r = cross_validate(model, data[feats], data["target"].astype(int), cv=cv,
                       scoring=["accuracy", "recall"])
    print(f"{name:45s} accuracy {r['test_accuracy'].mean():.3f}"
          f"  recall {r['test_recall'].mean():.3f}")