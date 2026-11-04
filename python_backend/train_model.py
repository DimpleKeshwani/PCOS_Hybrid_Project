import pickle
import re
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

print("--- Starting PCOS ML Model Training Pipeline ---")
FILE = "pcos_data.csv"


def norm(s):
    return re.sub(r"\s+", "", str(s)).lower()


try:
    df = pd.read_csv(FILE, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(FILE, encoding="latin-1")
df.columns = [norm(c) for c in df.columns]

wanted = {
    "age": "age(", "weight": "weight(kg", "height": "height(cm",
    "cycle": "cycle(r/i", "cycle_length": "cyclelength",
    "weight_gain": "weightgain", "hair_growth": "hairgrowth",
    "skin_darkening": "skindarkening", "pimples": "pimples",
    "follicle_l": "follicleno.(l", "follicle_r": "follicleno.(r",
}

chosen = {}
for friendly, prefix in {**wanted, "target": "pcos("}.items():
    match = [c for c in df.columns if c.startswith(prefix)]
    if not match:
        sys.exit(f"Error: no column starting with '{prefix}'.")
    chosen[friendly] = match[0]

data = df[list(chosen.values())].copy()
data.columns = list(chosen.keys())
data = data.apply(pd.to_numeric, errors="coerce").dropna()
print(f"Rows after cleaning: {len(data)}")

basic = list(wanted.keys())[:9]
advanced = list(wanted.keys())
y = data["target"].astype(int)

bundle = {}
for name, feats in {"basic": basic, "advanced": advanced}.items():
    X = data[feats]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced")
    model.fit(X_tr, y_tr)
    print(f"\n=== {name.upper()} model ({len(feats)} features) ===")
    print(f"Test accuracy: {accuracy_score(y_te, model.predict(X_te)):.3f}")
    print(classification_report(y_te, model.predict(X_te)))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    r = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "recall"])
    print(f"5-fold accuracy {r['test_accuracy'].mean():.3f} | "
          f"recall {r['test_recall'].mean():.3f}")
    bundle[name] = {"model": model, "features": feats}

with open("pcos_model.pkl", "wb") as f:
    pickle.dump(bundle, f)
print("\nSaved pcos_model.pkl (basic + advanced)")
print("--- Process Completed Successfully! ---")