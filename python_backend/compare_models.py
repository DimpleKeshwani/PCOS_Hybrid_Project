import re
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = pd.read_csv("pcos_data.csv", encoding="utf-8-sig")
df.columns = [re.sub(r"\s+", "", str(c)).lower() for c in df.columns]


def col(prefix):
    m = [c for c in df.columns if c.startswith(prefix)]
    if not m:
        raise SystemExit(f"Column starting with '{prefix}' not found")
    return m[0]


cols = {"age": col("age("), "weight": col("weight(kg"), "height": col("height(cm"),
        "cycle": col("cycle(r/i"), "cycle_length": col("cyclelength"),
        "weight_gain": col("weightgain"), "hair_growth": col("hairgrowth"),
        "skin_darkening": col("skindarkening"), "pimples": col("pimples"),
        "target": col("pcos(")}
data = df[list(cols.values())].copy()
data.columns = list(cols.keys())
data = data.apply(pd.to_numeric, errors="coerce").dropna()

X, y = data.drop(columns="target"), data["target"].astype(int)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                             scale_pos_weight=2, eval_metric="logloss", random_state=42),
}

print(f"Rows used: {len(data)}")
for name, model in models.items():
    r = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "recall", "roc_auc"])
    print(f"{name:20s} accuracy {r['test_accuracy'].mean():.3f}  "
          f"recall {r['test_recall'].mean():.3f}  AUC {r['test_roc_auc'].mean():.3f}")