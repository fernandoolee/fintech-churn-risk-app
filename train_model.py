"""Train the demo model on the synthetic dataset and save it as XGBoost JSON (no pickle).

    python train_model.py synthetic_new_users.csv
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import TARGET, build_features, clean_columns

raw = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "synthetic_new_users.csv", sep=";")
y = clean_columns(raw)[TARGET].map({"Install": 0, "Uninstall": 1}).astype(int)
X = build_features(raw)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
spw = (y_tr == 0).sum() / (y_tr == 1).sum()

# hyperparameters from the tuned configuration reported in the paper (Table 1)
model = XGBClassifier(n_estimators=413, max_depth=4, learning_rate=0.0104, min_child_weight=3.27,
                      subsample=0.765, colsample_bytree=0.666, reg_lambda=0.412, reg_alpha=1.698,
                      scale_pos_weight=spw, tree_method="hist", eval_metric="logloss", random_state=42)
model.fit(X_tr, y_tr)
proba = model.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, proba)

# 4 risk tiers by score quantiles on the test set: bottom 40% / next 30% / next 20% / top 10%
cuts = np.quantile(proba, [0.40, 0.70, 0.90]).round(4).tolist()

model.get_booster().save_model("model.json")
with open("model_meta.json", "w") as f:
    json.dump({"features": list(X.columns), "tier_cuts": cuts, "test_roc_auc": round(float(auc), 4),
               "trained_on": "synthetic_new_users.csv (synthetic data)"}, f, indent=2)
print(f"ROC-AUC {auc:.4f} | tier cuts {cuts} | features {X.shape[1]}")
