#!/usr/bin/env python3
"""
Baseline: 5-fold CV with min-max scaling and LogisticRegression.
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("qml_dataset copy.csv")          # same file as the QML runs
X  = df.drop("label", axis=1).values
y  = df["label"].astype(int).values               # keep 0/1 labels
indices = np.arange(len(df))                      # track original rows

# ─── 5-fold cross-validation ────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_acc = []
all_preds = []

for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
    print(f"\n—— fold {fold}/5 —————————————————————————")

    # pipeline = scaler + logistic regression
    clf = make_pipeline(
        MinMaxScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs"),
    )
    clf.fit(X[tr], y[tr])

    acc = clf.score(X[te], y[te])
    fold_acc.append(acc)
    print(f"accuracy = {acc:.3f}")

    y_pred = clf.predict(X[te])

    print("confusion matrix:")
    print(confusion_matrix(y[te], y_pred))
    print(classification_report(y[te], y_pred, digits=3))

    all_preds.append(pd.DataFrame({
        "fold":       fold,
        "index":      indices[te],
        "true_label": y[te],
        "pred_label": y_pred,
    }))

# ─── summary & export ───────────────────────────────────────────────────────
print("\n=== summary ===")
print("fold accuracies:", np.round(fold_acc, 3))
print("mean ± std =", np.mean(fold_acc), "±", np.std(fold_acc))

pd.concat(all_preds).to_csv("logreg_cv_results.csv", index=False)
print("\nPer-sample predictions saved to logreg_cv_results.csv")
