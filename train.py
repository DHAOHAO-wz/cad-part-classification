"""Train lightweight classifiers on hand-engineered CAD part features.

Baselines: Random Forest, SVM, k-NN, small MLP (scikit-learn).

Run:  python train.py
Outputs:
  - train_log.txt                  full run log
  - results/metrics.csv            per-baseline accuracy / weighted-F1 / latency
  - assets/confusion_matrix.png    confusion matrix of the best model
  - assets/feature_importance.png  Random Forest feature importance
"""

import csv
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from data import make_dataset, CATEGORIES
from features import extract_features, FEATURE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
RESULTS = os.path.join(HERE, "results")
os.makedirs(ASSETS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

# ----------------------------- experiment settings ----------------------------- #
RANDOM_STATE = 42
N_PER_CLASS = 120          # samples per category (synthetic dataset)
N_POINTS = 2048            # points per sampled point cloud
TEST_SIZE = 0.25          # held-out test ratio (stratified split)

MODEL_HYPERPARAMS = {
    "RandomForest": {"n_estimators": 200},
    "SVM (RBF)": {"kernel": "rbf", "C": 10},
    "k-NN": {"n_neighbors": 7},
    "MLP": {"hidden_layer_sizes": (32, 16), "max_iter": 800},
}


def build_models():
    """Return the four lightweight baselines with fixed hyperparameters."""
    return {
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE, **MODEL_HYPERPARAMS["RandomForest"]),
        "SVM (RBF)": SVC(
            random_state=RANDOM_STATE, **MODEL_HYPERPARAMS["SVM (RBF)"]),
        "k-NN": KNeighborsClassifier(**MODEL_HYPERPARAMS["k-NN"]),
        "MLP": MLPClassifier(
            random_state=RANDOM_STATE, **MODEL_HYPERPARAMS["MLP"]),
    }


def main():
    t0 = time.time()
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=== CAD part classification prototype ===")
    log(f"scikit-learn / numpy / matplotlib pipeline, seed={RANDOM_STATE}")

    # 1. data (synthetic placeholder; replace with real CAD dataset later)
    clouds, labels = make_dataset(n_per_class=N_PER_CLASS, n_points=N_POINTS,
                                  seed=RANDOM_STATE)
    log(f"\n[1] dataset: {clouds.shape[0]} samples x {clouds.shape[1]} points, "
        f"categories = {CATEGORIES}")

    # 2. feature engineering
    X = np.array([extract_features(pc) for pc in clouds])
    log(f"[2] feature matrix: {X.shape}, features = {len(FEATURE_NAMES)}")
    y = labels

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    log(f"    train={len(Xtr)}  test={len(Xte)} (stratified split)")

    # 3. baselines
    models = build_models()
    results = {}
    metric_rows = []
    log("\n[3] baseline comparison on held-out test set:")
    for name, clf in models.items():
        t1 = time.time()
        clf.fit(Xtr_s, ytr)
        pred = clf.predict(Xte_s)
        dt = time.time() - t1
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="weighted")
        results[name] = (acc, f1, pred)
        metric_rows.append({
            "model": name, "accuracy": f"{acc:.4f}",
            "weighted_f1": f"{f1:.4f}", "fit_predict_seconds": f"{dt:.3f}",
        })
        log(f"    {name:14s}  accuracy={acc:.4f}  weighted-F1={f1:.4f}")

    best_name = max(results, key=lambda k: results[k][0])
    best_acc, best_f1, best_pred = results[best_name]
    log(f"\n    best model: {best_name} (acc={best_acc:.4f}, F1={best_f1:.4f})")

    log("\n[4] classification report (best model):")
    log(classification_report(yte, best_pred, target_names=CATEGORIES, digits=3))

    # 4. confusion matrix figure
    cm = confusion_matrix(yte, best_pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(CATEGORIES, fontsize=9)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=10)
    ax.set_title(f"Confusion matrix - {best_name}", fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    cm_path = os.path.join(ASSETS, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    log(f"    saved {os.path.relpath(cm_path, HERE)}")

    # 5. feature importance from Random Forest
    rf = models["RandomForest"]
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh([FEATURE_NAMES[i] for i in order][::-1],
            imp[order][::-1], color="#5385BD")
    ax.set_title("Random Forest feature importance", fontsize=11)
    ax.set_xlabel("importance")
    fig.tight_layout()
    fi_path = os.path.join(ASSETS, "feature_importance.png")
    fig.savefig(fi_path, dpi=150)
    plt.close(fig)
    log(f"    saved {os.path.relpath(fi_path, HERE)}")

    # 6. structured metrics table
    metrics_path = os.path.join(RESULTS, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "accuracy", "weighted_f1", "fit_predict_seconds"])
        writer.writeheader()
        writer.writerows(metric_rows)
    log(f"    saved {os.path.relpath(metrics_path, HERE)}")

    log(f"\n[done] elapsed {time.time() - t0:.1f}s")

    with open(os.path.join(HERE, "train_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    return 0


if __name__ == "__main__":
    sys.exit(main())
