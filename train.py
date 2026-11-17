"""Train lightweight classifiers on hand-engineered CAD part features.

Baselines: Random Forest, SVM, k-NN, small MLP (scikit-learn).
An optional minimal PointNet-style network can be added later in PyTorch.

Run:  python3 train.py
Outputs: train_log.txt, assets/confusion_matrix.png, assets/feature_importance.png
"""

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
os.makedirs(ASSETS, exist_ok=True)

RANDOM_STATE = 42


def main():
    t0 = time.time()
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=== CAD part classification prototype ===")
    log(f"scikit-learn / numpy / matplotlib pipeline, seed={RANDOM_STATE}")

    # 1. data (synthetic placeholder; replace with real CAD dataset later)
    clouds, labels = make_dataset(n_per_class=120, n_points=2048,
                                  seed=RANDOM_STATE)
    log(f"\n[1] dataset: {clouds.shape[0]} samples x {clouds.shape[1]} points, "
        f"categories = {CATEGORIES}")

    # 2. feature engineering
    X = np.array([extract_features(pc) for pc in clouds])
    log(f"[2] feature matrix: {X.shape}, features = {len(FEATURE_NAMES)}")
    y = labels

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    log(f"    train={len(Xtr)}  test={len(Xte)} (stratified split)")

    # 3. baselines
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", C=10, random_state=RANDOM_STATE),
        "k-NN": KNeighborsClassifier(n_neighbors=7),
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800,
                             random_state=RANDOM_STATE),
    }

    results = {}
    log("\n[3] baseline comparison on held-out test set:")
    for name, clf in models.items():
        clf.fit(Xtr_s, ytr)
        pred = clf.predict(Xte_s)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="weighted")
        results[name] = (acc, f1, pred)
        log(f"    {name:14s}  accuracy={acc:.4f}  weighted-F1={f1:.4f}")

    best_name = max(results, key=lambda k: results[k][0])
    best_acc, best_f1, best_pred = results[best_name]
    log(f"\n    best model: {best_name} (acc={best_acc:.4f}, F1={best_f1:.4f})")

    log("\n[4] classification report (best model):")
    log(classification_report(yte, best_pred, target_names=CATEGORIES, digits=3))

    # 4. confusion matrix figure
    cm = confusion_matrix(yte, best_pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
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
    log(f"    saved {cm_path}")

    # 5. feature importance from RandomForest
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
    log(f"    saved {fi_path}")

    log(f"\n[done] elapsed {time.time() - t0:.1f}s")

    with open(os.path.join(HERE, "train_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    return 0


if __name__ == "__main__":
    sys.exit(main())
