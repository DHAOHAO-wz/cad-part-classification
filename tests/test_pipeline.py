"""Unit tests for the CAD part classification pipeline.

测试对象：合成点云数据集生成、几何特征提取、标准化与四个轻量基线。
输入：固定随机种子的小规模合成点云（20 样本/类，256 点/样本）。
预期结果：数据形状与类别均衡符合约定；特征向量长度固定且数值有限；
标准化后训练集近似零均值；各模型预测类别均在合法类别范围内。

运行：在项目根目录执行  pytest -q
"""

import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 允许从 tests/ 子目录导入上级项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import make_dataset, CATEGORIES          # noqa: E402
from features import extract_features, FEATURE_NAMES  # noqa: E402
from train import build_models, RANDOM_STATE       # noqa: E402

# 小规模配置，保证测试快速运行
N_PER_CLASS = 20
N_POINTS = 256
N_CLASSES = len(CATEGORIES)


def _make_small_dataset():
    return make_dataset(n_per_class=N_PER_CLASS, n_points=N_POINTS,
                         seed=RANDOM_STATE)


def test_dataset_shape_and_balance():
    """数据集形状为 (样本数, 点数, 3)，四类样本数均衡。"""
    clouds, labels = _make_small_dataset()
    assert clouds.shape == (N_PER_CLASS * N_CLASSES, N_POINTS, 3)
    assert labels.shape == (N_PER_CLASS * N_CLASSES,)
    # 每类样本数相等
    counts = np.bincount(labels)
    assert (counts == N_PER_CLASS).all()
    # 类别覆盖全部约定类别
    assert set(np.unique(labels)) == set(range(N_CLASSES))


def test_dataset_reproducible():
    """同一随机种子两次生成的数据应完全一致（可复现）。"""
    c1, y1 = _make_small_dataset()
    c2, y2 = _make_small_dataset()
    assert np.array_equal(c1, c2)
    assert np.array_equal(y1, y2)


def test_feature_vector_length_and_finite():
    """特征向量长度与 FEATURE_NAMES 一致，且不含 NaN / Inf。"""
    clouds, _ = _make_small_dataset()
    feat = extract_features(clouds[0])
    assert feat.shape == (len(FEATURE_NAMES),)
    assert len(FEATURE_NAMES) == 17
    assert np.isfinite(feat).all()


def test_standardized_train_zero_mean():
    """标准化后训练集各特征近似零均值。"""
    clouds, labels = _make_small_dataset()
    X = np.array([extract_features(pc) for pc in clouds])
    Xtr, _, ytr, _ = train_test_split(
        X, labels, test_size=0.25, random_state=RANDOM_STATE, stratify=labels)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    assert np.allclose(Xtr_s.mean(axis=0), 0.0, atol=1e-6)


def test_models_predict_valid_labels():
    """四个基线均可训练，且预测类别落在合法类别范围内。"""
    clouds, labels = _make_small_dataset()
    X = np.array([extract_features(pc) for pc in clouds])
    Xtr, Xte, ytr, yte = train_test_split(
        X, labels, test_size=0.25, random_state=RANDOM_STATE, stratify=labels)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    for name, model in build_models().items():
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)
        assert set(np.unique(pred)).issubset(set(range(N_CLASSES))), name
        assert len(pred) == len(yte)
