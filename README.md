# CAD 零件分类原型（Python 与 CAD 关联 · 课程中期项目）

## 选题
基于几何特征工程与轻量机器学习的 CAD 零件分类：输入三角网格 / 点云，
提取可解释的几何特征，用轻量分类模型输出零件类别（支架、法兰、垫块、带孔板件等）。

## 当前阶段（中期）
- 已跑通「点云 → 手工几何特征 → 轻量分类基线」全流程原型。
- 数据为**合成点云占位**，仅用于验证流程可端到端运行；真实 CAD 数据集
  （Fusion 360 Gallery / ABC 子集）接入后替换 `data.py` 即可，特征与训练代码不变。
- 合成数据类别极易区分，当前准确率为流程验证结果，**不代表真实数据性能**。

## 环境依赖
- Python 3.10+
- numpy、scikit-learn、matplotlib（已在课程环境验证：numpy 1.26.4 / sklearn 1.4.0 / matplotlib 3.11.0）
- 后续接入真实网格时再按需安装 trimesh / Open3D / PythonOCC。

## 运行方式
```bash
python3 train.py
```

## 输出
- `train_log.txt`：运行日志（数据规模、特征维度、各模型准确率与加权 F1、分类报告）。
- `assets/confusion_matrix.png`：最优模型混淆矩阵。
- `assets/feature_importance.png`：随机森林特征重要性。

## 代码结构
- `data.py`：点云数据生成（占位；后续替换为真实数据集加载）。
- `features.py`：可解释几何特征提取（包围盒比例、PCA 各向异性、局部平面度等）。
- `train.py`：特征标准化、训练 / 测试划分、基线对比（随机森林 / SVM / k-NN / MLP）与可视化。
- `skeleton/`：面向真实数据接入的配套代码骨架（`feature_extraction.py` 几何特征清单、
  `classification.py` 基线与评估流程、`README_skeleton.md` 接入说明）；标 `TODO` 处
  接入真实 CAD 数据集后补全，与本目录可运行原型区分。

## 演示与可验证材料
- 现场演示：运行 `python3 train.py`，展示日志输出与两张图。
- 可验证材料：本目录运行日志 `train_log.txt`、上述两张结果图，以及 Git commit 记录。
