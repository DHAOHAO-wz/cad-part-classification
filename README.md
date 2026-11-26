# 基于几何特征工程与轻量机器学习的 CAD 零件分类

Python 课程期末项目：输入采样点云，提取可解释的手工几何特征，用轻量机器学习模型完成四类 CAD 零件分类，并完成四基线对比实验、单元测试与量化结果分析。

## 1. 选题与方法

- **任务**：对 `block`（方块）、`bracket_L`（L 形支架）、`cylinder`（圆柱）、`plate_holes`（带孔板件）四类零件进行点云分类。
- **技术路线**：点云 → 17 维手工几何特征（包络尺寸/比例、质心距离分布、PCA 各向异性、局部平面度等）→ `StandardScaler` 标准化 → 四个轻量基线对比（Random Forest / SVM-RBF / k-NN / MLP）。
- **选择依据**：样本量有限、类别形态可由几何量刻画；手工特征 + 轻量模型数据需求小、CPU 秒级可训练、结果可解释，避免直接处理原始点云的深度学习方法对样本量与 GPU 的高要求。

## 2. 环境与依赖

- **Python 版本**：3.12.11（3.10+ 均可运行）
- **第三方依赖及版本**：见 [`requirements.txt`](requirements.txt)

| 依赖 | 版本 | 用途 |
| --- | --- | --- |
| numpy | 1.26.4 | 数值计算、点云与特征矩阵 |
| scikit-learn | 1.4.0 | 分类基线、划分、标准化、评估指标 |
| matplotlib | 3.11.0 | 混淆矩阵、特征重要性可视化 |
| pytest | 9.2.0 | 单元测试 |

## 3. 数据集说明

| 项目 | 内容 |
| --- | --- |
| 数据集名称 | 内置合成点云集 `synthetic-cad-parts`（占位数据） |
| 数据集版本 | v1（由脚本生成，无外部下载链接） |
| 生成脚本 | `data.py` 中的 `make_dataset()` |
| 样本规模 | 480 个样本，每类 120 个；每个点云 2048 个采样点 |
| 类别 | `block` / `bracket_L` / `cylinder` / `plate_holes` |
| 随机种子 | `RANDOM_STATE = 42`（可复现） |

### 数据获取与预处理步骤

1. **无需外部下载**：合成点云由 `data.py` 在运行时按固定种子生成，含少量高斯抖动模拟真实采样噪声。
2. **预处理**：点云进入特征提取前做中心化与尺度归一；特征矩阵在训练集上拟合 `StandardScaler`，再作用于测试集。
3. **真实数据接入（预留）**：后续可替换为真实 CAD 网格/点云数据集（如 Fusion 360 Gallery、ABC 子集）。接入骨架见 `skeleton/`（`feature_extraction.py`、`classification.py`、`README_skeleton.md`），接入时只需替换 `data.py` 的数据生成部分，特征与训练流程保持不变。

> 说明：当前合成数据类别几何差异被刻意放大，测试集高分仅为流程验证结果，**不代表真实 CAD 零件上的泛化性能**。

## 4. 主要超参数

| 模型 / 环节 | 超参数 |
| --- | --- |
| 数据划分 | `TEST_SIZE = 0.25`，分层划分（stratified） |
| Random Forest | `n_estimators=200`，`random_state=42` |
| SVM (RBF) | `kernel='rbf'`，`C=10`，`random_state=42` |
| k-NN | `n_neighbors=7` |
| MLP | `hidden_layer_sizes=(32, 16)`，`max_iter=800`，`random_state=42` |

## 5. 从安装到运行的完整命令序列

```bash
# 1. 进入项目目录
cd project

# 2. 安装全部第三方依赖（版本见 requirements.txt）
pip install -r requirements.txt

# 3. 运行训练与四基线对比实验（生成日志、指标表与结果图）
python train.py

# 4. 运行单元测试（pytest）
pytest -q
```

## 6. 目录结构

```
project/
├── README.md                  # 本说明文档
├── requirements.txt           # 第三方依赖及锁定版本
├── analysis.md                # 实验设置、量化指标、特征重要性与结果分析
├── data.py                    # 合成点云数据集生成（真实数据接入替换点）
├── features.py                # 17 维手工几何特征提取
├── train.py                   # 标准化、四基线对比训练、评估与可视化
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py       # pytest 单元测试（数据/特征/标准化/模型预测）
├── results/
│   └── metrics.csv            # 四基线 accuracy / weighted-F1 / 耗时汇总
├── assets/                    # 结果图与可验证截图（运行生成）
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── terminal_run.png
├── train_log.txt              # 完整运行日志
├── terminal_output.txt        # 终端原始输出文本
└── skeleton/                  # 真实 CAD 数据接入预留骨架
    ├── README_skeleton.md
    ├── feature_extraction.py
    └── classification.py
```

## 7. 测试与可复现说明

- **单元测试**：`tests/test_pipeline.py`，共 5 个用例，覆盖数据集形状与类别均衡、同种子可复现、特征向量长度与数值有限、标准化后训练集近似零均值、四模型预测类别合法。运行：`pytest -q`。
- **可复现性**：固定 `RANDOM_STATE = 42`；数据生成、划分、模型训练均使用该种子。
- **实验产物**：
  - `results/metrics.csv`：四基线 Accuracy / Weighted-F1 / 训练+预测耗时；
  - `train_log.txt`：终端完整日志与分类报告；
  - `assets/confusion_matrix.png`：最佳模型混淆矩阵；
  - `assets/feature_importance.png`：随机森林特征重要性；
  - 详细分析与讨论见 [`analysis.md`](analysis.md)。

## 8. 代码提交记录

- 远程仓库：https://github.com/DHAOHAO-wz/cad-part-classification （分支 `master`）
- 可验证材料（运行截图、日志、Git commit 记录）见 `VERIFICATION.md`。
