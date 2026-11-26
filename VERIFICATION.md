# 可验证材料清单（期末）

## 运行环境

- 操作系统：Linux（课程云主机）；Windows 本机亦可复现
- Python：3.12.11
- 第三方依赖：numpy 1.26.4 / scikit-learn 1.4.0 / matplotlib 3.11.0 / pytest 9.2.0（见 `requirements.txt`）
- 训练命令：`python train.py`
- 测试命令：`pytest -q`（5 个用例全部通过）
- 训练耗时：约 13.3 s（480 样本 × 2048 点，17 维特征）

## 代码提交记录

- 远程仓库：https://github.com/DHAOHAO-wz/cad-part-classification （分支 `master`）
- 期末提交 HEAD：[`d004ff1`](https://github.com/DHAOHAO-wz/cad-part-classification/commit/d004ff1faf0eda09f1358cb692e9f7366258841b) —— 期末交付：pytest 测试、metrics.csv 指标汇总、analysis 分析与规范 README/requirements

## 运行产物（均可由命令复现）

| 材料 | 路径 | 说明 |
| --- | --- | --- |
| 运行日志 | `train_log.txt` | 数据规模、特征维度、四基线 Accuracy / Weighted-F1、分类报告 |
| 终端原始输出 | `terminal_output.txt` | 命令行执行 `python train.py` 的完整文本输出 |
| 终端运行截图 | `assets/terminal_run.png` | 上述终端输出的终端风格截图 |
| 混淆矩阵 | `assets/confusion_matrix.png` | 最佳模型（Random Forest）混淆矩阵 |
| 特征重要性 | `assets/feature_importance.png` | 随机森林 17 维特征重要性排序 |
| 指标汇总表 | `results/metrics.csv` | 四基线 Accuracy / Weighted-F1 / 训练+预测耗时 |
| 单元测试 | `tests/test_pipeline.py` | 5 个 pytest 用例：数据形状与均衡、可复现、特征有限、标准化零均值、预测类别合法 |
| 结果分析 | `analysis.md` | 实验设置、量化指标、特征重要性观察、对比讨论与局限 |

## 复现步骤

```bash
pip install -r requirements.txt
python train.py     # 生成 train_log.txt、results/metrics.csv、assets/*.png
pytest -q           # 运行单元测试
```

## 说明

当前为合成点云占位数据，用于验证「特征提取 + 轻量分类」流程端到端可运行、测试可通过；合成数据类别易分，当前 100% 指标不代表真实 CAD 数据集性能。真实 CAD 数据接入预留骨架见 `skeleton/`。
