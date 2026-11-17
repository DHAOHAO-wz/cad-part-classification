# 可验证材料清单（中期）

## 运行环境
- 操作系统：Linux（课程云主机）
- Python：numpy 1.26.4 / scikit-learn 1.4.0 / matplotlib 3.11.0
- 运行命令：`python3 train.py`
- 运行耗时：约 14.9 s（480 样本 × 2048 点，17 维特征）

## 代码提交记录
- 仓库：本地 Git 仓库 `cad_classifier/`
- Commit：`dd913563cb66957f0f42d5b1e85ffe0f4876dbad`
- 提交信息：中期原型: 点云几何特征 + sklearn 轻量分类基线跑通

## 运行产物
- `train_log.txt`：完整运行日志（数据规模、特征维度、各模型准确率 / 加权 F1、分类报告）
- `assets/confusion_matrix.png`：最优模型（随机森林）混淆矩阵
- `assets/feature_importance.png`：随机森林特征重要性排序

## 说明
当前为合成点云占位原型，用于验证「特征提取 + 轻量分类」流程可端到端跑通；
合成数据类别易分，当前指标不代表真实 CAD 数据集性能，真实数据结果待后续阶段补充。
