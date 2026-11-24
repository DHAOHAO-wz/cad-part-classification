# 可验证材料清单（中期）

## 运行环境
- 操作系统：Linux（课程云主机）
- Python：numpy 1.26.4 / scikit-learn 1.4.0 / matplotlib 3.11.0
- 运行命令：`python3 train.py`
- 运行耗时：约 13.1 s（480 样本 × 2048 点，17 维特征）

## 代码提交记录
- 远程仓库：https://github.com/DHAOHAO-wz/cad-part-classification （分支 `master`）
- Commit：[`54d63e2`](https://github.com/DHAOHAO-wz/cad-part-classification/commit/54d63e239b36aa980999ec2e3a13fbd68babbfb8) —— 中期原型: 点云几何特征 + sklearn 轻量分类基线跑通
- Commit：[`0fc244e`](https://github.com/DHAOHAO-wz/cad-part-classification/commit/0fc244e70b48dc260fde82844ea8d205ee8004d6) —— 补充 README 与可验证材料清单
- Commit：[`8f7bb40`](https://github.com/DHAOHAO-wz/cad-part-classification/commit/8f7bb40a28d6987f4a8830e0672c72a9f7ba9449) —— 补充终端运行截图与输出文本，更新验证清单
- Commit：[`c8f0fc6`](https://github.com/DHAOHAO-wz/cad-part-classification/commit/c8f0fc67612a6c8714ff514b1e7cdaeea844f895) —— 回填最新 commit 哈希到验证清单
- Commit：[`e138f8b`](https://github.com/DHAOHAO-wz/cad-part-classification/commit/e138f8ba5ffc6f407de6ceaf2ce53b8363fbdce6) —— 纳入面向真实数据的特征提取/分类骨架，刷新运行日志与结果图
- Commit：[`a98ff48`](https://github.com/DHAOHAO-wz/cad-part-classification/commit/a98ff483972909de22af130a7efb220be87f2cd6) —— 回填骨架提交的 commit 哈希
- Commit：[`74a9cbd`](https://github.com/DHAOHAO-wz/cad-part-classification/commit/74a9cbda4c4fd61608d8f01b5f8eb5819e32aa2c) —— 更新远程 GitHub 仓库地址与 commit 链接

## 运行产物
- `train_log.txt`：完整运行日志（数据规模、特征维度、各模型准确率 / 加权 F1、分类报告）
- `terminal_output.txt`：终端原始输出文本（含运行命令）
- `assets/terminal_run.png`：终端运行过程截图（命令行执行 `python3 train.py` 的完整输出）
- `assets/confusion_matrix.png`：最优模型（随机森林）混淆矩阵
- `assets/feature_importance.png`：随机森林特征重要性排序
- `skeleton/`：面向真实数据接入的代码骨架（特征提取、分类评估流程与接入说明）

## 说明
当前为合成点云占位原型，用于验证「特征提取 + 轻量分类」流程可端到端跑通；
合成数据类别易分，当前指标不代表真实 CAD 数据集性能，真实数据结果待后续阶段补充。
