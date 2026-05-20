# 超声图像脂肪肝二分类实验

本仓库为医学影像深度学习课程大作业项目，任务是基于 B-mode 肝脏超声图像完成正常肝脏与脂肪肝的二分类。

## 项目地址

GitHub 仓库：

<https://github.com/pangchip14/fatty-liver-ultrasound-classification>

本仓库公开保存代码、非敏感实验结果、loss 曲线、模型拓扑图和报告文件。原始数据、私有测试集、模型权重和包含私有测试图像的 Grad-CAM 可视化图未上传。

## 实验内容

- 任务：正常肝脏 vs 脂肪肝二分类
- 训练/验证数据：`Fatty-Liver-public`，按 8:2 分层随机划分
- 独立测试数据：`Fatty-Liver-private-test`
- 模型：
  - ResNet18
  - MobileNetV3-Small
  - EfficientNet-B0
- 评价指标：
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - 双边配对 t 检验
- 可解释性方法：Grad-CAM
- 类别不平衡处理：加权交叉熵

## 重要隐私说明

`Fatty-Liver-private-test` 是老师提供的私有测试集，不具备公开授权。因此本仓库不包含：

- 原始数据集压缩包
- 解压后的训练、验证、测试图像
- 模型 checkpoint
- 含私有测试图像内容的 Grad-CAM 图片
- 结果打包压缩文件

仓库中的 `summary/` 仅保留可公开的汇总指标、loss 曲线和网络拓扑图。

## 主要文件

| 文件 | 说明 |
|---|---|
| `train.py` | 模型训练脚本 |
| `evaluate.py` | 独立测试集评估脚本 |
| `gradcam.py` | Grad-CAM 可视化脚本 |
| `summarize_results.py` | 汇总测试指标与 t 检验 |
| `export_report_assets.py` | 导出报告图表和实验说明 |
| `scripts/make_splits.py` | 解压数据并生成训练/验证/测试 CSV |
| `src/` | 数据集、模型、损失函数和指标代码 |
| `summary/` | 非敏感结果表、loss 曲线和模型拓扑图 |
| `彭浩楠231240027-期中作业.pdf` | 最终 PDF 版实验报告 |

## 复现实验

将数据集压缩包放到以下位置：

```text
data/ultrasoud-fatty-Liver-classification-data.zip
```

生成数据划分：

```bash
python scripts/make_splits.py \
  --zip data/ultrasoud-fatty-Liver-classification-data.zip \
  --out-dir data \
  --seed 42 \
  --val-ratio 0.2
```

运行全部实验：

```bash
python run_experiments.py \
  --models resnet18 mobilenet_v3_small efficientnet_b0 \
  --seeds 42 43 44 \
  --epochs 30 \
  --batch-size 32 \
  --loss weighted_ce
```

评估单个 checkpoint：

```bash
python evaluate.py \
  --checkpoint outputs/<run_name>/checkpoints/best.pt \
  --output-dir outputs/<run_name>/test_eval
```

汇总结果：

```bash
python summarize_results.py \
  --outputs-dir outputs \
  --summary-dir summary
```

导出报告素材：

```bash
python export_report_assets.py \
  --outputs-dir outputs \
  --summary-dir summary
```

## 当前结果摘要

独立测试集上，3 个随机种子结果的均值和标准差如下：

| 模型 | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| MobileNetV3-Small | 0.5534 ± 0.0846 | 0.7721 ± 0.0311 | 0.4343 ± 0.2012 | 0.5377 ± 0.1490 |
| ResNet18 | 0.4854 ± 0.0700 | 0.8155 ± 0.0247 | 0.2525 ± 0.1306 | 0.3743 ± 0.1523 |
| EfficientNet-B0 | 0.4628 ± 0.0297 | 0.8009 ± 0.0289 | 0.2172 ± 0.0683 | 0.3373 ± 0.0830 |

按平均 F1-score，MobileNetV3-Small 为本实验最优模型。

## 环境

- Python 3.12
- PyTorch 2.8.0+cu128
- torchvision 0.23.0+cu128
- GPU：NVIDIA GeForce RTX 4090
- 主要依赖：见 `requirements.txt`

