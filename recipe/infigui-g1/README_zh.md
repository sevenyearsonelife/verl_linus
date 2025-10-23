# InfiGUI-G1 配方

这个目录包含论文[InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization](https://arxiv.org/abs/2508.05731)的官方实现。

这项工作引入了自适应探索策略优化（AEPO），这是一个旨在增强多模态大型语言模型（MLLMs）中GUI定位的策略优化框架。AEPO通过采用多答案生成策略和理论上基于的自适应探索奖励（AER）函数来提高探索效率。这种方法有效解决了复杂GUI定位任务中的语义对齐挑战。

我们为3B和7B模型提供训练脚本，默认配置为单机8 GPU。

## 环境设置

请遵循`verl`的主要环境设置指南。

提供的脚本使用以下Docker镜像：`verlai/verl:app-verl0.5-transformers4.55.4-sglang0.4.10.post2-mcore0.13.0-te2.2`

## 数据准备

在开始训练之前，您需要下载示例数据集。这个数据集是[omniact](https://huggingface.co/datasets/Writer/omniact)的过滤版本，仅包含定位任务，排除了简单样本。

数据托管在Hugging Face上。您可以使用`huggingface-cli`下载它：

```bash
huggingface-cli download --repo-type dataset --resume-download InfiX-ai/omniact_grounding_filtered --local-dir data/omniact_grounding_filtered
```

此命令将训练和验证parquet文件下载到`data/omniact_grounding_filtered`目录，这是脚本使用的默认路径。

## 训练

我们提供训练3B和7B模型的脚本。请从`verl`的根目录运行它们。

-   **训练3B模型：**

    ```bash
    bash recipe/infigui-g1/run_3b.sh
    ```

-   **训练7B模型：**

    ```bash
    bash recipe/infigui-g1/run_7b.sh
    ```

## 使用自定义数据

如果您希望在自己的数据集上训练，请格式化您的数据以匹配位于`data/omniact_grounding_filtered`的示例文件结构。

数据准备好后，您需要更新训练脚本中的数据路径参数。

在`run_3b.sh`或`run_7b.sh`中，修改以下行：

```bash
    data.train_files=./path/to/your/train_data.parquet \
    data.val_files=./path/to/your/val_data.parquet \
```

将路径替换为自定义数据文件的位置。