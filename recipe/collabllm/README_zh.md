# CollabLLM

这个仓库使用verl框架实现了[CollabLLM](https://arxiv.org/pdf/2502.00640) (ICML 2025)。原始实现请参见[CollabLLM仓库](https://github.com/Wuyxin/collabllm)。

CollabLLM是一种训练语言模型在多轮对话中有效协作的方法。这个实现将原始实现适配到Verl训练框架中工作。

## 快速开始

### 0. 环境
确保已安装`verl`所需的包。此外，安装`litellm`并导出所需的API密钥。API模型将用于用户模拟器，以及可选的LLM评判器（见下面的配置部分）。

### 1. 准备数据集

首先使用提供的脚本处理您的数据集：

```bash
python process_dataset.py --dataset <> ... --dataset_type <sft or rl>
```

**要求：**
- 输入：一个Hugging Face多轮对话数据集。现有数据集：`collabllm/collabllm-multiturn-$DATASET`，其中`DATASET`为[`math-hard(-large)`, `medium(-large)`, `bigcodebench(-large)`]中的一个（*-large是CollabLLM论文中使用的数据集）
- 示例格式：参见[collabllm-multiturn-math-hard](https://huggingface.co/datasets/collabllm/collabllm-multiturn-math-hard)
- 生成自己的数据集：使用原始CollabLLM仓库中的[build_dataset.py](https://github.com/Wuyxin/collabllm/blob/main/scripts/engine/build_dataset.py)

*注意：查看`process_dataset.py`了解示例命令和用法。*

### 2. 训练模型

**（可选）监督微调（SFT）：**
```bash
bash train_sft_collabllm.sh
```

**强化学习（RL）：**

```bash
bash train_rl_collabllm.sh
```

RL脚本展示了在`math-hard-large`上训练CollabLLM的示例。

- 采样未来对话的配置在`recipe/collabllm/config/collabllm_interaction_config.yaml`中。
- 多轮感知奖励（Multiturn-aware Reward）从这三个对话级奖励聚合而来：

    ```
    +reward_model.reward_kwargs.metric_weights.accuracy=1 \
    +reward_model.reward_kwargs.metric_weights.interactivity=1 \
    +reward_model.reward_kwargs.metric_weights.token_amount=-0.0001 \
    ```

    您可以根据任务移除、添加或修改权重。已实现指标的列表在`recipe/collabllm/metrics`下。例如，在`medium-large`上，您可以通过以下方式用`bleu_score`替换`accuracy`：
    ```
    +reward_model.reward_kwargs.metric_weights.bleu_score=1
    ```
    这将对采样的未来对话应用bleu分数。

## 配置
详细配置请阅读[文档](https://verl.readthedocs.io/en/latest/)。

## 引用
如果CollabLLM对您的研究有用，请引用以下内容：

```bibtex
@inproceedings{collabllm2025,
    title={CollabLLM: From Passive Responders to Active Collaborators},
    author={Shirley Wu and Michel Galley and Baolin Peng and Hao Cheng and
            Gavin Li and Yao Dou and Weixin Cai and James Zou and
            Jure Leskovec and Jianfeng Gao},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2025}
}
```