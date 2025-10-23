# 配方：解耦裁剪和动态采样策略优化（DAPO）

> 开源算法实现与实验运行：[Yuxuan Tong](https://tongyx361.github.io/), [Guangming Sheng](https://hk.linkedin.com/in/guangming-sheng-b50640211)

> [!IMPORTANT]
>
> **🔥 新闻！！！**
>
> - [2025/04] 我们复现了两个版本DAPO的结果（[完整版本](./run_dapo_qwen2.5_32b.sh) & [无动态采样版本](./run_dapo_wo_ds_qwen2.5_32b.sh)），基于[recipe/dapo上的最新代码库](https://github.com/volcengine/verl/tree/recipe/dapo/recipe/dapo)，在AIME 2024上分别达到52%和50%的准确率。请查看[W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n)了解详细信息。
> - [2025/03] 我们发布了[早期版本DAPO（无Token级PG损失和动态采样）](./run_dapo_early_qwen2.5_32b.sh)的训练记录，在AIME 2024上达到44%准确率，详见[W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n)。

🏠 [主页](https://dapo-sia.github.io/) | 📝 [论文@arXiv](https://arxiv.org/abs/2503.14476) | 🤗 [数据集和模型@HF](https://huggingface.co/collections/BytedTsinghua-SIA/dapo-67d7f1517ee33c8aed059da0) | 🐱 [代码@GitHub](https://github.com/volcengine/verl/tree/recipe/dapo/recipe/dapo) | 🐱 [仓库@GitHub](https://github.com/BytedTsinghua-SIA/DAPO)

> 我们提出了**D**ecoupled Clip和Dynamic s**A**mpling **P**olicy **O**ptimization（DAPO）算法。通过公开我们的工作，我们为更广泛的研究界和社会提供了可扩展强化学习的实际访问权限，使所有人都能从这些进步中受益。我们的系统基于优秀的[verl](https://github.com/volcengine/verl)框架。感谢他们的伟大工作！将DAPO训练应用于Qwen2.5-32B基础模型证明在AIME 2024上优于之前的SOTA模型DeepSeek-R1-Zero-Qwen-32B，在**50%**的训练步数下实现**50%**的准确率。
>
> ![dapo-main-result](https://dapo-sia.github.io/static/images/score.png)

## 快速开始

1. 在Ray集群上准备数据集：

```bash
bash prepare_dapo_data.sh # 默认下载数据集到${HOME}/verl/data
```

2. 从任何机器提交作业到Ray集群：

```bash
cd verl # 仓库根目录
export RAY_ADDRESS="http://${RAY_IP:-localhost}:8265" # 要连接的Ray集群地址
export WORKING_DIR="${PWD}" # 打包到Ray集群的本地目录
# 在yaml中设置Ray集群的运行时环境，如环境变量和pip包
export RUNTIME_ENV="./recipe/dapo/runtime_env.yaml" # 这为Ray集群设置环境变量
bash recipe/dapo/run_dapo_qwen2.5_32b.sh # 或其他脚本
```

## 复现运行

| 设置                                        | AIME 2024 准确率 | 硬件         | 镜像                                                                 | 提交                                                                                             | 环境变量                                                                                                                     | 训练脚本                                                                                                                                           | 训练记录                                                                                |
| -------------------------------------------- | -------------- | ----------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| DAPO                                         | 52%            | 16x8xH800   | `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0` | [`4f80e4`](https://github.com/volcengine/verl/tree/4f80e465c2ec79ab9c3c30ec74b9745de61d0490) | [runtime_env.yaml](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/runtime_env.yaml) | [run_dapo_qwen2.5_32b.sh](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/run_dapo_qwen2.5_32b.sh)             | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n) |
| DAPO w/o Dynamic Sampling                    | 50%            | 16x8xH800   | `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0` | [`4f80e4`](https://github.com/volcengine/verl/tree/4f80e465c2ec79ab9c3c30ec74b9745de61d0490) | [runtime_env.yaml](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/runtime_env.yaml) | [run_dapo_wo_ds_qwen2.5_32b.sh](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/run_dapo_wo_ds_qwen2.5_32b.sh) | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n) |
| DAPO w/o Token-level Loss & Dynamic Sampling | 44%            | 16x8xH20    | `hiyouga/verl:ngc-th2.5.1-cu120-vllm0.7.4-hotfix`                    | [`4f80e4`](https://github.com/volcengine/verl/tree/4f80e465c2ec79ab9c3c30ec74b9745de61d0490) | [runtime_env.yaml](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/runtime_env.yaml) | [run_dapo_early_qwen2.5_32b.sh](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/run_dapo_early_qwen2.5_32b.sh) | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n) |

> [!IMPORTANT]
>
> **📢 征集贡献！**
>
> 欢迎提交您的复现运行和设置！

## 配置

### 分离的裁剪Epsilon（-> 更高裁剪）

示例配置：

```yaml
actor_rollout_ref:
  actor:
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
```

`clip_ratio_low`和`clip_ratio_high`指定DAPO目标中的$\varepsilon_{\text {low }}$和$\varepsilon_{\text {high }}$。

核心相关代码：

```python
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
pg_losses = torch.maximum(pg_losses1, pg_losses2)
```

### 动态采样（带组过滤）

示例配置：

```yaml
data:
  gen_batch_size: 1536
  train_batch_size: 512
algorithm:
  filter_groups:
    enable: True
    metric: acc # score / seq_reward / seq_final_reward / ...
    max_num_gen_batches: 10 # 非正值表示无上限
```

将`filter_groups.enable`设置为`True`将过滤掉输出的`metric`都相同的组，例如对于`acc`，输出准确率都为1或0的组。

训练器将重复以`gen_batch_size`采样，直到有足够的合格组用于`train_batch_size`或达到`max_num_gen_batches`指定的上限。

核心相关代码：

```python
prompt_bsz = self.config.data.train_batch_size
if num_prompt_in_batch < prompt_bsz:
    print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
    num_gen_batches += 1
    max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
    if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
        print(f'{num_gen_batches=} < {max_num_gen_batches=}. Keep generating...')
        continue
    else:
        raise ValueError(
            f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
        )
else:
    # 对齐批次
    traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
    batch = batch[:traj_bsz]
```

### 灵活损失聚合模式（-> Token级损失）

示例配置：

```yaml
actor_rollout_ref:
  actor:
    loss_agg_mode: "token-mean" # / "seq-mean-token-sum" / "seq-mean-token-mean"
    # 注意："token-mean"是默认行为
```

将`loss_agg_mode`设置为"token-mean"将在小批次中所有序列的所有token上取平均（策略梯度）损失。

核心相关代码：

```python
if loss_agg_mode == "token-mean":
    loss = verl_F.masked_mean(loss_mat, loss_mask)
elif loss_agg_mode == "seq-mean-token-sum":
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
    loss = torch.mean(seq_losses)  # seq-mean
elif loss_agg_mode == "seq-mean-token-mean":
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
    loss = torch.mean(seq_losses)  # seq-mean
else:
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
```

### 过长奖励塑形

示例配置：

```yaml
data:
  max_response_length: 20480 # 16384 + 4096
reward_model:
  overlong_buffer:
    enable: True
    len: 4096
    penalty_factor: 1.0
```

将`overlong_buffer.enable`设置为`True`将惩罚长度过长但仍在硬上下文限制内的输出。

具体来说，当输出长度超过`max_response_length`从0到`overlong_buffer.len`个token时，惩罚从`0`线性增加到`overlong_buffer.penalty_factor`。

核心相关代码：

```python
if self.overlong_buffer_cfg.enable:
    overlong_buffer_len = self.overlong_buffer_cfg.len
    expected_len = self.max_resp_len - overlong_buffer_len
    exceed_len = valid_response_length - expected_len
    overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
    reward += overlong_reward
```

## 常见问题

### 论文中的"过长过滤"在哪里？

论文中的大多数实验，包括表现最好的实验，都是在没有过长过滤的情况下运行的，因为它在从最长输出中适当学习方面与过长奖励塑形有些重叠。所以我们这里没有实现它。

### [主分支中的`recipe/dapo`目录](https://github.com/volcengine/verl/tree/main/recipe/dapo)和[`recipe/dapo`分支](https://github.com/volcengine/verl/tree/recipe/dapo/recipe/dapo)有什么区别？

[`recipe/dapo`分支](https://github.com/volcengine/verl/tree/recipe/dapo/recipe/dapo)用于**原样复现**，因此不会随新功能更新。

[主分支中的`recipe/dapo`目录](https://github.com/volcengine/verl/tree/main/recipe/dapo)作为如何扩展最新`verl`以实现算法配方的示例，将随新功能维护。

### 为什么修改后我无法产生相似的结果？

当今的RL基础设施仍然具有固有的不稳健性，我们正在努力改进。

我们强烈建议一次只修改一件事。

我们还列出了一些已知问题：

1. 启用CUDA图（`enforce_eager=False`）可能会导致模型性能下降，其原因仍在调查中。