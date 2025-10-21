# 近端策略优化 (PPO)

最后更新：2025年06月19日。

近端策略优化 (PPO) 是 OpenAI 于 2017 年提出的用于强化学习的策略梯度方法家族。PPO 在简单性、稳定性和性能之间取得了平衡，使其成为现代 RL 应用中最广泛使用的算法之一，包括大规模语言模型微调。

传统的策略梯度方法如 REINFORCE 或 Vanilla Policy Gradient 存在以下问题：

- 高方差和样本效率低下
- 由于策略更新过大导致的不稳定性

PPO 通过使用裁剪的代理目标来解决这个问题，避免了过大的更新，而不需要二阶导数。

关于 PPO 的更多技术细节，我们建议阅读 [OpenAI spinning up 教程](https://spinningup.openai.com/en/latest/algorithms/ppo.html) 中的介绍，以及论文 [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)。

## 核心组件

- **Actor-Critic 架构**：PPO 需要 actor 模型（策略）和 critic 模型（价值函数）。这与其他不需要 critic 模型的算法如 GRPO 和 RLOO 不同。

- **广义优势估计 (GAE)**：PPO 使用 GAE 计算优势值，有助于减少策略梯度估计的方差，同时保持低偏差。

- **裁剪代理目标**：PPO 的核心通过限制策略更新的裁剪代理目标函数实现。

## 配置

注意，所有包含 `micro_batch_size` 的配置都用于配置每次前向或反向传递的最大样本或令牌数量，以避免 GPU OOM，其值不应改变算法/收敛行为。

大多数 critic 配置与 actor 的配置相似。请注意，下图中省略了 critic 模型。

![image](https://github.com/user-attachments/assets/16aebad1-0da6-4eb3-806d-54a74e712c2d)

- `data.train_batch_size`：用于生成一组采样轨迹/推出的提示的全局批次大小。响应/轨迹的数量为 `data.train_batch_size * actor_rollout.ref.rollout.n`

- `actor_rollout_ref.actor.ppo_mini_batch_size`：采样的轨迹集被分割成多个小批次，batch_size=ppo_mini_batch_size，用于 PPO actor 更新。ppo_mini_batch_size 是所有工作器的全局大小

- `critic.ppo_mini_batch_size`：采样的轨迹集被分割成多个小批次，batch_size=ppo_mini_batch_size，用于 PPO critic 更新。ppo_mini_batch_size 是所有工作器的全局大小

- `actor_rollout_ref.actor.clip_ratio`：PPO 裁剪范围。默认为 0.2

- `actor_rollout_ref.actor.ppo_epochs`：在一组采样轨迹上对 actor 进行 PPO 更新的轮数

- `critic.ppo_epochs`：在一组采样轨迹上对 critic 进行 PPO 更新的轮数。默认为 `actor_rollout_ref.actor.ppo_epochs`

- `algorithm.gemma`：折扣因子

- `algorithm.lam`：在 GAE 估计器中权衡偏差和方差的 lambda 项

- `algorithm.adv_estimator`：支持 gae、grpo、reinforce_plus_plus、reinforce_plus_plus_baseline、rloo

## 高级扩展