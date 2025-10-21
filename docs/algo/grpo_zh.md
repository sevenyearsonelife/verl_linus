# 组群相对策略优化 (GRPO)

最后更新：2025年05月31日。

在强化学习中，像 PPO 这样的经典算法依赖于"评论者"模型来评估动作的价值，指导学习过程。然而，训练这个评论者模型可能会消耗大量资源。

GRPO 通过消除对独立评论者模型的需求来简化这个过程。它的运作方式如下：
- **组群采样**：对于给定问题，模型生成多个可能的解决方案，形成一个输出"组群"。
- **奖励分配**：每个解决方案根据其正确性或质量进行评估并分配奖励。
- **基线计算**：组群的平均奖励作为基线。
- **策略更新**：模型通过比较每个解决方案的奖励与组群基线来更新参数，强化优于平均水平的解决方案， discourage 低于平均水平的解决方案。

这种方法通过避免训练独立的价值评估模型来减少计算开销，使学习过程更高效。更多详情请参考原始论文 [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)

## 核心组件

- **无价值函数（无评论者）**：与 PPO 不同，GRPO 不训练独立的价值网络（评论者）
- **组群采样（分组推出）**：GRPO 不为每个输入评估一个推出，而是为每个提示从当前策略生成多个完成（响应）。这组完成被称为一个组群。
- **相对奖励**：在每个组群内，完成被评分（例如基于正确性），奖励相对于组群进行归一化。

## 配置

注意，所有包含 `micro_batch_size` 的配置都用于配置每次前向或反向传递的最大样本或令牌数量，以避免 GPU OOM，其值不应改变算法/收敛行为。

尽管许多配置以 `ppo_` 前缀开头，但它们在 verl 中的不同 RL 算法中都能工作，因为 GRPO 训练循环与 PPO 类似（没有评论者）。

![image](https://github.com/user-attachments/assets/16aebad1-0da6-4eb3-806d-54a74e712c2d)

- `actor_rollout.ref.rollout.n`：对于每个提示，采样 n 次。默认为 1。对于 GRPO，请将其设置为大于 1 的值以进行组群采样。

- `data.train_batch_size`：用于生成一组采样轨迹/推出的提示的全局批次大小。响应/轨迹的数量为 `data.train_batch_size * actor_rollout.ref.rollout.n`

- `actor_rollout_ref.actor.ppo_mini_batch_size`：采样的轨迹集被分割成多个小批次，batch_size=ppo_mini_batch_size，用于 PPO actor 更新。ppo_mini_batch_size 是所有工作器的全局大小。

- `actor_rollout_ref.actor.ppo_epochs`：在一组采样轨迹上对 actor 进行 GRPO 更新的轮数

- `actor_rollout_ref.actor.clip_ratio`：GRPO 裁剪范围。默认为 0.2

- `algorithm.adv_estimator`：默认为 gae。请将其设置为 grpo

- `actor_rollout_ref.actor.loss_agg_mode`：默认为 "token-mean"。选项包括 "token-mean"、"seq-mean-token-sum"、"seq-mean-token-mean"。原始 GRPO 论文采用样本级损失（seq-mean-token-mean），在长 CoT 场景中可能不稳定。verl 中提供的所有 GRPO 示例脚本都使用默认配置 "token-mean" 进行损失聚合。

GRPO 不是在奖励中添加 KL 惩罚，而是通过直接将训练策略与参考策略之间的 KL 散度添加到损失中进行正则化：

- `actor_rollout_ref.actor.use_kl_loss`：在 actor 中使用 kl 损失。使用时，我们不在奖励函数中应用 KL。默认为 False。对于 GRPO，请将其设置为 True。

- `actor_rollout_ref.actor.kl_loss_coef`：kl 损失的系数。默认为 0.001。

- `actor_rollout_ref.actor.kl_loss_type`：支持 kl(k1)、abs、mse(k2)、low_var_kl(k3) 和 full。在末尾附加 "+"（例如，'k1+' 和 'k3+'）将应用 straight through 来使用 k2 进行无偏梯度估计，无论 kl 值估计如何（更多详情参见 https://github.com/volcengine/verl/pull/2953#issuecomment-3162113848）。如何计算 actor 和参考策略之间的 kl 散度。详细分析请参见此博客文章：http://joschu.net/blog/kl-approx.html