# 字符计数

## 简介
字符计数是一个简单的NLP任务。我们创建它是为了让初学者理解RLVR（强化学习版本推理）的概念。这个任务可以在只有8GB显存的消费级GPU上使用小模型（例如 https://huggingface.co/HuggingFaceTB/SmolLM2-135M）进行训练。

## 问题建模
提示是："{word}中有多少个{char}？"。为了让LLM更好地回答这个问题，我们创建了包含中间步骤的SFT数据集。例如，

```text
问题：n-i-n-e中有多少个n？
答案：
n = n
i != n
n = n
e != n
\boxed{2}
```

注意：
- 我们在每个单独字符之间添加连字符，使任务更容易，因为大多数分词器会将每个单独字符分词为相同的token。
- 在SFT数据集中，我们通过列出所有单独字符以及它们是否等于目标来创建CoT（思维链）。最后，它输出框内的最终答案。
- 这个任务可以被验证。
- 单词不一定有意义。每个字符从a到z均匀采样。我们使总长度和答案在范围内均匀分布。

## 脚本
要创建数据集，运行
```bash
python3 create_dataset.py
```
我们创建训练集和验证集。两者都用于SFT和RL。您可以指定数据总数、最小/最大长度和数据路径。

运行SFT
```bash
bash train_sft.sh
```
我们训练SFT 3个epoch。3个epoch后，验证分数约为0.12。

运行GRPO
```bash
bash train_grpo.sh
```
我们训练GRPO 2个epoch。2个epoch后，验证分数约为0.36。