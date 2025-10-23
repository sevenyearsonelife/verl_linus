<div align="center">

# 大型语言模型推理的强化学习熵机制

[![论文](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2505.22617)  [![Github](https://img.shields.io/badge/PRIME-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL) [![alphaXiv](https://img.shields.io/badge/discussion-A42C25?style=for-the-badge&logo=arxiv&logoColor=white&color=blue
)](https://www.alphaxiv.org/abs/2505.22617) [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/stingning/status/1928088554166505667) [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/charlesfornlp/status/1928089451080585283) [![Twitter-ak](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/_akhaliq/status/1928077929105268861)


<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#🎉新闻" style="text-decoration: none; font-weight: bold;">🎉 新闻</a> •
    <a href="#✨快速入门" style="text-decoration: none; font-weight: bold;">✨ 快速入门</a> •
    <a href="#📖介绍" style="text-decoration: none; font-weight: bold;">📖 介绍</a>
  </p>
  <p>
    <a href="#🎈引用" style="text-decoration: none; font-weight: bold;">🎈 引用</a> •
    <a href="#🌻致谢" style="text-decoration: none; font-weight: bold;">🌻 致谢</a> •
    <a href="#📬联系" style="text-decoration: none; font-weight: bold;">📬 联系</a> •
    <a href="#📈星标历史" style="text-decoration: none; font-weight: bold;">📈 星标历史</a>
  </p>
</div>

</div>

# 🎉新闻

- **[2025/05/29]** 🎉 在[Huggingface每日论文](https://huggingface.co/papers?date=2025-05-29)中排名**#1**。
- **[2025/05/29]** 在arXiv上发布了我们的论文。参见[这里](https://arxiv.org/pdf/2505.22617)。我们提供了对LLMs强化学习熵机制的洞察，并提出了两种简单有效的策略来缓解熵崩溃。

# ✨快速入门

准备好训练数据后，在单节点上训练Qwen2.5-7B，以KL-Cov方法为例，您可以简单地运行：

```
cd verl
conda activate your_env
bash recipe/dapo/7b_kl_cov.sh
```

而对于在多节点上训练Qwen2.5-32B，您可以运行以下命令：

```
cd verl
conda activate your_env
bash recipe/dapo/32b_kl_cov.sh
```

# 📖介绍

<div align="left">
  <img src="https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/figures/e2a.jpg?raw=true" alt="issue" style="width: 96%; height: auto;">
</div>

本文解决了大型语言模型（LLMs）强化学习（RL）扩展中的熵崩溃问题，其中策略熵在训练期间急剧下降，导致过度自信和性能饱和。我们经验性地建立了熵（$H$）和性能（$R$）之间的关系：$R=-aexp(H)+b$，表明性能受熵耗尽瓶颈限制。

<div align="left">
  <img src="https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/figures/cov.jpg?raw=true" alt="issue" style="width: 96%; height: auto;">
</div>

理论上，我们发现熵变化由动作概率和logit更新之间的协方差驱动，这与策略梯度方法中的优势相关。高概率、高优势的动作减少熵，而罕见、高优势的动作增加熵。根据经验，协方差项保持正值，解释了熵的单调下降。为了缓解这个问题，我们提出了​​Clip-Cov​​和​​KL-Cov​​，它们限制高协方差token的更新。这些方法有效地防止熵崩溃，并提高性能。

# 📃评估

<div align="left">
  <img src="https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/figures/performance_fig.jpg?raw=true" alt="issue" style="width: 96%; height: auto;">
</div>

我们的方法能够在整个训练过程中保持相当高的熵水平。例如，当基线的熵达到平台期且无法再被消耗时，KL-Cov方法仍然维持超过10倍高的熵水平。同时，策略模型的响应长度稳步增加，其在测试集上的性能始终超过基线。这表明我们的模型能够在训练期间更自由地探索，通过RL学习更好的策略。

| **方法**        | **AIME24** | **AIME25** |  **AMC** | **MATH-500** | **OMNI-MATH** | **OlympiadBench** | **Minerva** | **平均** |
| ----------------- | ---------: | ---------: | -------: | -----------: | ------------: | ----------------: | ----------: | -------: |
| *Qwen2.5-7B*      |            |            |          |              |               |                   |             |          |
| GRPO              |       21.2 |        9.6 |     58.7 |         78.8 |          27.9 |              40.7 |        36.7 |     38.6 |
| w. Clip-higher    |       18.1 |       11.5 |     56.6 |         79.2 |          29.8 |              43.3 |        40.4 |     38.8 |
| w. **`CLIP-Cov`** |       22.1 |   **15.8** |     58.2 |         80.4 |      **30.5** |          **44.1** |    **41.1** |     40.4 |
| w. **`KL-Cov`**   |   **22.6** |       12.9 | **61.4** |     **80.8** |          29.1 |              42.6 |        38.2 | **40.6** |
| *Qwen2.5-32B*     |            |            |          |              |               |                   |             |          |
| GRPO              |       21.8 |       16.2 |     69.7 |         84.2 |          35.2 |              43.6 |        45.5 |     45.8 |
| w. Clip-higher    |       35.6 |       22.3 |     69.5 |         77.2 |          35.1 |              42.5 |        43.0 |     47.2 |
| w. **`CLIP-Cov`** |       32.3 |       22.7 |     67.2 |     **87.0** |      **42.0** |          **57.2** |        46.0 |     50.3 |
| w. **`KL-Cov`**   |   **36.8** |   **30.8** | **74.5** |         84.6 |          39.1 |              49.0 |    **46.3** | **52.2** |

我们的两种方法都在所有基准测试中取得了显著的改进。与GRPO相比，我们的方法在7B模型上平均优于2.0%，在32B模型上优于6.4%。此外，我们观察到我们的方法在更大的Qwen2.5-32B上产生更大的收益。具体来说，在最具挑战性的基准测试AIME24和AIME25上，我们的方法相比GRPO分别实现了15.0%和14.6%的改进。

# 🎈引用
如果您觉得这篇论文或仓库有帮助，请引用我们。

```bibtex
@article{cui2025entropy,
  title={The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models},
  author={Cui, Ganqu and Zhang, Yuchen and Chen, Jiacheng and Yuan, Lifan and Wang, Zhi and Zuo, Yuxin and Li, Haozhan and Fan, Yuchen and Chen, Huayu and Chen, Weize and others},
  journal={arXiv preprint arXiv:2505.22617},
  year={2025}
}
```

# 🌻致谢
我们从[verl](https://github.com/volcengine/verl)扩展实现了我们的强化学习算法。我们利用[vLLM](https://github.com/vllm-project/vllm)进行推理。我们的模型主要在[Qwen2.5系列](https://github.com/QwenLM/Qwen2.5)上训练。我们的训练数据从[DAPO-MATH](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)构建。感谢他们的伟大贡献！

# 📬 联系

如有问题、讨论或合作机会，请随时联系：
- Ganqu Cui: cuiganqu@pjlab.org.cn
- Yuchen Zhang: yuchen.zhang2003@gmail.com
- Jiacheng Chen: jackchan9345@gmail.com
- Ning Ding: ningding.cs@gmail.com