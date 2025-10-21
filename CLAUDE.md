# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

VERL (Volcano Engine Reinforcement Learning for LLMs) 是一个专为大型语言模型设计的强化学习训练框架，基于 HybridFlow 论文实现。该项目采用混合控制器架构，支持多种RL算法（PPO、GRPO、RLOO等）和多种训练后端（FSDP、Megatron-LM）。

## 常用命令

### 安装与环境设置
```bash
# 基础安装
pip install -e .

# 安装可选依赖（根据需要选择）
pip install -e ".[vllm]"      # vLLM推理引擎
pip install -e ".[sglang]"    # SGLang推理引擎
pip install -e ".[test]"      # 测试依赖
pip install -e ".[gpu]"       # GPU优化（flash-attn, liger-kernel）
pip install -e ".[math]"      # 数学验证工具

# 安装vLLM、SGLang和Megatron支持
scripts/install_vllm_sglang_mcore.sh
```

### 训练命令
```bash
# GRPO训练示例（GSM8K数据集，Qwen2-7B模型）
bash examples/grpo_trainer/run_qwen2-7b.sh

# PPO训练示例
bash examples/ppo_trainer/run_qwen2-7b.sh

# SFT训练示例
bash examples/sft/gsm8k/run_qwen_05.sh

# 多轮对话训练
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

### 测试命令
```bash
# 运行所有测试
pytest tests/

# 运行特定模块测试
pytest tests/test_protocol_on_cpu.py
pytest tests/trainer/ppo/
pytest tests/workers/

# 运行CPU测试（避免GPU依赖）
pytest tests/special_sanity/
pytest tests/utils/ -k "cpu"
```

### 工具脚本
```bash
# 生成训练器配置
python scripts/print_cfg.py --config-path verl/trainer/config --config-name ppo_trainer

# 模型格式转换（HF到Megatron）
python scripts/converter_hf_to_mcore.py

# 诊断工具
python scripts/diagnose.py

# 查看rollout结果
python scripts/rollout_viewer.py
```

## 高级架构

### 核心编程模型

**混合控制器架构（Hybrid-Controller）**：
- **控制流**：单进程执行RL算法的高级算子顺序（rollout → advantage computation → training）
- **计算流**：多进程分布式执行神经网络计算（前向/反向传播/优化器）

**DataProto协议**：基于TensorDict的统一数据传输协议，支持：
- 自动填充和批处理
- 分布式数据聚合
- 序列化/反序列化用于Ray远程传输

### 主要组件架构

**训练器模块** (`verl/trainer/`)：
- `main_ppo.py`：PPO训练主入口，Ray集群初始化
- `RayPPOTrainer`：分布式PPO训练器，管理工作组构建和训练循环
- `fsdp_sft_trainer.py`：基于FSDP的监督微调训练器

**工作器模块** (`verl/workers/`)：
- `ActorRolloutRefWorker`：管理actor、rollout和参考策略的混合工作器
- `FSDPPOActor` vs `MegatronPPOActor`：不同训练后端的Actor实现
- `rollout/`：推理引擎集成（vLLM、SGLang、HF Transformers）

**单一控制器** (`verl/single_controller/`)：
- `Worker`：分布式计算基本单元
- `WorkerGroup`：工作器集合管理
- `ResourcePool`：跨节点资源池管理

### 配置系统

采用Hydra配置管理，结构层次：
```
ppo_trainer.yaml
├── actor_rollout_ref/     # 策略、推理、参考模型配置
│   ├── actor/            # 策略模型（学习率、批大小等）
│   ├── rollout/          # 推理引擎配置
│   └── ref/              # 参考模型配置
├── critic/               # 价值模型配置
├── reward_model/         # 奖励模型配置
├── data/                 # 数据配置（路径、长度限制等）
└── algorithm/            # 算法配置（PPO、GRPO等参数）
```

### 算法支持

**核心算法**：
- PPO（近端策略优化）
- GRPO（Group Relative Policy Optimization）
- RLOO（Reward-based Leave-One-Out）
- ReMax、REINFORCE++等

**高级算法**（Recipe系统，`recipe/`目录）：
- DAPO：在AIME 2024达到SOTA的数学推理算法
- SPPO：Self-play偏好优化
- PRIME：Process reinforcement through implicit rewards

### 多后端支持

**训练后端**：
- **FSDP/FSDP2**：PyTorch原生分布式训练，支持LoRA、参数高效微调
- **Megatron-LM**：支持大规模MoE模型（DeepSeek-671B、Qwen3-235B），提供张量并行、流水线并行、专家并行

**推理引擎**：
- **vLLM**：高性能推理，支持tensor parallelism
- **SGLang**：支持多轮对话、工具调用、多模态
- **HF Transformers**：标准推理后端

## 开发指南

### 添加新模型

1. 在`verl/models/registry.py`注册新模型
2. 在`verl/models/transformers/`添加模型配置
3. 更新`verl/trainer/config/model/`中的模型配置文件

### 实现新算法

1. 继承`verl/trainer/po/core_algos.py`中的基础算法类
2. 在`verl/trainer/config/algorithm/`添加算法配置
3. 在`recipe/`目录创建完整实现示例

### 添加新的奖励函数

1. 在`verl/workers/reward_manager/`注册奖励函数
2. 实现`compute_reward()`方法，返回DataProto格式结果
3. 更新配置文件指定奖励函数

### 性能调优

**内存优化**：
- 使用`actor_rollout_ref.actor.fsdp_config.param_offload=True`
- 启用`actor_rollout_ref.model.enable_gradient_checkpointing=True`
- 考虑使用LoRA减少参数量

**速度优化**：
- 调整`ppo_mini_batch_size`和`ppo_micro_batch_size_per_gpu`
- 使用sequence packing：`data.use_sequence_balance=True`
- 启用liger-kernel：安装`".[gpu]"`依赖

**多节点部署**：
- 设置`trainer.nnodes`和`trainer.n_gpus_per_node`
- 配置Ray集群：`ray start --head`
- 使用Megatron后端处理超大模型

### 调试和监控

**实验跟踪**：
```yaml
trainer.logger='["console","wandb"]'  # 或 "mlflow", "tensorboard"
trainer.project_name='my_project'
trainer.experiment_name='my_experiment'
```

**性能分析**：
```python
# 使用内置profiler
python -m verl.trainer.main_ppo ... trainer.profiler=true

# 使用NVTX
export NVTX_PROFILER=1
python -m verl.trainer.main_ppo ...
```

**常见问题排查**：
- 检查GPU内存：`nvidia-smi`
- 验证Ray集群：`ray status`
- 查看详细日志：设置环境变量`RAY_BACKEND_LOG_LEVEL=debug`

## 测试策略

**测试类型**：
- 单元测试：测试单个组件功能
- 集成测试：测试组件间协作
- 端到端测试：测试完整训练流程
- CPU测试：避免GPU依赖的基础测试

**运行测试**：
```bash
# 快速验证
pytest tests/special_sanity/ -v

# 完整测试（需要GPU）
pytest tests/ --timeout=300

# 特定功能测试
pytest tests/trainer/ppo/test_core_algos_on_cpu.py
```

## 文档和资源

- **官方文档**：https://verl.readthedocs.io/
- **安装指南**：https://verl.readthedocs.io/en/latest/start/install.html
- **快速开始**：https://verl.readthedocs.io/en/latest/start/quickstart.html
- **性能调优**：https://verl.readthedocs.io/en/latest/perf/perf_tuning.html
- **算法基准**：https://verl.readthedocs.io/en/latest/algo/baseline.html