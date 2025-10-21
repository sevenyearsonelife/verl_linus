# 模型
像 huggingface/transformers 这样的常用模型库在使用 Pytorch 原生模型并行性时表现困难。遵循 vLLM 的设计原则，我们在 verl 中保持简单、可并行化、高度优化的打包输入。
## 添加新的 Huggingface 模型
### 步骤 1：将模型文件从 HF 复制到 verl
- 在 verl/models/hf 下添加新文件
- 仅将模型文件从 huggingface/transformers/models 复制到 verl/models/hf

### 步骤 2：修改模型文件以使用打包输入
- 删除所有与推理相关的代码（kv 缓存）
- 修改输入仅包括
    - input_ids (total_nnz,)
    - cu_seqlens (total_nnz + 1,)
    - max_seqlen_in_batch: int
- 请注意，这需要使用带有因果掩码的 flash attention。

### 步骤 2.5：添加测试
- 添加测试以比较此版本和 huggingface 版本
- 遵循基础设施并将测试添加到 tests/models/hf

### 步骤 3：添加应用张量并行性的函数
- 请遵循
    - https://pytorch.org/docs/stable/distributed.tensor.parallel.html
    - https://pytorch.org/tutorials/intermediate/TP_tutorial.html
- 一般注释
    - 原生 Pytorch 中的张量并行性不是自动并行性。其工作方式是使用配置指定如何重新分片模型参数和输入/输出。然后这些配置被注册为钩子，在模型前向之前/之后执行输入/输出重新分片。

### 步骤 4：添加应用数据并行性的函数
- 请使用 FSDP2 API
- 在此处查看演示 https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L413

### 步骤 5：添加应用流水线并行性的函数
- 在 Pytorch 2.4 中提供
- 目前仅在夜间版本的 alpha 中
- 查看 torchtitan 了解更多详情