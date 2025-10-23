# 生成奖励模型

## 脚本

### 第1步：启动vLLM服务器（可选）

使用vLLM部署预训练的GenRM模型。如果要使用外部API服务，请跳过此步骤。

```bash
vllm serve verl-team/GenRM-CI-Test-1.5B --served-model-name genrm-demo
```

### 第2步：使用GenRM进行强化学习

```bash
bash recipe/api-genrm/run_genrm_remote.sh
```

实现通过传递自定义奖励函数工作（参见`reward_function.py`）

为方便起见，我们在同一台机器上运行RL训练和服务器。要使用外部服务器，首先在`reward_function.py`中配置`BASE_URL`和`API_KEY`。

## 高级：自定义您的GenRM

您可以使用带数据并行的sglang服务器进行更快的推理：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang_router.launch_server --model-path verl-team/GenRM-CI-Test-1.5B --dp-size 4
```

请注意，您应该修改`reward_function.py`中的`BASE_URL`以匹配您的SGLang服务器地址。

您也可以通过实现自定义奖励函数来创建自己的定制GenRM。以下是基于`reward_function.py`定制自己的GenRM的一些技巧：

- 为您的GenRM设计适当的提示
- 将GenRM响应转换为RL奖励
- ...

由于这些方面高度灵活，我们只提供演示实现。GenRM的实际设计和实现由用户自行决定。