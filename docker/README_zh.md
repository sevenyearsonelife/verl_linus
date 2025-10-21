# verl 的 Dockerfile

我们提供预构建的 Docker 镜像以供快速设置。从这个版本开始，我们利用新的镜像发布层次来提高生产力和稳定性。

镜像类型分为三大类：

- **基础镜像**：不包含推理和训练框架，仅安装基本依赖。可以直接在其上安装 vllm 或 SGLang，无需重新安装 torch 或 CUDA。
- **应用镜像**：安装了推理和训练框架的稳定版本。
- **预览镜像**：包含最新框架和功能的不稳定版本。

前两种类型的镜像托管在 dockerhub [verlai/verl](https://hub.docker.com/r/verlai/verl) 仓库中，而预览镜像托管在社区仓库中。

> 镜像版本与 verl 发布版本映射，例如，标签为 ``verl0.4`` 的镜像为 verl 发布版本 ``v0.4.x`` 构建。

## 基础镜像

稳定的基础镜像是 ``verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.1-fa2.7.4``，有不同的 CUDA 版本。

基础镜像的更新不频繁，应用镜像可以在其上构建而无需重新安装基础包。

## 应用镜像

从这个版本开始，我们将为 vLLM 和 SGLang 构建的镜像分开，因为像 FlashInfer 这样的依赖包存在分歧。
有两种类型的应用镜像可用：

- **vLLM 与 FSDP 和 Megatron**：``verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2``
- **SGLang 与 FSDP 和 Megatron**：`verlai/verl:app-verl0.5-transformers4.55.4-sglang0.4.10.post2-mcore0.13.0-te2.2`

带有 Megatron 后端的 Docker 镜像可以运行大语言模型，如 ``Qwen/Qwen3-235B-A22B``、``deepseek-ai/DeepSeek-V3-0324`` 的后训练。更多详情请参考 :doc:`大语言模型后训练文档<../perf/dpsk>`。

应用镜像可以频繁更新，Dockerfile 可以在 ``docker/verl[version]-[packages]/Dockerfile.app.[frameworks]`` 中找到。基于基础镜像，很容易构建自己的应用镜像，包含所需的推理和训练框架。

## 社区镜像

对于带有 FSDP 的 vLLM，请参考 [hiyouga/verl](https://hub.docker.com/r/hiyouga/verl) 仓库，最新版本是 ``hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0``。

对于带有 FSDP 的 SGLang，请参考 [ocss884/verl-sglang](https://hub.docker.com/r/ocss884/verl-sglang) 仓库，最新版本是 ``ocss884/verl-sglang:ngc-th2.6.0-cu126-sglang0.4.6.post5``，由 SGLang RL Group 提供。

对于最新的带 Megatron 的 vLLM，请参考 [iseekyan/verl](https://hub.docker.com/r/iseekyan/verl) 仓库，最新版本是 ``iseekyan/verl:nemo.gptoss_vllm0.11.0``。

基于 NGC 的镜像请参见 ``docker/`` 下的文件，或者如果您想构建自己的镜像。

请注意，对于带有 EFA 网络接口的 aws 实例（Sagemaker AI Pod），您需要安装 EFA 驱动，如 ``docker/Dockerfile.extenstion.awsefa`` 所示。

## 从 Docker 安装

拉取所需的 Docker 镜像并安装所需的推理和训练框架后，您可以按以下步骤运行：

1. 启动所需的 Docker 镜像并附加到其中：

```sh
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl <image:tag> sleep infinity
docker start verl
docker exec -it verl bash
```

2. 如果您使用提供的镜像，只需要安装 verl 本身而无需依赖：

```sh
# 安装夜间版本（推荐）
git clone https://github.com/volcengine/verl && cd verl
pip3 install --no-deps -e .
```

[可选] 如果您希望在 different 框架之间切换，可以使用以下命令安装 verl：

```sh
# 安装夜间版本（推荐）
git clone https://github.com/volcengine/verl && cd verl
pip3 install -e .[vllm]
pip3 install -e .[sglang]
```