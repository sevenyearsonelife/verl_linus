# 测试布局

tests/ 下的每个文件夹对应于 verl 中子命名空间的测试类别。例如：
- `tests/trainer` 用于测试与 `verl/trainer` 相关的功能
- `tests/models` 用于测试与 `verl/models` 相关的功能
- ...

有几个带有 `special_` 前缀的文件夹，为特殊目的创建：
- `special_distributed`：必须在多个 GPU 上运行的单元测试
- `special_e2e`：使用训练/生成脚本的端到端测试
- `special_npu`：NPU 测试
- `special_sanity`：一套快速健全性测试
- `special_standalone`：设计为在专用环境中运行的一组测试

测试加速器
- 默认情况下，测试在有 GPU 可用时运行，除了 `special_npu` 下的测试，以及任何名称以 `on_cpu.py` 结尾的测试脚本。
- 对于名称以 `on_cpu.py` 结尾的测试脚本，将在 linux 环境中的 CPU 资源上测试。

# 工作流程布局

所有 CI 测试都由 `.github/workflows/` 中的 yaml 文件配置。以下是所有测试配置的概述：
1. 一总是触发的 CPU 健全性测试列表：`check-pr-title.yml`、`secrets_scan.yml`、`check-pr-title,yml`、`pre-commit.yml`、`doc.yml`
2. 一些繁重的多 GPU 单元测试，如 `model.yml`、`vllm.yml`、`sgl.yml`
3. 端到端测试：`e2e_*.yml`
4. 单元测试
  - `cpu_unit_tests.yml`，在所有文件名模式为 `tests/**/test_*_on_cpu.py` 的脚本上运行 pytest
  - `gpu_unit_tests.yml`，在所有没有 `on_cpu.py` 后缀的文件脚本上运行 pytest。
  - 由于 cpu/gpu 单元测试默认运行 `tests` 下的所有测试，请确保在以下情况下手动排除测试：
    - 新的工作流程 yaml 添加到 `.github/workflows`
    - 新测试添加到上述 2 中提到的工作流程