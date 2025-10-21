# 为 verl 做贡献

感谢您考虑为 verl 做出贡献！我们欢迎任何形式的贡献——无论是错误修复、功能增强、文档改进，还是仅仅提供反馈。无论您是经验丰富的开发者，还是这是您的第一个开源项目，您的帮助都是无价的。

您的支持可以采取多种形式：
- 报告问题或意外行为
- 建议或实现新功能
- 改进或扩展文档
- 审查拉取请求并协助其他贡献者
- 宣传推广：在博客文章、社交媒体中分享 verl，或给仓库点个 ⭐

## 寻找可贡献的问题

想要开始贡献？查看这些问题：
- [适合初学者的问题](https://github.com/volcengine/verl/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
- [征集贡献](https://github.com/volcengine/verl/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22call%20for%20contribution%22)
此外，您可以通过 [RFC](https://github.com/volcengine/verl/issues?q=is%3Aissue%20state%3Aopen%20label%3ARFC) 和 [路线图](https://github.com/volcengine/verl/issues?q=state%3Aopen%20label%3A%22roadmap%22) 了解开发计划和路线图。

## 开发

- **仅 Python 环境**：通过 `pip install -e .[test,vllm]` 或 `pip install -e .[test,sglang]` 安装 verl 并快速迭代。完整的依赖设置请查看 verl [安装文档](https://verl.readthedocs.io/en/latest/start/install.html)。

## 代码检查和格式化

我们依赖 pre-commit 来保持代码一致性。设置方法：

```bash
pip install pre-commit
pre-commit install
# 对暂存的更改运行
pre-commit run
# 对仓库中的所有文件运行
pre-commit run --all-files
# 使用 pre-commit 运行特定的钩子
# pre-commit run --all-files --show-diff-on-failure --color=always <hook-id>
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
```

## 测试

我们的测试套件在 GitHub Actions 上运行。查看这些工作流程以获取详细信息：
- [GPU 单元测试](https://github.com/volcengine/verl/blob/main/.github/workflows/gpu_unit_tests.yml)
- [CPU 单元测试](https://github.com/volcengine/verl/blob/main/.github/workflows/cpu_unit_tests.yml)
- [vLLM 测试](https://github.com/volcengine/verl/blob/main/.github/workflows/vllm.yml)
- [SGLang 测试](https://github.com/volcengine/verl/blob/main/.github/workflows/sgl.yml)

### 添加 CI 测试

如果可能，请为您的新功能添加 CI 测试：

1. 找到最相关的工作流程 yml 文件，通常对应于一个 `hydra` 默认配置（如 `ppo_trainer`、`ppo_megatron_trainer`、`sft_trainer` 等）。
2. 如果尚未包含，请将相关路径模式添加到 `paths` 部分。
3. 最小化测试脚本的工作量（参考现有脚本以获取示例）。

## 构建文档

```
# 确保 verl 在您的 PYTHONPATH 中，例如：
pip install -e .[test]

# 安装文档依赖
pip install -r requirements-docs.txt

# 生成 HTML 文档
make clean
make html

# 本地预览
python -m http.server -d _build/html/
```
在浏览器中打开 http://localhost:8000 来浏览文档。

## 拉取请求和代码审查

感谢提交 PR！为了简化审查流程：
- 遵循我们的拉取请求模板，包括标题格式和检查清单。
- 遵守我们的 pre-commit 检查规则，确保所有检查都通过。
- 为面向用户的更改更新文档。
- 在 CI 工作流程中添加或更新测试，或解释为什么测试不适用。

## 许可证

完整详情请参阅 [LICENSE](https://github.com/volcengine/verl/blob/main/LICENSE) 文件。

## 感谢您

我们感谢您对 verl 的贡献。您的努力帮助使项目变得更强大、更用户友好。编码愉快！