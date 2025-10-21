# Repository Guidelines

## 项目结构与模块组织
verl 主库位于 `verl/`，包含 trainer、workers、algorithms 等子包，是实现 RLHF 数据流的核心。标准训练方案集中在 `recipe/`，每个子目录提供独立的配置与说明。自动化脚本放在 `scripts/`，用于环境检测、模型部署等任务；容器与集群示例在 `docker/`。测试按照硬件和主题拆分在 `tests/`，常见后缀如 `_on_cpu`、`_integration` 便于筛选。文档素材位于 `docs/`，快速上手示例可在 `examples/` 查阅。元数据与打包逻辑统一在 `pyproject.toml` 与 `setup.py`，依赖矩阵以 `requirements*.txt` 和 `requirements_sglang.txt` 描述不同硬件配置。版权与第三方声明集中在 `LICENSE` 与 `Notice.txt`，引入外部模型或数据前请确认授权兼容。

## 构建、测试与开发命令
建议使用 Python 3.10+ 虚拟环境：
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[test]
```
- `pytest -q`：运行默认 CPU 测试，CI 亦采用该入口。
- `pytest tests/trainer/ppo/test_rollout_is_integration.py -v`：验证 PPO rollout 集成链路。
- `pre-commit run --all-files --show-diff-on-failure --color=always ruff`：执行 Ruff 规则与生成式配置校验。
- `pre-commit install`：在本地启用提交钩子，阻断不符合规范的变更。
- `python -m build`：本地生成分发包时使用。
- `pip install -r requirements-cuda.txt` 或 `requirements-npu.txt`：按目标加速硬件同步依赖。
- `python3 -m uv pip install -e .[sglang]`：在需要 SGLang 相关功能时快捷安装扩展依赖。

## 编码风格与命名约定
全局遵循 PEP 8，统一四空格缩进。Ruff 行宽 120，保留 `F405`、`E731` 等必要豁免。模块、函数、变量使用 `snake_case`；类与异常采用 `PascalCase`；YAML 配置以短横线命名。提交前运行 `ruff check .`，如启用格式化同时运行 `ruff format`。类型检查通过 `mypy`（`pyproject.toml` 已开启 pretty 输出），新增模块应避免扩大 `ignore_errors` 范围。公共 API 保持清晰 docstring，确保 sphinx 文档可自动生成；新增配置字段需在相应 dataclass 或 schema 中补充默认值和注释。

## 测试规范
所有功能改动需覆盖 `pytest` 用例；涉及分布式或 GPU 的逻辑在 `tests/utils/` 或 `tests/trainer/` 内补充场景化测试。测试文件保持 `test_*.py` 命名，长耗时或硬件依赖用 `@pytest.mark.skipif` 控制。浮点断言优先使用 `pytest.approx`，避免环境噪声导致波动。持续关注覆盖率指标（CI 中的 `coverage.xml` 导出），新增模块建议涵盖关键路径和失败分支，并在文档中写明复现步骤。

## 提交与 Pull Request 要求
沿用历史格式 `[scope] type: summary (#issue)`，scope 如 `[worker]`、`[data]` 与目录一致，type 采用 `feat`、`fix`、`refactor` 等。每次提交需确保 `ruff`、`pytest` 全部通过，并同步更新相关配置或脚本。PR 描述包含：变更要点、影响面、验证步骤（日志或截图）、关联 issue；跨设备改动需写明测试硬件。保持 PR 聚焦单一主题，避免引入无关重构；如需同时修改多个子系统，请拆分为独立 PR 并使用草稿模式同步进度。

## 安全与配置提示
敏感凭证通过环境变量或外部密钥管理注入，禁止写入仓库。核心配置位于 `verl/trainer/config/*.yaml`，修改时需同步文档与示例。进行多机训练前显式设置 `CUDA_VISIBLE_DEVICES`、`MASTER_ADDR`、`MASTER_PORT` 等变量，脚本中避免硬编码集群参数。团队内部共享的 `.env` 文件仅用于本地开发，提交前请确认未被 git 追踪，必要时更新 `.gitignore` 规则确保隔离。如需额外部署模板，请参阅 `docs/start/install.rst` 的分步指南。
