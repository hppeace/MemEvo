# MemEvo

MemEvo 是一个简洁的对话记忆系统评测框架。当前提供 LoCoMo 数据集加载、
完整上下文（Full Context）与 Mem0 基线、OpenAI 兼容模型客户端、token
用量统计和 LLM-as-a-Judge 评测。

## 快速开始

项目要求 Python 3.12，推荐使用 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync
cp .env.example .env
cp src/configs/full_context.example.toml config.toml
```

把 LoCoMo JSON 数据放到 `data/locomo10.json`，然后在 `.env` 中填写模型
API key，并在 `config.toml` 中设置模型名称和 OpenAI 兼容服务地址。

```bash
uv run memevo --config config.toml
```

运行结果写入配置中的 `output_dir`：

- `answers.json`：各问题的标准答案和模型回答；
- `evaluation.json`：judge 结果和准确率；
- `usage.json`：MemEvo 管理的 answer/judge 模型调用次数和 token 用量；
- `memory/`：算法按 conversation 保存的中间记忆。

程序会在每个回答后刷新 `answers.json`，中断时也能保留已完成结果。
`usage.json` 按配置中的模型名称统计 token，并在程序退出前刷新。
终端会显示 conversation、阶段操作和 judge 问题的进度条。
`retrieve`、`answer` 和 `judge` 会并发执行，默认并发数为 32，可通过
`[run] concurrency` 调整。

## Mem0 基线

Mem0 适配直接使用官方 `mem0ai` 库，不依赖 `memory-benchmarks`：

```bash
cp src/configs/mem0.example.toml config.toml
uv run memevo --config config.toml
```

在 `.env` 中配置 `MEM0_LLM_API_KEY`、`MEM0_EMBEDDING_API_KEY`、
`ANSWER_API_KEY` 和 `JUDGE_API_KEY`。Mem0 默认使用输出目录下的本地 Qdrant，
逐条写入 LoCoMo turn，再按问题检索 top-200 memories，并使用分数最高的 10 条
生成答案。可通过 `[algorithm] top_k` 和 `cutoff` 分别调整。Mem0 内部抽取 LLM
和 embedding 的 token 由 Mem0 自己管理，不计入 `usage.json`。

首次运行前安装 Mem0 OSS 混合检索使用的英文 spaCy 模型：

```bash
uv run python -m spacy download en_core_web_sm
```

## 扩展实验

runner 只编排通用阶段，不依赖具体算法和数据集。新增实现后，在对应包中注册：

```python
from memevo.algorithms import register_algorithm
from memevo.datasets import register_dataset

register_algorithm("my_memory", my_factory)
register_dataset("my_dataset", my_factory)
```

然后在 TOML 的 `[algorithm] name` 或 `[dataset] name` 中选择。算法 factory
会收到命名模型池、工作目录和算法配置；通过 `models.llm("answer")` 或
`models.embedder("embedding")` 获取模型。模型调用会自动归入当前实验阶段。

## 开发

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

新增算法时继承 `src/memevo/algorithms/base/algorithm.py` 中的 `BaseAlgorithm`，
并实现 `ingest`、`retrieve`、`answer` 和 `reset_all`。数据集解析与评测逻辑
放在 `src/memevo/datasets/`，通用客户端和运行工具放在 `src/memevo/utils/`。
