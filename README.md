# MemEvo

用于评测对话记忆系统的轻量实验框架。目前支持 LoCoMo、Full Context 和
Mem0，并统一记录模型调用与 token 用量。

## 使用

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync
cp .env.example .env
cp src/configs/full_context.example.toml config.toml
```

将 LoCoMo 数据放到 `data/locomo10.json`，然后配置 `.env` 和
`config.toml`：

```bash
uv run memevo --config config.toml
```

运行 Mem0：

```bash
cp src/configs/mem0.example.toml config.toml
uv run memevo --config config.toml
```

## 输出

结果保存在 `[run] output_dir`：

- `answers.json`：模型回答
- `evaluation.json`：评测结果
- `usage.json`：调用次数和 token 用量
- `memory/`：算法记忆数据

## 扩展

算法放在 `src/memevo/algorithms/<name>/`，并提供 `create()` 以及
`ingest`、`retrieve`、`answer`、`reset_all`、`close` 方法。

数据集放在 `src/memevo/datasets/<name>.py` 并提供 `create()`。模型调用使用
`memevo.utils.models.LLM` 或 `Embedder`，即可自动统计 token。

## 开发

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
