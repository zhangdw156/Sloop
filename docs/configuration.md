# 配置说明

本文档说明 Sloop 使用的环境变量及命名约定。

## 环境变量

复制 `.env.example` 为 `.env` 后，配置以下变量：

| 变量名 | 说明 |
|--------|------|
| `MODEL_NAME` | 主对话模型名称（Assistant / UserProxy / Simulator / Intent 等） |
| `MODEL_API_KEY` | 主对话模型 API Key |
| `MODEL_BASE_URL` | 主对话模型 API Base URL |
| `EMBEDDING_MODEL_NAME` | 向量模型名称（GraphBuilder 使用） |
| `EMBEDDING_MODEL_API_KEY` | 向量模型 API Key |
| `EMBEDDING_MODEL_BASE_URL` | 向量模型 API Base URL |

## 迁移说明

主模型相关变量已从 `OPENAI_MODEL_*` 统一为 `MODEL_*`：

- `OPENAI_MODEL_NAME` → `MODEL_NAME`
- `OPENAI_MODEL_API_KEY` → `MODEL_API_KEY`
- `OPENAI_MODEL_BASE_URL` → `MODEL_BASE_URL`

若仍使用旧变量名，请更新为上述新命名后再运行。
