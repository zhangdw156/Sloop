# 数据质量约束说明

当前版本在生成与落盘阶段采用以下约束，以提升 SFT 数据质量。

## User Proxy

- **可见性边界**：仅使用对话中用户可见的信息，不引用隐藏的 tool 输出、ID 或内部状态。
- **禁止伪造**：缺参数时明确说明“不知道”或请求替代方案，不编造 tool 返回值或环境数据。

## Simulator

- **返回契约**：统一使用 JSON 结构：
  - 成功：`{"status":"ok","data":{...}}`
  - 失败：`{"status":"error","error":{"code":"...","message":"..."}}`
- **一致性**：同一实体（如 ID、名称）在多轮调用中保持一致。

## 轨迹拓扑校验

在 `exps/verify_agents.py` 写入轨迹前，会调用 `validate_openai_tool_topology` 进行校验：

- 每条消息的 `role` 必须在 `system` / `user` / `assistant` / `tool` 之中。
- `assistant` 消息若包含 `tool_calls`，其后必须有对应 `tool` 消息且带有 `tool_call_id`。
- 不通过校验的样本会被跳过，不写入输出文件，并在日志中记录原因。

## Assistant 协议

- 缺必填参数时先向用户追问，不发起工具调用。
- 工具报错时解释并重试或追问。
- 最终回复前核验是否满足用户目标。
