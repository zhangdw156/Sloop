# Sloop Agent Prompts Configuration

这是Sloop项目的集中提示词管理系统，所有Agent的提示词都通过YAML配置文件进行管理和版本控制。

## 目录结构

```
src/sloop/configs/
├── agent_prompts.yaml    # 主要配置文件
└── README.md            # 说明文档
```

## 配置格式

### 1. Agent配置

每个Agent包含以下字段：
- `role`: Agent的角色名称
- `goal`: Agent的目标描述
- `backstory`: Agent的背景故事和行为指导

### 2. Task配置

每个Task包含以下字段：
- `description`: Task的详细描述
- `expected_output`: 期望的输出格式

### 3. 用户画像配置

用户画像包含以下字段：
- `name`: 用户类型名称
- `personality`: 性格特征
- `communication_style`: 沟通风格
- `error_handling`: 错误处理方式
- `interaction_pattern`: 交互模式
- `typical_behavior`: 典型行为

## 模板变量

配置文件支持模板变量替换，变量使用`${variable_name}`语法：

### Agent模板变量
- `${user_type}`: 用户类型
- `${user_profile}`: 用户画像JSON
- `${communication_style}`: 沟通风格
- `${error_handling}`: 错误处理方式
- `${interaction_pattern}`: 交互模式

### Task模板变量
- `${conversation_history}`: 对话历史
- `${available_apis}`: 可用API列表
- `${tool_result}`: 工具执行结果
- `${user_name}`: 用户姓名
- `${user_occupation}`: 用户职业
- `${user_age}`: 用户年龄
- `${user_personality}`: 用户性格
- `${user_communication_style}`: 用户沟通风格
- `${user_behavior}`: 用户行为

## 使用方法

### 1. 修改提示词

直接编辑 `agent_prompts.yaml` 文件，修改相应的Agent或Task配置。

### 2. 添加新Agent

在 `agents` 部分添加新的Agent配置：

```yaml
agents:
  new_agent:
    role: "新Agent角色"
    goal: "新Agent目标"
    backstory: |
      新Agent的背景故事和行为指导
```

### 3. 添加新Task

在 `tasks` 部分添加新的Task配置：

```yaml
tasks:
  new_task:
    description: |
      新Task的详细描述
    expected_output: "期望的输出格式"
```

### 4. 版本控制

提示词配置支持Git版本控制，可以：
- 查看修改历史
- 回滚到之前的版本
- 分支管理不同版本的提示词

## 最佳实践

### 1. 版本管理
- 每次修改提示词时更新 `version` 字段
- 在提交信息中说明修改内容
- 定期review和优化提示词

### 2. 模板使用
- 合理使用模板变量，避免硬编码
- 为不同场景准备不同的提示词模板
- 测试模板变量的替换效果

### 3. 文档维护
- 更新 `description` 字段说明配置用途
- 为复杂的提示词添加注释
- 维护变更日志

### 4. 性能优化
- 避免过长的提示词影响响应速度
- 合理使用上下文窗口
- 定期清理不再使用的配置

## 示例

查看 `agent_prompts.yaml` 文件中的完整示例配置。

## 故障排除

### 1. 配置文件不存在
确保 `agent_prompts.yaml` 文件位于正确的路径：`src/sloop/configs/agent_prompts.yaml`

### 2. 模板变量错误
检查模板变量是否正确定义，使用 `${variable_name}` 语法。

### 3. YAML格式错误
使用YAML验证工具检查配置文件格式是否正确。

### 4. 编码问题
确保文件使用UTF-8编码保存。
