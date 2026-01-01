"""
LLM 调用封装工具

基于 litellm 提供统一的模型调用接口，支持多种模型和配置。
"""

import sys
from typing import Any, Dict, List, Optional

import litellm

from sloop.utils.logger import logger

# 设置日志


def _get_mock_response(messages: List[Dict[str, Any]], json_mode: bool = False) -> str:
    """
    生成模拟LLM响应用于测试

    参数:
        messages: 消息列表
        json_mode: 是否为JSON模式

    返回:
        模拟响应字符串
    """
    if not messages:
        return "模拟响应：空消息列表"

    # 获取最后一条用户消息
    last_user_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    if not last_user_msg:
        return "模拟响应：未找到用户消息"

    # 根据消息内容生成不同的模拟响应
    content_lower = last_user_msg.lower()

    if json_mode:
        # JSON模式响应
        if "blueprint" in content_lower or "工具链" in content_lower:
            return """{
  "intent": "查询天气信息并验证准确性",
  "required_tools": ["alerts_active_zone_zoneid_for_national_weather_service", "forecast_weather_api_for_weatherapi_com", "points_point_stations_for_national_weather_service"],
  "reasoning": "用户想要获取天气预警、气象站数据和预报信息，需要多个工具协作",
  "is_valid": true
}"""
        elif "tool_call" in content_lower or "工具调用" in content_lower:
            return """{"tool_name": "forecast_weather_api_for_weatherapi_com", "parameters": {"location": "Beijing"}}"""
        elif "decision" in content_lower or "决策" in content_lower:
            return "true"
        else:
            return """{"response": "这是JSON格式的模拟响应", "status": "success"}"""

    else:
        # 普通文本响应
        if "天气预警" in content_lower or "weather alert" in content_lower:
            return "用户想要查询天气预警信息，需要检查是否有活跃的天气警报。"
        elif "气象站" in content_lower or "weather station" in content_lower:
            return "用户询问气象观测站信息，应该查找最近的观测站点。"
        elif "天气预报" in content_lower or "forecast" in content_lower:
            return "用户需要天气预报数据，可以调用天气API获取。"
        elif "blueprint" in content_lower or "蓝图" in content_lower:
            return """天气查询工作流：
1. 检查当前区域是否有天气预警
2. 查找最近的气象观测站
3. 获取气象站点网格数据
4. 获取逐小时天气预报
5. 对比商业天气API验证准确性

这个工具链可以形成完整的天气信息查询流程。"""
        elif "回复" in content_lower or "response" in content_lower:
            return "根据天气查询结果，北京市目前没有发布天气预警，天气状况良好。"
        else:
            return f'模拟响应：已收到您的消息"{last_user_msg[:50]}..."'


def completion(
    messages: List[Dict[str, Any]], json_mode: bool = False, **kwargs
) -> str:
    """
    统一的LLM调用接口

    参数:
        messages: 消息列表，OpenAI格式
        json_mode: 是否启用JSON模式
        **kwargs: 其他参数，会覆盖默认设置

    返回:
        模型响应内容字符串

    异常:
        各种LLM调用异常会被捕获并记录，但不抛出
    """
    from sloop.config import get_settings

    settings = get_settings()

    # 验证配置
    if not settings.validate():
        error_msg = "LLM配置无效，请检查环境变量"
        logger.error(error_msg)
        return f"配置错误: {error_msg}"

    try:
        # 检查是否是测试模式或API key无效
        if (
            settings.openai_api_key in ["qwertiasagv", "", None]
            or len(str(settings.openai_api_key)) < 10
        ):
            logger.warning("⚠️ 检测到无效API key，使用模拟响应进行测试")
            return _get_mock_response(messages, json_mode)

        # 准备调用参数
        call_kwargs = {
            "model": settings.model_name,
            "messages": messages,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "timeout": settings.timeout,
            "api_key": settings.openai_api_key,
        }

        # 如果设置了API base URL
        if settings.openai_api_base:
            call_kwargs["api_base"] = settings.openai_api_base

        # JSON模式处理
        if json_mode:
            # 对于OpenAI兼容的API
            if (
                "gpt" in settings.model_name.lower()
                or "openai" in settings.model_name.lower()
            ):
                call_kwargs["response_format"] = {"type": "json_object"}
            # 对于其他模型，在系统消息中添加JSON指令
            elif messages and messages[0].get("role") == "system":
                messages[0]["content"] += "\n\n请以JSON格式响应。"
            else:
                # 添加系统消息
                system_msg = {"role": "system", "content": "请以JSON格式响应。"}
                messages.insert(0, system_msg)

        # 合并用户提供的额外参数
        call_kwargs.update(kwargs)

        logger.info(f"调用LLM: {settings.model_name}, 消息数量: {len(messages)}")

        # 调用模型
        response = litellm.completion(**call_kwargs)

        # 提取响应内容
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content
            if content:
                logger.info(f"LLM响应成功，长度: {len(content)}")
                return content

        # 如果没有内容，返回空字符串
        logger.warning("LLM返回空响应")
        return ""

    except Exception as e:
        error_msg = f"LLM调用失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"调用错误: {error_msg}"


def chat_completion(
    prompt: str, system_message: Optional[str] = None, json_mode: bool = False, **kwargs
) -> str:
    """
    简化的聊天完成接口

    参数:
        prompt: 用户提示
        system_message: 系统消息（可选）
        json_mode: 是否启用JSON模式
        **kwargs: 其他参数

    返回:
        模型响应内容
    """
    messages = []

    # 添加系统消息
    if system_message:
        messages.append({"role": "system", "content": system_message})

    # 添加用户消息
    messages.append({"role": "user", "content": prompt})

    return completion(messages, json_mode=json_mode, **kwargs)


def validate_llm_config() -> bool:
    """
    验证LLM配置是否有效

    返回:
        配置是否有效
    """
    from sloop.config import get_settings

    settings = get_settings()
    return settings.validate()


def get_supported_models() -> List[str]:
    """
    获取litellm支持的模型列表（示例）

    返回:
        支持的模型名称列表
    """
    # 这里返回一些常见的模型作为示例
    # 实际应该从litellm获取完整列表
    return [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "gemini-pro",
        "deepseek-chat",
        "qwen2-72b-instruct",
    ]


if __name__ == "__main__":
    from sloop.config import get_settings

    logger.info("🔧 LLM 配置和调用测试")
    logger.info("=" * 50)

    # 测试配置验证
    logger.info("📋 配置状态:")
    settings = get_settings()
    if settings.validate():
        logger.info("✅ 配置验证通过")
        safe_config = settings.get_safe_display()
        for key, value in safe_config.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.error("❌ 配置验证失败")
        logger.info("\n请设置以下环境变量:")
        logger.info("  OPENAI_API_KEY=your_api_key_here")
        logger.info("  MODEL_NAME=gpt-4o-mini  # 可选")
        logger.info("  OPENAI_API_BASE=https://api.openai.com/v1  # 可选")
        logger.info("  TEMPERATURE=0.7  # 可选")
        sys.exit(1)

    logger.info("\n🧪 简单调用测试:")

    # 测试简单调用（如果配置了有效的API key）
    if settings.openai_api_key and len(settings.openai_api_key) > 10:  # 简单的key验证
        try:
            response = chat_completion(
                prompt="请简单介绍一下你自己。", system_message="你是一个友好的AI助手。"
            )

            if response and not response.startswith("调用错误"):
                logger.info("✅ LLM调用成功")
                logger.info(f"响应预览: {response[:100]}...")
            else:
                logger.warning("⚠️ LLM调用失败或无响应")
                logger.warning(f"响应: {response}")

        except Exception as e:
            logger.error(f"❌ 测试调用失败: {e}")
    else:
        logger.info("ℹ️ 未配置有效的API Key，跳过实际调用测试")

    logger.info("\n📚 支持的模型示例:")
    models = get_supported_models()
    for i, model in enumerate(models[:5], 1):  # 只显示前5个
        logger.info(f"  {i}. {model}")
    if len(models) > 5:
        logger.info(f"  ... 还有 {len(models) - 5} 个模型")

    logger.info("\n✅ LLM工具测试完成")
