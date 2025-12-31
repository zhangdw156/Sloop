"""
LLM 调用封装工具

基于 litellm 提供统一的模型调用接口，支持多种模型和配置。
"""

import logging
from typing import List, Dict, Any, Optional, Union
import litellm
from sloop.config import get_settings

# 设置日志
logger = logging.getLogger(__name__)


def completion(
    messages: List[Dict[str, Any]],
    json_mode: bool = False,
    **kwargs
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
    settings = get_settings()

    # 验证配置
    if not settings.validate():
        error_msg = "LLM配置无效，请检查环境变量"
        logger.error(error_msg)
        return f"配置错误: {error_msg}"

    try:
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
            if "gpt" in settings.model_name.lower() or "openai" in settings.model_name.lower():
                call_kwargs["response_format"] = {"type": "json_object"}
            else:
                # 对于其他模型，在系统消息中添加JSON指令
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] += "\n\n请以JSON格式响应。"
                else:
                    # 添加系统消息
                    system_msg = {
                        "role": "system",
                        "content": "请以JSON格式响应。"
                    }
                    messages.insert(0, system_msg)

        # 合并用户提供的额外参数
        call_kwargs.update(kwargs)

        logger.info(f"调用LLM: {settings.model_name}, 消息数量: {len(messages)}")

        # 调用模型
        response = litellm.completion(**call_kwargs)

        # 提取响应内容
        if hasattr(response, 'choices') and response.choices:
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
    prompt: str,
    system_message: Optional[str] = None,
    json_mode: bool = False,
    **kwargs
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
        messages.append({
            "role": "system",
            "content": system_message
        })

    # 添加用户消息
    messages.append({
        "role": "user",
        "content": prompt
    })

    return completion(messages, json_mode=json_mode, **kwargs)


def validate_llm_config() -> bool:
    """
    验证LLM配置是否有效

    返回:
        配置是否有效
    """
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
        "qwen2-72b-instruct"
    ]


if __name__ == "__main__":
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
        exit(1)

    logger.info("\n🧪 简单调用测试:")

    # 测试简单调用（如果配置了有效的API key）
    if settings.openai_api_key and len(settings.openai_api_key) > 10:  # 简单的key验证
        try:
            response = chat_completion(
                prompt="请简单介绍一下你自己。",
                system_message="你是一个友好的AI助手。"
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
