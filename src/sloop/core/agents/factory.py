"""
Agent 工厂模块，负责根据配置动态加载和创建 Agent 实例。
"""

import importlib
from typing import Dict

import yaml
from openai import OpenAI

from sloop.core.config import SloopConfig
from sloop.core.generation.sloop import Sloop


def load_agent_config(config_path: str = None) -> Dict[str, str]:
    """
    从 YAML 配置文件中加载 Agent 的类名。
    如果未提供路径，则返回默认配置。

    Args:
        config_path (str, optional): YAML 配置文件的路径。

    Returns:
        Dict[str, str]: 包含 agent 类名的字典。
    """
    if config_path is None:
        # 返回内置的默认配置
        return {
            "user_agent": "SimpleUserAgent",
            "assistant_agent": "SimpleAssistantAgent",
            "service_agent": "SimpleServiceAgent",
            "planner": "SimplePlanner",
            "user_profile_agent": "SimpleUserProfileAgent",
            "api_sampler": "RandomAPISampler",
        }

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        return config.get("generation", {})


def create_agent(
    agent_type: str, config: SloopConfig, agent_class_name: str, **extra_args
) -> object:
    """
    根据类型和类名创建一个 Agent 实例。

    Args:
        agent_type (str): Agent 的类型，用于确定导入路径。
        config (SloopConfig): Sloop 配置对象。
        agent_class_name (str): 要创建的 Agent 类的名称。
        **extra_args: 传递给特定 Agent 构造函数的额外参数。

    Returns:
        object: 创建的 Agent 实例。

    Raises:
        ImportError: 如果无法导入指定的类。
        AttributeError: 如果模块中不存在指定的类。
    """
    # 定义类型到模块的映射
    module_map = {
        "user_agent": "sloop.core.agents.user_agent",
        "assistant_agent": "sloop.core.agents.assistant_agent",
        "service_agent": "sloop.core.agents.service_agent",
        "planner": "sloop.core.agents.planner",
        "user_profile_agent": "sloop.core.agents.user_profile_agent",
        "api_sampler": "sloop.core.generation.samplers.random_sampler",
    }

    module_name = module_map.get(agent_type)
    if not module_name:
        raise ValueError(f"未知的 Agent 类型: {agent_type}")

    # 动态导入模块
    module = importlib.import_module(module_name)
    agent_class = getattr(module, agent_class_name)

    # 根据类型创建实例
    if agent_type == "api_sampler":
        return agent_class()
    elif agent_type == "user_agent":
        # SimpleUserAgent 需要 client 和 user_profile_agent
        client = OpenAI(api_key=config.strong.api_key, base_url=config.strong.base_url)
        return agent_class(client, extra_args["user_profile_agent"])
    elif agent_type in [
        "assistant_agent",
        "service_agent",
        "planner",
        "user_profile_agent",
    ]:
        # 其他 agent 需要 client
        client = OpenAI(api_key=config.strong.api_key, base_url=config.strong.base_url)
        return agent_class(client)
    else:
        # 对于未知类型，尝试使用 config
        return agent_class(config)


def create_sloop_system(config: SloopConfig, agent_config_path: str = None):
    """
    根据配置创建一个完整的 Sloop 系统。

    Args:
        config (SloopConfig): Sloop 配置对象。
        agent_config_path (str, optional): Agent 配置文件的路径。

    Returns:
        Sloop: 一个已初始化的 Sloop 系统实例。
    """
    # 加载 Agent 配置
    agent_config = load_agent_config(agent_config_path)

    # 创建所有 Agent 实例
    user_profile_agent = create_agent(
        "user_profile_agent", config, agent_config["user_profile_agent"]
    )
    user_agent = create_agent(
        "user_agent",
        config,
        agent_config["user_agent"],
        user_profile_agent=user_profile_agent,
    )
    assistant_agent = create_agent(
        "assistant_agent", config, agent_config["assistant_agent"]
    )
    service_agent = create_agent("service_agent", config, agent_config["service_agent"])
    planner = create_agent("planner", config, agent_config["planner"])
    api_sampler = create_agent("api_sampler", config, agent_config["api_sampler"])

    # 创建并返回 Sloop 系统
    return Sloop(
        agents={
            "user_agent": user_agent,
            "assistant_agent": assistant_agent,
            "service_agent": service_agent,
            "planner": planner,
            "user_profile_agent": user_profile_agent,
            "api_sampler": api_sampler,
        },
        config=config,
    )
