"""
提示词集中管理器
负责加载和管理所有Agent的提示词配置
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from string import Template


class PromptManager:
    """
    提示词集中管理器
    从YAML配置文件加载和管理所有Agent提示词
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化提示词管理器

        Args:
            config_path: 配置文件路径，默认使用configs/agent_prompts.yaml
        """
        if config_path is None:
            # 从当前包目录查找configs目录
            package_root = Path(__file__).parent.parent
            config_path = package_root / "configs" / "agent_prompts.yaml"

        self.config_path = Path(config_path)
        self._config = None
        self._templates = {}

        self._load_config()

    def _load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"提示词配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        # 预编译模板
        self._compile_templates()

    def _compile_templates(self):
        """预编译所有模板"""
        for agent_name, agent_config in self._config.get('agents', {}).items():
            self._templates[agent_name] = {}
            for key, value in agent_config.items():
                if isinstance(value, str):
                    self._templates[agent_name][key] = Template(value)

        for task_name, task_config in self._config.get('tasks', {}).items():
            self._templates[task_name] = {}
            for key, value in task_config.items():
                if isinstance(value, str):
                    self._templates[task_name][key] = Template(value)

    def get_agent_config(self, agent_name: str, **kwargs) -> Dict[str, Any]:
        """
        获取Agent配置

        Args:
            agent_name: Agent名称
            **kwargs: 模板变量

        Returns:
            Agent配置字典
        """
        if agent_name not in self._config.get('agents', {}):
            raise ValueError(f"未知的Agent: {agent_name}")

        agent_config = self._config['agents'][agent_name].copy()

        # 应用模板替换
        if agent_name in self._templates:
            for key, template in self._templates[agent_name].items():
                try:
                    agent_config[key] = template.substitute(**kwargs)
                except KeyError as e:
                    # 如果缺少必需的变量，使用原始值
                    pass

        return agent_config

    def get_task_config(self, task_name: str, **kwargs) -> Dict[str, Any]:
        """
        获取Task配置

        Args:
            task_name: Task名称
            **kwargs: 模板变量

        Returns:
            Task配置字典
        """
        if task_name not in self._config.get('tasks', {}):
            raise ValueError(f"未知的Task: {task_name}")

        task_config = self._config['tasks'][task_name].copy()

        # 应用模板替换
        if task_name in self._templates:
            for key, template in self._templates[task_name].items():
                try:
                    task_config[key] = template.substitute(**kwargs)
                except KeyError as e:
                    # 如果缺少必需的变量，使用原始值
                    pass

        return task_config

    def get_user_profile(self, profile_type: str) -> Dict[str, Any]:
        """
        获取用户画像配置

        Args:
            profile_type: 用户画像类型

        Returns:
            用户画像配置
        """
        profiles = self._config.get('user_profiles', {})
        if profile_type not in profiles:
            # 返回默认配置
            return {
                "name": "普通用户",
                "personality": "普通性格",
                "communication_style": "口语化",
                "error_handling": "一般",
                "interaction_pattern": "正常交流",
                "typical_behavior": "正常交流"
            }

        return profiles[profile_type]

    def get_variable(self, var_name: str) -> Any:
        """
        获取全局变量

        Args:
            var_name: 变量名

        Returns:
            变量值
        """
        return self._config.get('variables', {}).get(var_name)

    def reload_config(self):
        """重新加载配置文件"""
        self._load_config()

    @property
    def version(self) -> str:
        """获取配置版本"""
        return self._config.get('version', 'unknown')

    @property
    def description(self) -> str:
        """获取配置描述"""
        return self._config.get('description', '')


# 全局提示词管理器实例
prompt_manager = PromptManager()
