# FILE: sloop/services/llm.py
from typing import Dict, List, Optional

from openai import APIError, OpenAI

from sloop.configs import env_config
from sloop.utils.logger import logger


class LLMService:
    def __init__(self):
        self.base_url = env_config.get("OPENAI_MODEL_BASE_URL")
        self.api_key = env_config.get("OPENAI_MODEL_API_KEY")
        self.model_name = env_config.get(
            "OPENAI_MODEL_NAME", "Qwen3-235B-A22B-Instruct-2507"
        )

        if not self.base_url or not self.api_key:
            logger.warning(
                "Missing LLM configuration in .env. LLM features will be disabled."
            )
            self.client = None
        else:
            try:
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                )
                logger.info(f"LLM Service initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        response_format: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[str]:
        """
        发送聊天请求并返回内容字符串
        """
        if not self.client:
            logger.error("LLM client is not initialized.")
            return None

        try:
            # 构造参数，过滤掉 None 的值
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                **kwargs,
            }
            if response_format:
                params["response_format"] = response_format

            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content

        except APIError as e:
            logger.error(f"LLM API Error: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM Call Failed: {e}")
            return None
