
import requests

from sloop.configs.env import env_config


def test_embedding_model_config():
    """验证embedding model配置的正确性和可用性"""

    # 1. 验证环境变量是否存在
    model_name = env_config.get("EMBEDDING_MODEL_NAME")
    model_base_url = env_config.get("EMBEDDING_MODEL_BASE_URL")
    model_api_key = env_config.get("EMBEDDING_MODEL_API_KEY")

    assert model_name is not None, "EMBEDDING_MODEL_NAME 环境变量未设置"
    assert model_base_url is not None, "EMBEDDING_MODEL_BASE_URL 环境变量未设置"
    assert model_api_key is not None, "EMBEDDING_MODEL_API_KEY 环境变量未设置"

    print("✓ 环境变量检查通过")
    print(f"  - EMBEDDING_MODEL_NAME: {model_name}")
    print(f"  - EMBEDDING_MODEL_BASE_URL: {model_base_url}")
    print(f"  - EMBEDDING_MODEL_API_KEY: {'*' * len(model_api_key) if model_api_key else 'None'}")

    # 2. 验证URL格式
    assert model_base_url.startswith(("http://", "https://")), "EMBEDDING_MODEL_BASE_URL 必须以 http:// 或 https:// 开头"

    # 3. 构建测试请求
    test_url = f"{model_base_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {model_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "input": ["这是一个测试句子，用于验证embedding服务是否正常工作。"]
    }

    print("\n✓ 请求配置完成")
    print(f"  - 测试URL: {test_url}")
    print(f"  - 请求头: {headers.keys()}")
    print(f"  - 请求体: {payload}")

    # 4. 发送测试请求
    try:
        response = requests.post(test_url, json=payload, headers=headers, timeout=30)
        print("\n✓ HTTP请求发送成功")
        print(f"  - 状态码: {response.status_code}")
        print(f"  - 响应时间: {response.elapsed.total_seconds():.2f}秒")

        # 5. 验证响应
        assert response.status_code == 200, f"HTTP请求失败，状态码: {response.status_code}"

        response_data = response.json()
        print(f"  - 响应内容: {list(response_data.keys())}")

        # 6. 验证响应数据结构
        assert "data" in response_data, "响应中缺少'data'字段"
        assert len(response_data["data"]) > 0, "响应数据为空"
        assert "embedding" in response_data["data"][0], "响应中缺少'embedding'字段"
        assert isinstance(response_data["data"][0]["embedding"], list), "'embedding'字段必须是列表"
        assert len(response_data["data"][0]["embedding"]) > 0, "'embedding'列表不能为空"

        print("\n✓ 响应验证通过")
        print(f"  - embedding维度: {len(response_data['data'][0]['embedding'])}")

        # 7. 验证模型信息
        if "model" in response_data:
            assert response_data["model"] == model_name, f"返回的模型名称不匹配: {response_data['model']} != {model_name}"
            print("  - 模型名称验证: ✓")

        print("\n✅ 所有验证通过！Embedding模型配置正确且服务可用。")

    except requests.exceptions.Timeout:
        raise AssertionError("请求超时：Embedding服务响应时间过长")
    except requests.exceptions.ConnectionError:
        raise AssertionError("连接错误：无法连接到Embedding服务，请检查URL和网络连接")
    except requests.exceptions.RequestException as e:
        raise AssertionError(f"请求异常：{str(e)}")
    except Exception as e:
        raise AssertionError(f"验证失败：{str(e)}")

if __name__ == "__main__":
    test_embedding_model_config()
