"""
公共工具模块 - 封装LLM和embedding模型函数
用于所有demo测试文件的统一配置
"""
import asyncio
import requests
import numpy as np
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

# API配置
API_KEY = ''
BASE_URL = "http://10.25.20.246:6109/v1"
LLM_API_URL = "http://223.109.239.14:10000/v1/chat/completions"

def get_llm_model_func():
    """获取LLM模型函数（同步版本）"""
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        # 调用自托管聊天接口，构造标准OpenAI messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # 尝试将history_messages并入（若为list[dict]格式）
        if isinstance(history_messages, list):
            for msg in history_messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        # 当前用户问题
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": "qwen2__5-72b",
            "messages": messages,
            "stream": False,
            "temperature": 0,
            "top_p": 1,
        }
        headers = {"Content-Type": "application/json"}
        resp = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # 兼容常见返回格式
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            choice = data["choices"][0]
            # OpenAI-style
            if isinstance(choice, dict) and "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            # Some providers use 'text'
            if "text" in choice:
                return choice["text"]
        raise RuntimeError(f"Unexpected LLM response format: {data}")
    
    return llm_model_func

def get_vision_model_func():
    """获取视觉模型函数（同步版本）"""
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # 如果提供了messages格式（用于多模态VLM增强查询），直接使用
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=API_KEY,
                base_url=BASE_URL,
                **kwargs,
            )
        # 传统单图片格式
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=(
                    ([{"role": "system", "content": system_prompt}] if system_prompt else [])
                    + [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt}
                    ]
                ),
                api_key=API_KEY,
                base_url=BASE_URL,
                **kwargs,
            )
        # 纯文本格式
        else:
            return get_llm_model_func()(prompt, system_prompt, history_messages, **kwargs)
    
    return vision_model_func

def get_embedding_func():
    """获取嵌入函数（同步版本）"""
    def _http_embed(texts):
        api_url = BASE_URL.rstrip('/') + '/embeddings'
        payload = {"input": texts}
        headers = {"Content-Type": "application/json"}
        resp = requests.post(api_url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        vectors = []
        for item in data.get("data", []):
            if "embedding" in item:
                vectors.append(item["embedding"])
        if not vectors:
            raise RuntimeError(f"Embeddings response invalid: {data}")
        return np.array(vectors, dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=_http_embed,
    )

# 异步版本的函数
async def get_async_llm_model_func():
    """获取异步LLM模型函数"""
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        # 调用自托管聊天接口，构造标准OpenAI messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # 尝试将history_messages并入（若为list[dict]格式）
        if isinstance(history_messages, list):
            for msg in history_messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        # 当前用户问题
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "qwen2__5-72b",
            "messages": messages,
            "stream": False,
            "temperature": 0,
            "top_p": 1,
        }
        headers = {"Content-Type": "application/json"}

        def _call():
            resp = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # 兼容常见返回格式
            if isinstance(data, dict) and "choices" in data and data["choices"]:
                choice = data["choices"][0]
                # OpenAI-style
                if isinstance(choice, dict) and "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
                # Some providers use 'text'
                if "text" in choice:
                    return choice["text"]
            raise RuntimeError(f"Unexpected LLM response format: {data}")

        return await asyncio.to_thread(_call)
    
    return llm_model_func

async def get_async_vision_model_func():
    """获取异步视觉模型函数"""
    async def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # 如果提供了messages格式（用于多模态VLM增强查询），直接使用
        if messages:
            return await openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=API_KEY,
                base_url=BASE_URL,
                **kwargs,
            )
        # 传统单图片格式
        elif image_data:
            return await openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=(
                    ([{"role": "system", "content": system_prompt}] if system_prompt else [])
                    + [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt}
                    ]
                ),
                api_key=API_KEY,
                base_url=BASE_URL,
                **kwargs,
            )
        # 纯文本格式
        else:
            llm_func = await get_async_llm_model_func()
            return await llm_func(prompt, system_prompt, history_messages, **kwargs)
    
    return vision_model_func

async def get_async_embedding_func():
    """获取异步嵌入函数"""
    async def _http_embed(texts):
        api_url = BASE_URL.rstrip('/') + '/embeddings'
        payload = {"input": texts}
        headers = {"Content-Type": "application/json"}

        def _call():
            resp = requests.post(api_url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            vectors = []
            for item in data.get("data", []):
                if "embedding" in item:
                    vectors.append(item["embedding"])
            if not vectors:
                raise RuntimeError(f"Embeddings response invalid: {data}")
            return np.array(vectors, dtype=np.float32)

        return await asyncio.to_thread(_call)

    return EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=_http_embed,
    )
