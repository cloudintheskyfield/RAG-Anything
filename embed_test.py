#!/usr/bin/env python3
"""
测试自部署的embedding模型
"""
import asyncio
import json
import requests
import numpy as np
from typing import List

async def test_embed_api(texts: List[str], base_url: str):
    """测试自部署的embedding API（仅使用 input，不传 model）"""
    
    # 构建请求URL - 去掉路径部分，只保留根URL
    if "/v1/embeddings" in base_url:
        api_url = base_url
    else:
        api_url = base_url.rstrip('/') + '/embeddings'
    
    print(f"测试URL: {api_url}")
    print(f"测试文本: {texts}")
    
    # 构建请求数据（仅 input，不传 model）
    data = {
        "input": texts
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # 发送请求
        print("发送embedding请求...")
        response = requests.post(api_url, json=data, headers=headers, timeout=30)
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # 检查响应格式
            if "data" in result:
                embeddings = []
                for item in result["data"]:
                    if "embedding" in item:
                        embeddings.append(item["embedding"])
                
                if embeddings:
                    embeddings_array = np.array(embeddings)
                    print(f"✓ 成功获取embeddings")
                    print(f"  - 文本数量: {len(texts)}")
                    print(f"  - 向量维度: {embeddings_array.shape[1]}")
                    print(f"  - 向量形状: {embeddings_array.shape}")
                    print(f"  - 第一个向量前5维: {embeddings_array[0][:5]}")
                    
                    return embeddings_array
                else:
                    print("✗ 响应中没有找到embedding数据")
            else:
                print("✗ 响应格式不正确，缺少data字段")
                print(f"响应内容: {result}")
        else:
            print(f"✗ API请求失败")
            print(f"响应内容: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ 网络请求错误: {e}")
    except json.JSONDecodeError as e:
        print(f"✗ JSON解析错误: {e}")
    except Exception as e:
        print(f"✗ 其他错误: {e}")
    
    return None

# 已移除 lightrag 的二次封装测试（该服务无需 model 字段）

async def test_chat_api(question: str, api_url: str, model: str = "qwen2__5-72b"):
    """测试自托管 Chat Completions 接口可用性"""
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": question}
        ],
        "stream": False,
        "temperature": 0,
        "top_p": 1
    }
    headers = {"Content-Type": "application/json"}
    print(f"\n测试Chat模型URL: {api_url}")
    print(f"问题: {question}")
    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=30)
        print(f"响应状态码: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            # OpenAI 风格解析
            if isinstance(data, dict) and "choices" in data and data["choices"]:
                choice = data["choices"][0]
                if isinstance(choice, dict) and "message" in choice and "content" in choice["message"]:
                    answer = choice["message"]["content"]
                    print("✓ 成功获取回答:")
                    print(answer)
                    return answer
                if "text" in choice:
                    answer = choice["text"]
                    print("✓ 成功获取文本回答:")
                    print(answer)
                    return answer
            print("✗ 响应格式不符合预期:", data)
        else:
            print("✗ Chat API请求失败")
            print(f"响应内容: {resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"✗ 网络请求错误: {e}")
    except json.JSONDecodeError as e:
        print(f"✗ JSON解析错误: {e}")
    except Exception as e:
        print(f"✗ 其他错误: {e}")
    return None

async def main():
    print("=" * 60)
    print("自部署Embedding模型测试")
    print("=" * 60)
    
    texts = ["Hello, world", "人工智能正在快速发展"]
    base_url = "http://10.25.20.246:6109/v1/embeddings"
    chat_url = "http://223.109.239.14:10000/v1/chat/completions"
    
    # 测试1: 直接API调用
    print("\n1. 直接API测试:")
    await test_embed_api(texts, base_url)
    
    # 测试2: Chat模型连通性
    print("\n2. Chat模型连通性测试:")
    await test_chat_api("3.19和3.8哪个大", chat_url)
    
    # 跳过lightrag测试（避免隐式传递 model 字段到 embeddings）
    
    print("\n=" * 60)
    print("测试完成")

if __name__ == "__main__":
    asyncio.run(main())
