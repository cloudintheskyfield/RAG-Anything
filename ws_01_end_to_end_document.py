import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
import numpy as np
import requests

async def main():
    # 设置 API 配置
    # api_key = "your-api-key"
    api_key = ''
    base_url = "http://10.25.20.246:6109/v1"  # 修正为根路径
    llm_api_url = "http://223.109.239.14:10000/v1/chat/completions"

    # 创建 RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",  # 选择解析器：mineru 或 docling
        parse_method="auto",  # 解析方法：auto, ocr 或 txt
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # 定义 LLM 模型函数
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
        resp = requests.post(llm_api_url, json=payload, headers=headers, timeout=60)
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

    # 定义视觉模型函数用于图像处理
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
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # 传统单图片格式
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
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
                    else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # 纯文本格式
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # 定义嵌入函数（自建服务：仅 input，不传 model）
    def _http_embed(texts):
        api_url = base_url.rstrip('/') + '/embeddings'
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

    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=_http_embed,
    )

    # 初始化 RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # 处理文档 - 请替换为你的实际文档路径
    await rag.process_document_complete(
        # file_path=r"C:\Users\shuang_wang\Desktop\qwen_2_5_omini.pdf",
        file_path=r"C:\Users\shuang_wang\Downloads\Dummy-PDF-3Pages.pdf",
        output_dir="./output",
        parse_method="auto"
    )
    print("请先设置正确的文档路径，然后取消注释上面的代码")

    # 查询处理后的内容
    # 纯文本查询 - 基本知识库搜索
    text_result = await rag.aquery(
        "文档的主要内容是什么？",
        mode="hybrid"
    )
    print("文本查询结果:", text_result)

    # 多模态查询 - 包含具体多模态内容的查询
    multimodal_result = await rag.aquery_with_multimodal(
        "分析这个性能数据并解释与现有文档内容的关系",
        multimodal_content=[{
            "type": "table",
            "table_data": """系统,准确率,F1分数
                            RAGAnything,95.2%,0.94
                            基准方法,87.3%,0.85""",
            "table_caption": "性能对比结果"
        }],
        mode="hybrid"
    )
    print("多模态查询结果:", multimodal_result)

if __name__ == "__main__":
    asyncio.run(main())