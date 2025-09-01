import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from common_utils import get_llm_model_func, get_vision_model_func, get_embedding_func
from typing import Union
import sys
if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

moe = r'C:\Users\shuang_wang\Desktop\moe-1-5.pdf'

async def main():
    # 创建 RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",  # 选择解析器：mineru 或 docling
        parse_method="auto",  # 解析方法：auto, ocr 或 txt
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # 使用 common_utils 中的同步函数
    llm_model_func = get_llm_model_func()
    vision_model_func = get_vision_model_func()
    embedding_func = get_embedding_func()

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
        file_path=moe,
        output_dir="./output",
        parse_method="auto"
    )

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