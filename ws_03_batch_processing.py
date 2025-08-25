"""
ws_03_batch_processing.py
批量处理示例 - 基于README.md示例3
"""
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from common_utils import get_async_llm_model_func, get_async_vision_model_func, get_async_embedding_func

async def main():
    print("=== 批量处理示例 ===")
    
    # 获取异步模型函数
    llm_model_func = await get_async_llm_model_func()
    vision_model_func = await get_async_vision_model_func()
    embedding_func = await get_async_embedding_func()
    
    # 创建 RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir="./rag_storage_03",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # 初始化 RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )
    print("✅ RAGAnything 初始化完成")

    # 批量处理多个文档
    print("\n--- 批量处理文档 ---")
    try:
        await rag.process_folder_complete(
            folder_path="./documents",  # 请确保此目录存在并包含文档
            output_dir="./output_03",
            file_extensions=[".pdf", ".docx", ".pptx"],
            recursive=True,
            max_workers=4
        )
        print("✅ 批量处理完成")
    except Exception as e:
        print(f"批量处理失败: {e}")
        print("提示: 请确保 ./documents 目录存在并包含支持的文档格式")
        
        # 如果批量处理失败，尝试处理单个文档作为演示
        print("\n--- 尝试处理单个文档作为演示 ---")
        try:
            await rag.process_document_complete(
                file_path=r"C:\Users\shuang_wang\Downloads\Dummy-PDF-3Pages.pdf",
                output_dir="./output_03",
                parse_method="auto"
            )
            print("✅ 单个文档处理完成")
        except Exception as e2:
            print(f"单个文档处理也失败: {e2}")

    # 查询处理后的内容
    print("\n--- 查询处理后的内容 ---")
    try:
        result = await rag.aquery(
            "处理的文档中有哪些主要内容和关键信息？",
            mode="hybrid",
            enable_rerank=False,
            vlm_enhanced=False
        )
        print(f"查询结果: {result}")
    except Exception as e:
        print(f"查询失败: {e}")

    # 清理资源
    try:
        await rag.finalize_storages()
        print("✅ 资源清理完成")
    except Exception as e:
        print(f"资源清理失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
