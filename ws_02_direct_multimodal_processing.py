"""
ws_02_direct_multimodal_processing.py
直接多模态内容处理示例 - 基于README.md示例2
"""
import asyncio
from lightrag import LightRAG
from raganything.modalprocessors import ImageModalProcessor, TableModalProcessor
from common_utils import get_llm_model_func, get_vision_model_func, get_embedding_func

async def main():
    print("=== 直接多模态内容处理示例 ===")
    
    # 获取模型函数
    llm_model_func = get_llm_model_func()
    vision_model_func = get_vision_model_func()
    embedding_func = get_embedding_func()
    
    # 初始化 LightRAG
    rag = LightRAG(
        working_dir="./rag_storage_02",
        llm_model_func=llm_model_func,
        embedding_func=embedding_func
    )
    await rag.initialize_storages()
    print("✅ LightRAG 初始化完成")

    # 处理图像内容
    print("\n--- 处理图像内容 ---")
    image_processor = ImageModalProcessor(
        lightrag=rag,
        modal_caption_func=vision_model_func
    )

    # 示例图像内容（请替换为实际图像路径）
    image_content = {
        "img_path": r"C:\Users\shuang_wang\Desktop\tree.png",  # 请替换为实际图像路径
        "img_caption": ["图1: 实验结果"],
        "img_footnote": ["数据收集于2024年"]
    }

    try:
        description, entity_info = await image_processor.process_multimodal_content(
            modal_content=image_content,
            content_type="image",
            file_path="research_paper.pdf",
            entity_name="实验结果图"
        )
        print(f"图像处理完成: {description[:100]}...")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"图像处理失败 (可能是路径不存在): {e}")

    # 处理表格内容
    print("\n--- 处理表格内容 ---")
    table_processor = TableModalProcessor(
        lightrag=rag,
        modal_caption_func=llm_model_func
    )

    table_content = {
        "table_body": """
        | 方法 | 准确率 | F1分数 |
        |--------|----------|----------|
        | RAGAnything | 95.2% | 0.94 |
        | 基准方法 | 87.3% | 0.85 |
        """,
        "table_caption": ["性能对比"],
        "table_footnote": ["测试数据集结果"]
    }

    try:
        description, entity_info = await table_processor.process_multimodal_content(
            modal_content=table_content,
            content_type="table",
            file_path="research_paper.pdf",
            entity_name="性能结果表"
        )
        print(f"表格处理完成: {description[:100]}...")
    except Exception as e:
        print(f"表格处理失败: {e}")

    # 查询处理后的内容
    print("\n--- 查询处理后的内容 ---")
    try:
        result = await rag.aquery(
            "刚才处理的表格和图像内容有什么关键信息？",
            # mode="hybrid"
        )
        print(f"查询结果: {result}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"查询失败: {e}")

    # 清理资源
    try:
        await rag.finalize_storages()
        print("✅ 资源清理完成")
    except Exception as e:
        print(f"资源清理失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
