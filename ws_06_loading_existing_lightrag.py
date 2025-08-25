"""
ws_06_loading_existing_lightrag.py
加载现有LightRAG实例示例 - 基于README.md示例6
"""
import asyncio
import os
from raganything import RAGAnything, RAGAnythingConfig
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from common_utils import get_llm_model_func, get_vision_model_func, get_embedding_func

async def main():
    print("=== 加载现有LightRAG实例示例 ===")
    
    # 获取模型函数
    llm_model_func = get_llm_model_func()
    vision_model_func = get_vision_model_func()
    embedding_func = get_embedding_func()
    
    # 设置LightRAG工作目录
    lightrag_working_dir = "./existing_lightrag_storage_06"

    # 检查是否存在之前的LightRAG实例
    if os.path.exists(lightrag_working_dir) and os.listdir(lightrag_working_dir):
        print("✅ 发现现有LightRAG实例，正在加载...")
    else:
        print("❌ 未发现现有LightRAG实例，将创建新实例")

    # 创建/加载LightRAG实例
    lightrag_instance = LightRAG(
        working_dir=lightrag_working_dir,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func
    )

    # 初始化存储（如果有现有数据会自动加载）
    await lightrag_instance.initialize_storages()
    await initialize_pipeline_status()
    print("✅ LightRAG存储初始化完成")

    # 使用现有LightRAG实例初始化RAGAnything
    rag = RAGAnything(
        lightrag=lightrag_instance,  # 传入现有LightRAG实例
        vision_model_func=vision_model_func,
        # 注意: working_dir, llm_model_func, embedding_func等从lightrag_instance继承
    )
    print("✅ RAGAnything初始化完成")

    # 查询现有知识库
    print("\n--- 查询现有知识库 ---")
    try:
        result = await rag.aquery(
            "这个LightRAG实例中已经处理了哪些数据？",
            mode="hybrid",
            enable_rerank=False,
            vlm_enhanced=False
        )
        print(f"现有数据查询结果: {result}")
    except Exception as e:
        print(f"查询现有数据失败: {e}")

    # 向现有LightRAG实例添加新的多模态文档
    print("\n--- 向现有实例添加新文档 ---")
    try:
        await rag.process_document_complete(
            file_path=r"C:\Users\shuang_wang\Downloads\Dummy-PDF-3Pages.pdf",
            output_dir="./output_06"
        )
        print("✅ 新文档添加完成")
    except Exception as e:
        print(f"添加新文档失败: {e}")

    # 再次查询以验证新内容
    print("\n--- 查询更新后的知识库 ---")
    try:
        updated_result = await rag.aquery(
            "现在知识库中包含哪些文档和内容？",
            mode="hybrid",
            enable_rerank=False,
            vlm_enhanced=False
        )
        print(f"更新后查询结果: {updated_result}")
    except Exception as e:
        print(f"查询更新后数据失败: {e}")

    # 演示直接使用LightRAG功能
    print("\n--- 直接使用LightRAG功能 ---")
    try:
        # 直接向LightRAG插入文本
        await lightrag_instance.ainsert("这是一个测试文本，用于演示直接LightRAG操作。")
        
        # 查询刚插入的内容
        direct_result = await lightrag_instance.aquery(
            "刚才插入的测试文本内容是什么？"
        )
        print(f"直接LightRAG查询结果: {direct_result}")
    except Exception as e:
        print(f"直接LightRAG操作失败: {e}")

    # 演示混合查询
    print("\n--- 演示混合查询 ---")
    try:
        mixed_result = await rag.aquery_with_multimodal(
            "结合现有知识库内容，分析这个新数据",
            multimodal_content=[{
                "type": "table",
                "table_data": """项目,状态,完成度
                                文档处理,完成,100%
                                知识库构建,进行中,80%
                                查询优化,计划中,0%""",
                "table_caption": "项目进度表"
            }],
            mode="hybrid",
            enable_rerank=False,
            vlm_enhanced=False
        )
        print(f"混合查询结果: {mixed_result}")
    except Exception as e:
        print(f"混合查询失败: {e}")

    # 显示知识库统计信息
    print("\n--- 知识库统计信息 ---")
    try:
        # 获取存储状态信息
        if hasattr(lightrag_instance, 'doc_status_storage'):
            print("文档状态存储已初始化")
        if hasattr(lightrag_instance, 'full_docs_storage'):
            print("完整文档存储已初始化")
        if hasattr(lightrag_instance, 'text_chunks_storage'):
            print("文本块存储已初始化")
        if hasattr(lightrag_instance, 'llm_response_cache'):
            print("LLM响应缓存已初始化")
        
        print(f"工作目录: {lightrag_instance.working_dir}")
    except Exception as e:
        print(f"获取统计信息失败: {e}")

    # 清理资源
    try:
        await rag.finalize_storages()
        print("\n✅ 资源清理完成")
        print(f"LightRAG数据已保存到: {lightrag_working_dir}")
    except Exception as e:
        print(f"资源清理失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
