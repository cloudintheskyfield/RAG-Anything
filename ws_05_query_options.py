"""
ws_05_query_options.py
查询选项示例 - 基于README.md示例5
演示不同的查询模式和选项
"""
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from common_utils import get_async_llm_model_func, get_async_vision_model_func, get_async_embedding_func

async def main():
    print("=== 查询选项示例 ===")
    
    # 获取异步模型函数
    llm_model_func = await get_async_llm_model_func()
    vision_model_func = await get_async_vision_model_func()
    embedding_func = await get_async_embedding_func()
    
    # 创建 RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir="./rag_storage_05",
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

    # 先处理一个文档以便查询
    print("\n--- 处理文档 ---")
    try:
        await rag.process_document_complete(
            file_path=r"C:\Users\shuang_wang\Downloads\Dummy-PDF-3Pages.pdf",
            output_dir="./output_05",
            parse_method="auto"
        )
        print("✅ 文档处理完成")
    except Exception as e:
        print(f"文档处理失败: {e}")
        return

    # 1. 纯文本查询 - 不同模式
    print("\n=== 1. 纯文本查询 - 不同模式 ===")
    
    query_text = "文档的主要内容是什么？"
    
    modes = ["hybrid", "local", "global", "naive"]
    for mode in modes:
        try:
            print(f"\n--- {mode.upper()} 模式查询 ---")
            result = await rag.aquery(
                query_text, 
                mode=mode,
                enable_rerank=False,
                vlm_enhanced=False
            )
            print(f"{mode} 查询结果: {result[:200]}...")
        except Exception as e:
            print(f"{mode} 查询失败: {e}")

    # 2. VLM增强查询
    print("\n=== 2. VLM增强查询 ===")
    
    try:
        print("\n--- VLM增强启用 ---")
        vlm_result = await rag.aquery(
            "分析文档中的图表和图像内容",
            mode="hybrid",
            vlm_enhanced=True,
            enable_rerank=False
        )
        print(f"VLM增强查询结果: {vlm_result[:200]}...")
    except Exception as e:
        print(f"VLM增强查询失败: {e}")

    try:
        print("\n--- VLM增强禁用 ---")
        no_vlm_result = await rag.aquery(
            "分析文档中的图表和图像内容",
            mode="hybrid",
            vlm_enhanced=False,
            enable_rerank=False
        )
        print(f"非VLM查询结果: {no_vlm_result[:200]}...")
    except Exception as e:
        print(f"非VLM查询失败: {e}")

    # 3. 多模态查询 - 表格数据
    print("\n=== 3. 多模态查询 - 表格数据 ===")
    
    try:
        table_result = await rag.aquery_with_multimodal(
            "比较这些性能指标与文档内容",
            multimodal_content=[{
                "type": "table",
                "table_data": """方法,准确率,速度
                                RAGAnything,95.2%,120ms
                                传统方法,87.3%,180ms""",
                "table_caption": "性能对比"
            }],
            mode="hybrid",
            enable_rerank=False,
            vlm_enhanced=False
        )
        print(f"表格查询结果: {table_result[:200]}...")
    except Exception as e:
        print(f"表格查询失败: {e}")

    # 4. 多模态查询 - 公式内容
    print("\n=== 4. 多模态查询 - 公式内容 ===")
    
    try:
        equation_result = await rag.aquery_with_multimodal(
            "解释这个公式及其与文档内容的相关性",
            multimodal_content=[{
                "type": "equation",
                "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
                "equation_caption": "文档相关性概率"
            }],
            mode="hybrid",
            enable_rerank=False,
            vlm_enhanced=False
        )
        print(f"公式查询结果: {equation_result[:200]}...")
    except Exception as e:
        print(f"公式查询失败: {e}")

    # 5. 同步查询示例
    print("\n=== 5. 同步查询示例 ===")
    
    try:
        sync_result = rag.query(
            "使用同步方式查询文档内容", 
            mode="hybrid"
        )
        print(f"同步查询结果: {sync_result[:200]}...")
    except Exception as e:
        print(f"同步查询失败: {e}")

    # 6. 复杂多模态查询
    print("\n=== 6. 复杂多模态查询 ===")
    
    try:
        complex_result = await rag.aquery_with_multimodal(
            "综合分析以下多种类型的数据",
            multimodal_content=[
                {
                    "type": "table",
                    "table_data": """指标,数值,单位
                                    准确率,95.2,%
                                    召回率,92.8,%
                                    F1分数,0.94,无量纲""",
                    "table_caption": "模型性能指标"
                },
                {
                    "type": "equation", 
                    "latex": "F1 = 2 \\cdot \\frac{precision \\cdot recall}{precision + recall}",
                    "equation_caption": "F1分数计算公式"
                }
            ],
            mode="hybrid",
            enable_rerank=False,
            vlm_enhanced=False
        )
        print(f"复杂查询结果: {complex_result[:200]}...")
    except Exception as e:
        print(f"复杂查询失败: {e}")

    # 清理资源
    try:
        await rag.finalize_storages()
        print("\n✅ 资源清理完成")
    except Exception as e:
        print(f"资源清理失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
