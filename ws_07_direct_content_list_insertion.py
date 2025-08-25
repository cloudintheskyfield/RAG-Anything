"""
ws_07_direct_content_list_insertion.py
直接内容列表插入示例 - 基于README.md示例7
"""
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from common_utils import get_async_llm_model_func, get_async_vision_model_func, get_async_embedding_func

async def main():
    print("=== 直接内容列表插入示例 ===")
    
    # 获取异步模型函数
    llm_model_func = await get_async_llm_model_func()
    vision_model_func = await get_async_vision_model_func()
    embedding_func = await get_async_embedding_func()
    
    # 创建 RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir="./rag_storage_07",
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

    # 示例：来自外部源的预解析内容列表
    print("\n--- 准备预解析内容列表 ---")
    content_list = [
        {
            "type": "text",
            "text": "这是我们研究论文的引言部分。",
            "page_idx": 0  # 内容出现的页码
        },
        {
            "type": "image",
            "img_path": r"C:\Users\shuang_wang\Downloads\sample_figure.jpg",  # 重要：使用绝对路径
            "img_caption": ["图1: 系统架构"],
            "img_footnote": ["来源: 作者原创设计"],
            "page_idx": 1  # 图像出现的页码
        },
        {
            "type": "table",
            "table_body": "| 方法 | 准确率 | F1分数 |\n|--------|----------|----------|\n| 我们的方法 | 95.2% | 0.94 |\n| 基准方法 | 87.3% | 0.85 |",
            "table_caption": ["表1: 性能对比"],
            "table_footnote": ["测试数据集结果"],
            "page_idx": 2  # 表格出现的页码
        },
        {
            "type": "equation",
            "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
            "text": "文档相关性概率公式",
            "page_idx": 3  # 公式出现的页码
        },
        {
            "type": "text",
            "text": "总之，我们的方法在所有指标上都表现出优越的性能。",
            "page_idx": 4  # 内容出现的页码
        }
    ]

    # 直接插入内容列表
    print("\n--- 插入内容列表 ---")
    try:
        await rag.insert_content_list(
            content_list=content_list,
            file_path="research_paper.pdf",  # 引用文件名
            split_by_character=None,         # 可选的文本分割
            split_by_character_only=False,   # 可选的文本分割模式
            doc_id=None,                     # 可选的自定义文档ID（如果不提供会自动生成）
            display_stats=True               # 显示内容统计
        )
        print("✅ 内容列表插入完成")
    except Exception as e:
        print(f"内容列表插入失败: {e}")

    # 查询插入的内容
    print("\n--- 查询插入的内容 ---")
    try:
        result = await rag.aquery(
            "研究中提到的关键发现和性能指标是什么？",
            mode="hybrid",
            enable_rerank=False,
            vlm_enhanced=False
        )
        print(f"查询结果: {result}")
    except Exception as e:
        print(f"查询失败: {e}")

    # 插入另一个内容列表，使用不同的文档ID
    print("\n--- 插入另一个文档的内容列表 ---")
    another_content_list = [
        {
            "type": "text",
            "text": "这是来自另一个文档的内容。",
            "page_idx": 0
        },
        {
            "type": "table",
            "table_body": "| 特性 | 数值 |\n|---------|-------|\n| 速度 | 快 |\n| 准确率 | 高 |",
            "table_caption": ["特性对比"],
            "page_idx": 1
        }
    ]

    try:
        await rag.insert_content_list(
            content_list=another_content_list,
            file_path="another_document.pdf",
            doc_id="custom-doc-id-123"  # 自定义文档ID
        )
        print("✅ 第二个内容列表插入完成")
    except Exception as e:
        print(f"第二个内容列表插入失败: {e}")

    # 演示复杂内容类型
    print("\n--- 插入复杂内容类型 ---")
    complex_content_list = [
        {
            "type": "text",
            "text": "机器学习模型评估指标详解",
            "page_idx": 0
        },
        {
            "type": "equation",
            "latex": "Accuracy = \\frac{TP + TN}{TP + TN + FP + FN}",
            "text": "准确率计算公式",
            "page_idx": 1
        },
        {
            "type": "equation", 
            "latex": "Precision = \\frac{TP}{TP + FP}",
            "text": "精确率计算公式",
            "page_idx": 1
        },
        {
            "type": "equation",
            "latex": "Recall = \\frac{TP}{TP + FN}",
            "text": "召回率计算公式", 
            "page_idx": 1
        },
        {
            "type": "table",
            "table_body": """| 指标 | 公式 | 含义 |
|------|------|------|
| 准确率 | (TP+TN)/(TP+TN+FP+FN) | 正确预测的比例 |
| 精确率 | TP/(TP+FP) | 预测为正例中实际为正例的比例 |
| 召回率 | TP/(TP+FN) | 实际正例中被正确预测的比例 |""",
            "table_caption": ["评估指标汇总"],
            "table_footnote": ["TP=真正例, TN=真负例, FP=假正例, FN=假负例"],
            "page_idx": 2
        },
        {
            "type": "custom_content",
            "content": {
                "algorithm": "随机森林",
                "parameters": {"n_estimators": 100, "max_depth": 10},
                "performance": {"accuracy": 0.95, "f1_score": 0.94}
            },
            "page_idx": 3
        }
    ]

    try:
        await rag.insert_content_list(
            content_list=complex_content_list,
            file_path="ml_evaluation_guide.pdf",
            doc_id="ml-guide-001",
            display_stats=True
        )
        print("✅ 复杂内容列表插入完成")
    except Exception as e:
        print(f"复杂内容列表插入失败: {e}")

    # 综合查询所有插入的内容
    print("\n--- 综合查询所有内容 ---")
    try:
        comprehensive_result = await rag.aquery(
            "总结所有文档中关于机器学习性能评估的内容，包括公式、指标和实验结果。",
            mode="hybrid",
            enable_rerank=False,
            vlm_enhanced=False
        )
        print(f"综合查询结果: {comprehensive_result}")
    except Exception as e:
        print(f"综合查询失败: {e}")

    # 多模态查询测试
    print("\n--- 多模态查询测试 ---")
    try:
        multimodal_result = await rag.aquery_with_multimodal(
            "结合已有知识，分析这个新的实验结果",
            multimodal_content=[{
                "type": "table",
                "table_data": """模型,准确率,精确率,召回率,F1分数
                                BERT,0.92,0.91,0.93,0.92
                                RoBERTa,0.94,0.93,0.95,0.94
                                GPT-3,0.96,0.95,0.97,0.96""",
                "table_caption": "最新模型性能对比"
            }],
            mode="hybrid",
            enable_rerank=False,
            vlm_enhanced=False
        )
        print(f"多模态查询结果: {multimodal_result}")
    except Exception as e:
        print(f"多模态查询失败: {e}")

    # 清理资源
    try:
        await rag.finalize_storages()
        print("\n✅ 资源清理完成")
    except Exception as e:
        print(f"资源清理失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
