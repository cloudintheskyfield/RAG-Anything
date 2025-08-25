"""
ws_04_custom_modal_processors.py
自定义模态处理器示例 - 基于README.md示例4
"""
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from raganything.modalprocessors import GenericModalProcessor
from common_utils import get_async_llm_model_func, get_async_vision_model_func, get_async_embedding_func

class CustomModalProcessor(GenericModalProcessor):
    """自定义模态处理器示例"""
    
    async def process_multimodal_content(self, modal_content, content_type, file_path, entity_name):
        """自定义处理逻辑"""
        print(f"处理自定义内容类型: {content_type}")
        
        # 自定义内容分析逻辑
        enhanced_description = await self.analyze_custom_content(modal_content)
        entity_info = self.create_custom_entity(enhanced_description, entity_name)
        
        return await self._create_entity_and_chunk(enhanced_description, entity_info, file_path)
    
    async def analyze_custom_content(self, modal_content):
        """分析自定义内容"""
        content_str = str(modal_content)
        
        # 使用LLM分析内容
        prompt = f"""
        请分析以下自定义内容并提供详细描述：
        
        内容: {content_str}
        
        请提供：
        1. 内容的主要特征
        2. 可能的用途和意义
        3. 与其他内容的潜在关联
        """
        
        try:
            description = await self.modal_caption_func(
                prompt,
                system_prompt="你是一个专业的内容分析师，擅长分析各种类型的数据和内容。"
            )
            return description
        except Exception as e:
            print(f"自定义内容分析失败: {e}")
            return f"自定义内容: {content_str[:200]}..."
    
    def create_custom_entity(self, description, entity_name):
        """创建自定义实体信息"""
        return {
            "entity_name": entity_name,
            "entity_type": "CUSTOM_CONTENT",
            "description": description,
            "custom_attributes": {
                "processor_type": "CustomModalProcessor",
                "analysis_method": "LLM_enhanced"
            }
        }

async def main():
    print("=== 自定义模态处理器示例 ===")
    
    # 获取异步模型函数
    llm_model_func = await get_async_llm_model_func()
    vision_model_func = await get_async_vision_model_func()
    embedding_func = await get_async_embedding_func()
    
    # 创建 RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir="./rag_storage_04",
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
    
    # 确保初始化完成
    await rag._ensure_lightrag_initialized()
    print("✅ RAGAnything 初始化完成")

    # 创建自定义处理器
    print("\n--- 创建自定义处理器 ---")
    custom_processor = CustomModalProcessor(
        lightrag=rag.lightrag,
        modal_caption_func=llm_model_func
    )

    # 处理自定义内容
    print("\n--- 处理自定义内容 ---")
    custom_content_examples = [
        {
            "type": "code_snippet",
            "language": "python",
            "code": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
            """,
            "description": "斐波那契数列递归实现"
        },
        {
            "type": "json_data",
            "data": {
                "users": [
                    {"name": "Alice", "age": 30, "role": "developer"},
                    {"name": "Bob", "age": 25, "role": "designer"}
                ],
                "total_count": 2
            },
            "schema": "用户信息数据"
        },
        {
            "type": "workflow",
            "steps": [
                "数据收集",
                "数据预处理", 
                "模型训练",
                "结果评估",
                "部署上线"
            ],
            "domain": "机器学习项目流程"
        }
    ]

    for i, content in enumerate(custom_content_examples):
        try:
            print(f"\n处理自定义内容 {i+1}: {content['type']}")
            description, entity_info = await custom_processor.process_multimodal_content(
                modal_content=content,
                content_type=content["type"],
                file_path="custom_document.txt",
                entity_name=f"自定义内容_{i+1}"
            )
            print(f"处理完成: {description[:100]}...")
        except Exception as e:
            print(f"处理自定义内容 {i+1} 失败: {e}")

    # 查询处理后的内容
    print("\n--- 查询处理后的内容 ---")
    try:
        result = await rag.aquery(
            "刚才处理的自定义内容包含哪些类型的信息？请总结主要特点。",
            mode="hybrid",
            enable_rerank=False,
            vlm_enhanced=False
        )
        print(f"查询结果: {result}")
    except Exception as e:
        print(f"查询失败: {e}")

    # 测试多模态查询
    print("\n--- 测试多模态查询 ---")
    try:
        multimodal_result = await rag.aquery_with_multimodal(
            "分析这个算法的复杂度和应用场景",
            multimodal_content=[{
                "type": "algorithm",
                "name": "快速排序",
                "complexity": "O(n log n) 平均情况, O(n²) 最坏情况",
                "description": "分治算法，选择基准元素进行分区排序"
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
        print("✅ 资源清理完成")
    except Exception as e:
        print(f"资源清理失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
