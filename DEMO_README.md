# RAG-Anything Demo 测试文件说明

本目录包含基于README.md示例创建的完整demo测试文件，展示RAG-Anything的各种功能。

## 文件结构

### 公共工具模块
- **`common_utils.py`** - 封装了LLM和embedding模型的公共函数，使用部署的自托管服务

### Demo测试文件

1. **`ws_01_end_to_end_document.py`** - 端到端文档处理示例
   - 完整的文档解析和多模态内容处理流程
   - 包含文本查询和多模态查询示例

2. **`ws_02_direct_multimodal_processing.py`** - 直接多模态内容处理
   - 直接使用LightRAG和模态处理器
   - 演示图像和表格内容的处理

3. **`ws_03_batch_processing.py`** - 批量处理示例
   - 批量处理多个文档
   - 支持递归文件夹处理和并发处理

4. **`ws_04_custom_modal_processors.py`** - 自定义模态处理器
   - 创建自定义内容处理器
   - 处理代码片段、JSON数据、工作流程等自定义内容类型

5. **`ws_05_query_options.py`** - 查询选项示例
   - 演示不同查询模式（hybrid、local、global、naive）
   - VLM增强查询和多模态查询示例

6. **`ws_06_loading_existing_lightrag.py`** - 加载现有LightRAG实例
   - 加载和复用现有的LightRAG知识库
   - 向现有实例添加新内容

7. **`ws_07_direct_content_list_insertion.py`** - 直接内容列表插入
   - 绕过文档解析，直接插入预处理的内容列表
   - 支持多种内容类型和自定义文档ID

### 工具文件
- **`ws_test_runner.py`** - 测试运行器，用于验证所有demo文件的语法和基本功能

## 配置说明

### API配置
所有demo文件使用`common_utils.py`中的配置：

```python
# LLM API配置
LLM_API_URL = "http://223.109.239.14:10000/v1/chat/completions"
MODEL = "qwen2__5-72b"

# 嵌入API配置  
BASE_URL = "http://10.25.20.246:6109/v1"
EMBEDDING_DIM = 1024
```

### 文档路径配置
需要根据实际情况修改以下路径：
- 测试PDF文档: `C:\Users\shuang_wang\Downloads\Dummy-PDF-3Pages.pdf`
- 测试图像: `C:\Users\shuang_wang\Downloads\sample_image.jpg`
- 批量处理目录: `./documents`

## 运行方法

### 1. 运行测试检查器
```bash
conda run -n llm python ws_test_runner.py
```

### 2. 运行单个demo
```bash
# 端到端文档处理
conda run -n llm python ws_01_end_to_end_document.py

# 直接多模态处理
conda run -n llm python ws_02_direct_multimodal_processing.py

# 批量处理
conda run -n llm python ws_03_batch_processing.py

# 自定义处理器
conda run -n llm python ws_04_custom_modal_processors.py

# 查询选项
conda run -n llm python ws_05_query_options.py

# 加载现有实例
conda run -n llm python ws_06_loading_existing_lightrag.py

# 内容列表插入
conda run -n llm python ws_07_direct_content_list_insertion.py
```

## 存储目录说明

每个demo使用独立的存储目录：
- `ws_01`: `./rag_storage` (原有目录)
- `ws_02`: `./rag_storage_02`
- `ws_03`: `./rag_storage_03`
- `ws_04`: `./rag_storage_04`
- `ws_05`: `./rag_storage_05`
- `ws_06`: `./existing_lightrag_storage_06`
- `ws_07`: `./rag_storage_07`

## 功能特点

### 统一的模型配置
- 所有demo使用相同的LLM和embedding配置
- 支持同步和异步两种模式
- 统一的错误处理和资源清理

### 完整的示例覆盖
- 涵盖README.md中的所有主要使用场景
- 包含错误处理和异常情况演示
- 提供详细的中文注释和说明

### 模块化设计
- 公共功能封装在`common_utils.py`中
- 每个demo专注于特定功能演示
- 易于维护和扩展

## 注意事项

1. **文件路径**: 请确保测试文件路径存在，否则相关功能会跳过
2. **网络连接**: 需要能够访问配置的LLM和embedding服务
3. **存储空间**: 每个demo会创建独立的存储目录
4. **资源清理**: 所有demo都包含资源清理逻辑
5. **错误处理**: 包含完整的异常处理，不会因单个错误中断整个流程

## 扩展建议

- 可以基于这些demo创建自己的应用场景
- 修改`common_utils.py`来适配不同的模型服务
- 添加更多自定义内容类型处理器
- 集成到现有的工作流程中
