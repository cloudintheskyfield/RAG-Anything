"""
ws_test_runner.py
测试运行器 - 用于测试所有demo文件的基本功能
"""
import asyncio
import sys
import traceback
from pathlib import Path

# 测试文件列表
TEST_FILES = [
    "ws_01_end_to_end_document.py",
    "ws_02_direct_multimodal_processing.py", 
    "ws_03_batch_processing.py",
    "ws_04_custom_modal_processors.py",
    "ws_05_query_options.py",
    "ws_06_loading_existing_lightrag.py",
    "ws_07_direct_content_list_insertion.py"
]

async def test_import(file_name):
    """测试文件是否可以正常导入"""
    try:
        module_name = file_name.replace('.py', '')
        # 动态导入模块
        spec = __import__(module_name)
        print(f"✅ {file_name}: 导入成功")
        return True
    except Exception as e:
        print(f"❌ {file_name}: 导入失败 - {str(e)}")
        return False

async def test_syntax(file_name):
    """测试文件语法是否正确"""
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, file_name, 'exec')
        print(f"✅ {file_name}: 语法检查通过")
        return True
    except SyntaxError as e:
        print(f"❌ {file_name}: 语法错误 - {str(e)}")
        return False
    except Exception as e:
        print(f"❌ {file_name}: 检查失败 - {str(e)}")
        return False

async def main():
    print("=== Demo文件测试运行器 ===\n")
    
    # 检查文件是否存在
    print("--- 检查文件存在性 ---")
    missing_files = []
    for file_name in TEST_FILES:
        if Path(file_name).exists():
            print(f"✅ {file_name}: 文件存在")
        else:
            print(f"❌ {file_name}: 文件不存在")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n缺失文件: {missing_files}")
        return
    
    # 语法检查
    print("\n--- 语法检查 ---")
    syntax_results = []
    for file_name in TEST_FILES:
        result = await test_syntax(file_name)
        syntax_results.append((file_name, result))
    
    # 检查common_utils.py
    print("\n--- 检查公共工具模块 ---")
    if Path("common_utils.py").exists():
        print("✅ common_utils.py: 文件存在")
        utils_syntax = await test_syntax("common_utils.py")
        if utils_syntax:
            print("✅ common_utils.py: 语法检查通过")
        else:
            print("❌ common_utils.py: 语法检查失败")
    else:
        print("❌ common_utils.py: 文件不存在")
    
    # 总结
    print("\n=== 测试总结 ===")
    passed = sum(1 for _, result in syntax_results if result)
    total = len(syntax_results)
    print(f"语法检查通过: {passed}/{total}")
    
    if passed == total:
        print("✅ 所有demo文件语法检查通过！")
        print("\n可以使用以下命令运行各个demo:")
        for file_name in TEST_FILES:
            print(f"  conda run -n llm python {file_name}")
    else:
        print("❌ 部分文件存在语法问题，请检查修复")
        for file_name, result in syntax_results:
            if not result:
                print(f"  需要修复: {file_name}")

if __name__ == "__main__":
    asyncio.run(main())
