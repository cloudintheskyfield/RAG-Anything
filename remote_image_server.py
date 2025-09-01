"""
远程图像服务器客户端
用于将base64图片上传到远程服务器并获取访问URL
"""
import requests
import base64
import hashlib
import json

class RemoteImageClient:
    def __init__(self, server_url="http://223.109.239.14:10017"):
        self.server_url = server_url.rstrip('/')
        
    def upload_base64_image(self, base64_data, file_extension="jpg"):
        """上传base64图片到远程服务器，返回远程访问URL"""
        try:
            # 移除data:image前缀（如果存在）
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',', 1)[1]
            
            # 解码base64数据以生成哈希
            image_bytes = base64.b64decode(base64_data)
            image_hash = hashlib.md5(image_bytes).hexdigest()
            filename = f"{image_hash}.{file_extension}"
            
            # 首先检查文件是否已存在
            check_url = f"{self.server_url}/{filename}"
            try:
                check_response = requests.head(check_url, timeout=5)
                if check_response.status_code == 200:
                    # 文件已存在，直接返回本地访问URL
                    return f"http://127.0.0.1:10017/{filename}"
            except:
                pass  # 文件不存在，继续上传
            
            # 构建上传URL
            upload_url = f"{self.server_url}/upload"
            
            # 准备上传数据
            payload = {
                "filename": filename,
                "data": base64_data,
                "extension": file_extension
            }
            
            # 发送POST请求上传图片
            headers = {"Content-Type": "application/json"}
            response = requests.post(upload_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    # 返回本地访问URL，因为vLLM在同一服务器上
                    return f"http://127.0.0.1:10017/{filename}"
                else:
                    print(f"Upload failed: {result.get('error', 'Unknown error')}")
                    return None
            else:
                print(f"HTTP error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error uploading base64 image: {e}")
            return None
    
    def check_server_status(self):
        """检查远程服务器状态"""
        try:
            response = requests.get(f"{self.server_url}/status", timeout=5)
            return response.status_code == 200
        except:
            return False

# 全局远程图像客户端实例
_remote_client = None

def get_remote_image_client():
    """获取全局远程图像客户端实例"""
    global _remote_client
    if _remote_client is None:
        _remote_client = RemoteImageClient()
    return _remote_client

def convert_base64_to_remote_url(base64_data):
    """将base64图片数据上传到远程服务器并返回URL"""
    client = get_remote_image_client()
    return client.upload_base64_image(base64_data)

# 简单的本地转换函数（作为备用）
def convert_base64_to_url(base64_data):
    """
    将base64图片转换为URL
    优先尝试远程服务器，失败则使用本地方案
    """
    # 首先尝试远程服务器
    client = get_remote_image_client()
    if client.check_server_status():
        url = client.upload_base64_image(base64_data)
        if url:
            return url
    
    # 远程服务器不可用，使用本地方案
    try:
        from image_server import get_image_server
        server = get_image_server()
        return server.save_base64_image(base64_data)
    except ImportError:
        print("Warning: Both remote and local image servers unavailable")
        return None

if __name__ == "__main__":
    # 测试客户端
    client = RemoteImageClient()
    print(f"Server status: {'OK' if client.check_server_status() else 'Not available'}")
