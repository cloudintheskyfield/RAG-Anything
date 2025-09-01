"""
简单的图片文件服务器
用于将base64图片数据转换为可访问的URL
"""
import os
import base64
import hashlib
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import unquote
import tempfile

class ImageServer:
    def __init__(self, port=10017, temp_dir=None, host='0.0.0.0'):
        self.port = port
        self.host = host
        self.temp_dir = temp_dir or "/mnt/data3/nlp/ws/data"
        self.server = None
        self.server_thread = None
        self.running = False
        
        # 确保临时目录存在
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def _get_image_hash(self, image_data):
        """根据图片数据生成唯一哈希值"""
        return hashlib.md5(image_data.encode()).hexdigest()
    
    def save_base64_image(self, base64_data, file_extension="jpg"):
        """保存base64图片数据到临时文件，返回访问URL"""
        try:
            # 移除data:image前缀（如果存在）
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',', 1)[1]
            
            # 解码base64数据
            image_bytes = base64.b64decode(base64_data)
            
            # 生成唯一文件名
            image_hash = hashlib.md5(image_bytes).hexdigest()
            filename = f"{image_hash}.{file_extension}"
            filepath = os.path.join(self.temp_dir, filename)
            
            # 如果文件不存在则保存
            if not os.path.exists(filepath):
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
            
            # 返回访问URL
            return f"http://127.0.0.1:{self.port}/{filename}"
            
        except Exception as e:
            print(f"Error saving base64 image: {e}")
            return None
    
    def start_server(self):
        """启动文件服务器"""
        if self.running:
            return
            
        class CustomHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=self.temp_dir, **kwargs)
            
            def log_message(self, format, *args):
                # 禁用访问日志
                pass
        
        try:
            self.server = HTTPServer((self.host, self.port), CustomHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            self.running = True
            print(f"Image server started on http://{self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to start image server: {e}")
    
    def stop_server(self):
        """停止文件服务器"""
        if self.server and self.running:
            self.server.shutdown()
            self.server.server_close()
            self.running = False
            print("Image server stopped")
    
    def cleanup_old_files(self, max_age_hours=24):
        """清理超过指定时间的临时文件"""
        try:
            current_time = time.time()
            for filename in os.listdir(self.temp_dir):
                filepath = os.path.join(self.temp_dir, filename)
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > max_age_hours * 3600:
                        os.remove(filepath)
        except Exception as e:
            print(f"Error cleaning up old files: {e}")

# 全局图片服务器实例
_image_server = None

def get_image_server():
    """获取全局图片服务器实例"""
    global _image_server
    if _image_server is None:
        _image_server = ImageServer()
        _image_server.start_server()
    return _image_server

def convert_base64_to_url(base64_data):
    """将base64图片数据转换为可访问的URL"""
    server = get_image_server()
    return server.save_base64_image(base64_data)

if __name__ == "__main__":
    # 测试服务器
    server = ImageServer()
    server.start_server()
    
    try:
        print("Image server is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop_server()
        print("Server stopped.")
