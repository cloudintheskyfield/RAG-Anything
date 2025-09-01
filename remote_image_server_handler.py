"""
远程图像服务器处理程序
在vLLM服务器上运行，提供图片上传和访问服务
"""
import os
import base64
import hashlib
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

class ImageUploadHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, storage_dir="/mnt/data3/nlp/ws/data", **kwargs):
        self.storage_dir = storage_dir
        # 确保存储目录存在
        os.makedirs(self.storage_dir, exist_ok=True)
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        # 禁用访问日志
        pass
    
    def do_GET(self):
        """处理GET请求 - 文件访问和状态检查"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/status':
            # 状态检查
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "ok", "storage_dir": self.storage_dir}
            self.wfile.write(json.dumps(response).encode())
            
        elif parsed_path.path.startswith('/'):
            # 文件访问
            filename = parsed_path.path[1:]  # 移除开头的 '/'
            if filename and self._is_safe_filename(filename):
                filepath = os.path.join(self.storage_dir, filename)
                if os.path.exists(filepath) and os.path.isfile(filepath):
                    self._serve_file(filepath)
                else:
                    self.send_error(404, "File not found")
            else:
                self.send_error(400, "Invalid filename")
        else:
            self.send_error(404, "Not found")
    
    def do_POST(self):
        """处理POST请求 - 图片上传"""
        if self.path == '/upload':
            try:
                # 读取请求数据
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # 解析JSON数据
                data = json.loads(post_data.decode())
                filename = data.get('filename')
                base64_data = data.get('data')
                extension = data.get('extension', 'jpg')
                
                if not filename or not base64_data:
                    self.send_error(400, "Missing filename or data")
                    return
                
                # 解码并保存图片
                image_bytes = base64.b64decode(base64_data)
                filepath = os.path.join(self.storage_dir, filename)
                
                # 如果文件不存在则保存
                if not os.path.exists(filepath):
                    with open(filepath, 'wb') as f:
                        f.write(image_bytes)
                
                # 返回成功响应
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = {
                    "success": True,
                    "url": f"http://127.0.0.1:10017/{filename}",
                    "filename": filename
                }
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_error(500, f"Upload failed: {str(e)}")
        else:
            self.send_error(404, "Not found")
    
    def _is_safe_filename(self, filename):
        """检查文件名是否安全"""
        if not filename or '..' in filename or '/' in filename or '\\' in filename:
            return False
        return filename.replace('.', '').replace('-', '').replace('_', '').isalnum()
    
    def _serve_file(self, filepath):
        """提供文件服务"""
        try:
            # 根据文件扩展名设置Content-Type
            ext = os.path.splitext(filepath)[1].lower()
            content_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp'
            }
            content_type = content_types.get(ext, 'application/octet-stream')
            
            with open(filepath, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            
        except Exception as e:
            self.send_error(500, f"Error serving file: {str(e)}")

class RemoteImageServer:
    def __init__(self, port=10017, storage_dir="/mnt/data3/nlp/ws/data", host='0.0.0.0'):
        self.port = port
        self.host = host
        self.storage_dir = storage_dir
        self.server = None
        self.server_thread = None
        self.running = False
        
        # 确保存储目录存在
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def start_server(self):
        """启动远程图像服务器"""
        if self.running:
            return
        
        def handler_factory(*args, **kwargs):
            return ImageUploadHandler(*args, storage_dir=self.storage_dir, **kwargs)
        
        try:
            self.server = HTTPServer((self.host, self.port), handler_factory)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            self.running = True
            print(f"Remote image server started on http://{self.host}:{self.port}")
            print(f"Storage directory: {self.storage_dir}")
        except Exception as e:
            print(f"Failed to start remote image server: {e}")
    
    def stop_server(self):
        """停止远程图像服务器"""
        if self.server and self.running:
            self.server.shutdown()
            self.server.server_close()
            self.running = False
            print("Remote image server stopped")

if __name__ == "__main__":
    # 启动服务器
    server = RemoteImageServer()
    server.start_server()
    
    try:
        print("Remote image server is running. Press Ctrl+C to stop.")
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop_server()
        print("Server stopped.")
