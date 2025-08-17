#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蓝藻预测API服务器启动脚本
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
import uvicorn

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        raise RuntimeError("需要Python 3.8或更高版本")

def check_dependencies():
    """检查依赖是否安装"""
    logger = logging.getLogger(__name__)
    
    try:
        import fastapi
        import torch
        import numpy
        import pandas
        import sklearn
        import xgboost
        logger.info("✓ 所有依赖已安装")
        return True
    except ImportError as e:
        logger.error(f"× 缺少依赖: {e}")
        logger.info("请运行: pip install -r requirements.txt")
        return False

def create_directories():
    """创建必要的目录"""
    logger = logging.getLogger(__name__)
    
    directories = ["logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✓ 确保目录存在: {directory}")

def check_model_files():
    """检查模型文件是否存在"""
    logger = logging.getLogger(__name__)
    
    stations = ["胥湖心", "锡东水厂", "平台山", "tuoshan", "lanshanzui", "五里湖心"]
    model_types = ["lstm", "GRUD", "TCN", "XGB"]
    
    missing_files = []
    found_files = []
    
    for station in stations:
        for model_type in model_types:
            # 根据不同模型类型构建文件名
            if model_type == "lstm":
                filename = f"00-lstm_model_data_{station}-去除负数.pkl"
            else:
                filename = f"00-{model_type}_model_data_{station}-去除负数.pkl"
            
            if os.path.exists(filename):
                found_files.append(filename)
            else:
                missing_files.append(filename)
    
    logger.info(f"✓ 找到 {len(found_files)} 个模型文件")
    
    if missing_files:
        logger.warning(f"⚠️  缺少 {len(missing_files)} 个模型文件:")
        for file in missing_files[:5]:  # 只显示前5个
            logger.warning(f"   - {file}")
        if len(missing_files) > 5:
            logger.warning(f"   ... 还有 {len(missing_files) - 5} 个文件")
        logger.info("注意: 某些预测功能可能不可用")
    
    return len(found_files) > 0

def start_server(host="0.0.0.0", port=8000, reload=False, workers=1):
    """启动API服务器"""
    logger = logging.getLogger(__name__)
    
    cmd = [
        "uvicorn",
        "main:app",
        "--host", host,
        "--port", str(port),
        "--workers", str(workers)
    ]
    
    if reload:
        cmd.append("--reload")
    
    logger.info(f"🚀 启动服务器: http://{host}:{port}")
    logger.info(f"📚 API文档: http://{host}:{port}/docs")
    logger.info(f"🔍 健康检查: http://{host}:{port}/health")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("🛑 服务器已停止")
    except subprocess.CalledProcessError as e:
        logger.error(f"服务器启动失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="蓝藻预测API服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="启用自动重载（开发模式）")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数量")
    parser.add_argument("--skip-checks", action="store_true", help="跳过预检查")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    try:
        logger.info("🧪 蓝藻预测API服务器启动检查")
        
        if not args.skip_checks:
            # 检查Python版本
            logger.info("检查Python版本...")
            check_python_version()
            logger.info(f"✓ Python版本: {sys.version}")
            
            # 检查依赖
            logger.info("检查依赖...")
            if not check_dependencies():
                return 1
            
            # 创建目录
            logger.info("创建必要目录...")
            create_directories()
            
            # 检查模型文件
            logger.info("检查模型文件...")
            if not check_model_files():
                logger.warning("⚠️  未找到任何模型文件，某些功能可能不可用")
                
                response = input("是否继续启动服务器？(y/N): ")
                if response.lower() not in ['y', 'yes']:
                    logger.info("取消启动")
                    return 1
        
        logger.info("✅ 所有检查通过")
        
        # 启动服务器
        success = start_server(
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers
        )
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"启动失败: {e}")
        return 1

if __name__ == "__main__":
    # 启动V3版本的API服务
    uvicorn.run("main_v3:app", host="0.0.0.0", port=8000, reload=True)
