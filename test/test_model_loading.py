#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型加载功能
"""

import asyncio
import logging
import pickle
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.model_manager import ModelManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_model_loading():
    """测试模型加载功能"""
    try:
        logger.info("开始测试模型加载...")
        
        # 创建模型管理器
        model_manager = ModelManager()
        
        # 初始化（包含pickle兼容性修复）
        await model_manager.initialize()
        
        # 检查加载状态
        status = await model_manager.get_loaded_models_status()
        logger.info(f"模型加载状态: {status}")
        
        # 测试获取一个特定模型
        if status:
            # 选择第一个可用的站点和模型类型
            station = list(status.keys())[0]
            model_type = status[station][0] if status[station] else None
            
            if model_type:
                logger.info(f"测试获取模型: {station} - {model_type}")
                
                # 尝试获取模型（预测1天）
                model = await model_manager.get_model(station, model_type, 1)
                if model:
                    logger.info(f"成功获取模型: {type(model)}")
                else:
                    logger.warning("无法获取模型")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pickle_file_directly():
    """直接测试pickle文件加载"""
    try:
        logger.info("直接测试pickle文件加载...")
        
        # 测试加载一个模型文件
        test_file = "models/00-GRUD_model_data_胥湖心-去除负数.pkl"
        
        if os.path.exists(test_file):
            logger.info(f"尝试加载文件: {test_file}")
            
            with open(test_file, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"文件加载成功，包含的键: {list(data.keys())}")
            
            # 检查数据结构
            for key, value in data.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}: dict with keys {list(value.keys())}")
                else:
                    logger.info(f"  {key}: {type(value)}")
            
            return True
        else:
            logger.error(f"文件不存在: {test_file}")
            return False
            
    except Exception as e:
        logger.error(f"直接加载pickle文件失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 蓝藻预测系统模型加载测试 ===\n")
    
    # 先测试直接pickle加载
    print("1. 测试直接pickle文件加载...")
    pickle_result = test_pickle_file_directly()
    print(f"结果: {'成功' if pickle_result else '失败'}\n")
    
    # 再测试模型管理器
    print("2. 测试模型管理器...")
    manager_result = asyncio.run(test_model_loading())
    print(f"结果: {'成功' if manager_result else '失败'}\n")
    
    # 总结
    print("=== 测试总结 ===")
    print(f"直接pickle加载: {'✅' if pickle_result else '❌'}")
    print(f"模型管理器: {'✅' if manager_result else '❌'}")
    
    if pickle_result and manager_result:
        print("\n🎉 模型加载功能正常！")
    else:
        print("\n⚠️  模型加载存在问题，需要进一步修复。")
