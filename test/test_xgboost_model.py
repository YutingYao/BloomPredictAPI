#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试XGBoost模型文件结构
"""

import pickle
import os

def test_xgboost_model():
    """测试XGBoost模型文件"""
    model_path = "models/00-XGB_model_data_胥湖心-去除负数.pkl"
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"模型文件: {model_path}")
        print(f"数据类型: {type(model_data)}")
        print(f"包含的键: {list(model_data.keys())}")
        
        if 'models' in model_data:
            print(f"models键包含的预测天数: {list(model_data['models'].keys())}")
            
            # 检查第一个预测天数的模型结构
            first_day = list(model_data['models'].keys())[0]
            model_tuple = model_data['models'][first_day]
            print(f"预测天数 {first_day} 的模型类型: {type(model_tuple)}")
            
            if isinstance(model_tuple, tuple):
                print(f"模型元组长度: {len(model_tuple)}")
                for i, item in enumerate(model_tuple):
                    print(f"  元素 {i}: 类型={type(item)}")
            else:
                print(f"模型对象: {model_tuple}")
        
    except Exception as e:
        print(f"加载模型文件失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_xgboost_model()
