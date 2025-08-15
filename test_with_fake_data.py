#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用模拟数据测试胥湖心预报接口
"""

import requests
import json
import numpy as np
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

def generate_fake_data():
    """
    生成模拟的水质和气象数据
    基于胥湖心站点的典型数据范围
    """
    # 基础数据范围（基于参考代码中的典型值）
    base_data = {
        "temperature": 25.5,  # 温度变化范围 20-30°C
        "oxygen": 8.2,        # 溶解氧 6-12 mg/L
        "TN": 1.5,           # 总氮 1.0-2.5 mg/L
        "TP": 0.08,          # 总磷 0.05-0.15 mg/L
        "NH": 0.5,           # 氨氮 0.2-1.0 mg/L
        "pH": 7.8,           # pH 7.0-8.5
        "turbidity": 15.2,   # 浊度 5-50 NTU
        "conductivity": 420.0, # 电导率 300-600 μS/cm
        "permanganate": 3.5,   # 高锰酸盐指数 2-8 mg/L
        "rain_sum": 0.0,       # 降雨量 0-20 mm
        "wind_speed_10m_max": 3.2,      # 风速 0-10 m/s
        "shortwave_radiation_sum": 18.5  # 短波辐射 10-30 MJ/m²
    }
    
    # 添加一些随机变化，模拟真实数据
    np.random.seed(42)  # 固定随机种子以便复现
    
    variations = {
        "temperature": np.random.normal(0, 2, 1)[0],
        "oxygen": np.random.normal(0, 1, 1)[0],
        "TN": np.random.normal(0, 0.3, 1)[0],
        "TP": np.random.normal(0, 0.02, 1)[0],
        "NH": np.random.normal(0, 0.1, 1)[0],
        "pH": np.random.normal(0, 0.2, 1)[0],
        "turbidity": np.random.normal(0, 5, 1)[0],
        "conductivity": np.random.normal(0, 50, 1)[0],
        "permanganate": np.random.normal(0, 1, 1)[0],
        "rain_sum": max(0, np.random.normal(0, 2, 1)[0]),
        "wind_speed_10m_max": max(0, np.random.normal(0, 1, 1)[0]),
        "shortwave_radiation_sum": max(0, np.random.normal(0, 3, 1)[0])
    }
    
    # 应用变化并确保值在合理范围内
    fake_data = {}
    for key, base_value in base_data.items():
        new_value = base_value + variations[key]
        # 确保值在合理范围内
        if key == "temperature":
            new_value = max(0, min(35, new_value))
        elif key == "pH":
            new_value = max(6, min(9, new_value))
        elif key in ["oxygen", "TN", "TP", "NH", "turbidity", "conductivity", "permanganate"]:
            new_value = max(0, new_value)
        
        fake_data[key] = round(new_value, 2)
    
    return fake_data

def test_prediction_single_day():
    """测试单天预测"""
    print("🧪 测试胥湖心1天预报接口...")
    
    fake_data = generate_fake_data()
    print(f"📊 使用模拟数据: {json.dumps(fake_data, indent=2, ensure_ascii=False)}")
    
    payload = {
        "station": "胥湖心",
        "model_type": "grud",
        "predict_days": 1,
        "input_data": fake_data
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/predict", json=payload, timeout=30)
        print(f"📡 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 预测成功!")
            print(f"📈 预测结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ 预测失败: {response.status_code}")
            print(f"错误详情: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

def test_multiple_days():
    """测试多天预测"""
    print("\n🧪 测试胥湖心多天预报接口...")
    
    test_days = [1, 3, 7, 15, 30]
    results = {}
    
    fake_data = generate_fake_data()
    
    for days in test_days:
        print(f"\n📅 测试 {days} 天预报...")
        
        payload = {
            "station": "胥湖心",
            "model_type": "grud",
            "predict_days": days,
            "input_data": fake_data
        }
        
        try:
            response = requests.post(f"{BASE_URL}/api/predict", json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                results[days] = {
                    "status": "success",
                    "prediction_count": len(result.get("data", {}).get("prediction", [])),
                    "confidence": result.get("data", {}).get("confidence", 0),
                    "rmse": result.get("data", {}).get("rmse", 0)
                }
                print(f"✅ {days}天预测成功 - 预测点数: {results[days]['prediction_count']}")
            else:
                results[days] = {
                    "status": "failed",
                    "error": response.text
                }
                print(f"❌ {days}天预测失败: {response.status_code}")
                
        except Exception as e:
            results[days] = {
                "status": "error", 
                "error": str(e)
            }
            print(f"❌ {days}天预测异常: {e}")
    
    print(f"\n📊 测试结果汇总:")
    for days, result in results.items():
        status = result['status']
        if status == "success":
            print(f"  {days:2d}天: ✅ 成功 (预测点数: {result['prediction_count']}, 置信度: {result['confidence']:.2f})")
        else:
            print(f"  {days:2d}天: ❌ {status}")
    
    return results

def test_different_models():
    """测试不同模型"""
    print("\n🧪 测试胥湖心不同模型...")
    
    models = ["lstm", "grud", "tcn", "xgboost"]
    fake_data = generate_fake_data()
    results = {}
    
    for model in models:
        print(f"\n🤖 测试 {model.upper()} 模型...")
        
        payload = {
            "station": "胥湖心",
            "model_type": model,
            "predict_days": 7,  # 测试7天预报
            "input_data": fake_data
        }
        
        try:
            response = requests.post(f"{BASE_URL}/api/predict", json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                results[model] = {
                    "status": "success",
                    "prediction_count": len(result.get("data", {}).get("prediction", [])),
                    "confidence": result.get("data", {}).get("confidence", 0),
                    "rmse": result.get("data", {}).get("rmse", 0),
                    "improvement": result.get("data", {}).get("model_info", {}).get("improvement_over_lstm", "0%")
                }
                print(f"✅ {model.upper()}模型预测成功")
            else:
                results[model] = {
                    "status": "failed",
                    "error": response.text
                }
                print(f"❌ {model.upper()}模型预测失败: {response.status_code}")
                
        except Exception as e:
            results[model] = {
                "status": "error",
                "error": str(e)
            }
            print(f"❌ {model.upper()}模型预测异常: {e}")
    
    print(f"\n📊 模型测试结果汇总:")
    for model, result in results.items():
        status = result['status']
        if status == "success":
            print(f"  {model.upper():8s}: ✅ 成功 (RMSE: {result['rmse']:.4f}, 改善率: {result['improvement']})")
        else:
            print(f"  {model.upper():8s}: ❌ {status}")
    
    return results

def main():
    """主测试函数"""
    print("🚀 开始测试蓝藻预测API - 胥湖心站点")
    print("=" * 60)
    
    # 检查服务器状态
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=10)
        if health_response.status_code == 200:
            print("✅ 服务器健康检查通过")
        else:
            print("⚠️ 服务器状态异常")
            return
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
        return
    
    # 测试单天预测
    success = test_prediction_single_day()
    
    if success:
        # 测试多天预测
        test_multiple_days()
        
        # 测试不同模型
        test_different_models()
    
    print("\n" + "=" * 60)
    print("📋 测试完成!")

if __name__ == "__main__":
    main()
