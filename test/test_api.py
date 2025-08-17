#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蓝藻预测API测试脚本
"""

import requests
import json
import time
from typing import Dict, Any

# API基础URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """测试健康检查接口"""
    print("=== 测试健康检查接口 ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"请求失败: {e}")
        return False

def test_get_stations():
    """测试获取站点列表接口"""
    print("\n=== 测试获取站点列表接口 ===")
    try:
        response = requests.get(f"{BASE_URL}/api/stations")
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"支持的站点数量: {len(result['stations'])}")
        for station in result['stations']:
            print(f"- {station['name']} ({station['name_en']}) - {station['zone']}")
        return response.status_code == 200
    except Exception as e:
        print(f"请求失败: {e}")
        return False

def test_get_models():
    """测试获取模型列表接口"""
    print("\n=== 测试获取模型列表接口 ===")
    try:
        response = requests.get(f"{BASE_URL}/api/models")
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"支持的模型数量: {len(result['models'])}")
        for model in result['models']:
            recommended = " (推荐)" if model.get('recommended') else ""
            print(f"- {model['name']} ({model['type']}) - {model['description']}{recommended}")
        return response.status_code == 200
    except Exception as e:
        print(f"请求失败: {e}")
        return False

def test_prediction():
    """测试预测接口"""
    print("\n=== 测试预测接口 ===")
    
    # 测试数据
    test_data = {
        "station": "胥湖心",
        "model_type": "grud",
        "predict_days": 7,
        "input_data": {
            "temperature": 25.5,
            "oxygen": 8.2,
            "TN": 1.5,
            "TP": 0.08,
            "NH": 0.5,
            "pH": 7.8,
            "turbidity": 15.2,
            "conductivity": 420.0,
            "permanganate": 3.5,
            "rain_sum": 0.0,
            "wind_speed_10m_max": 3.2,
            "shortwave_radiation_sum": 18.5
        }
    }
    
    try:
        print(f"测试站点: {test_data['station']}")
        print(f"模型类型: {test_data['model_type']}")
        print(f"预测天数: {test_data['predict_days']}")
        
        response = requests.post(f"{BASE_URL}/api/predict", json=test_data)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                data = result["data"]
                print(f"预测成功!")
                print(f"置信度: {data.get('confidence', 0):.3f}")
                print(f"RMSE: {data.get('rmse', 0):.4f}")
                predictions = data.get('prediction', [])
                print(f"预测结果 ({len(predictions)}天): {predictions}")
                return True
            else:
                print(f"预测失败: {result.get('message', '未知错误')}")
        else:
            print(f"请求失败: {response.text}")
        
        return False
        
    except Exception as e:
        print(f"请求异常: {e}")
        return False

def test_model_performance():
    """测试模型性能接口"""
    print("\n=== 测试模型性能接口 ===")
    
    test_data = {
        "station": "胥湖心",
        "model_types": ["lstm", "grud", "tcn"]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/model-performance", json=test_data)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                data = result["data"]
                print("性能对比获取成功!")
                
                # 显示性能对比
                for station_data in data.get("station_performance", []):
                    print(f"\n站点: {station_data['station']} ({station_data['zone']})")
                    for model in station_data['models']:
                        print(f"  {model['model_name']}: RMSE={model['rmse']:.4f}, 改善率={model['improvement_over_lstm']:.2f}%")
                
                # 显示总结
                summary = data.get("summary", {})
                print(f"\n总结:")
                print(f"  最佳模型: {summary.get('best_model', 'N/A')}")
                print(f"  整体改善率: {summary.get('overall_improvement', 0):.2f}%")
                print(f"  推荐模型: {summary.get('recommended_model', 'N/A')}")
                
                return True
            else:
                print(f"获取性能信息失败: {result.get('message', '未知错误')}")
        else:
            print(f"请求失败: {response.text}")
        
        return False
        
    except Exception as e:
        print(f"请求异常: {e}")
        return False

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    # 测试无效站点
    invalid_data = {
        "station": "无效站点",
        "model_type": "grud",
        "predict_days": 7,
        "input_data": {
            "temperature": 25.5,
            "oxygen": 8.2,
            "TN": 1.5,
            "TP": 0.08,
            "NH": 0.5,
            "pH": 7.8,
            "turbidity": 15.2,
            "conductivity": 420.0,
            "permanganate": 3.5,
            "rain_sum": 0.0,
            "wind_speed_10m_max": 3.2,
            "shortwave_radiation_sum": 18.5
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/predict", json=invalid_data)
        print(f"测试无效站点 - 状态码: {response.status_code}")
        
        if response.status_code == 422:  # 验证错误
            print("✓ 正确返回验证错误")
            return True
        else:
            print(f"× 未按预期返回错误: {response.text}")
            return False
            
    except Exception as e:
        print(f"请求异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 蓝藻预测API测试开始")
    print(f"测试目标: {BASE_URL}")
    
    # 等待服务启动
    print("\n等待服务启动...")
    time.sleep(2)
    
    # 执行测试
    tests = [
        ("健康检查", test_health_check),
        ("获取站点列表", test_get_stations),
        ("获取模型列表", test_get_models),
        ("预测接口", test_prediction),
        ("模型性能", test_model_performance),
        ("错误处理", test_error_handling),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✓ {test_name} - 通过")
            else:
                print(f"× {test_name} - 失败")
        except Exception as e:
            print(f"× {test_name} - 异常: {e}")
            results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "="*50)
    print("📊 测试总结")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "× 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！API运行正常。")
    else:
        print("⚠️  部分测试失败，请检查API服务状态。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
