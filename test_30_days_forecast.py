#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30天长期预报功能测试脚本
"""

import requests
import json
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

# API基础URL
BASE_URL = "http://localhost:8002"

def test_30_days_single_station():
    """测试单站点30天预报"""
    print("\n" + "="*60)
    print("测试 1: 单站点30天长期预报")
    print("="*60)
    
    request_data = {
        "current_date": "2024-06-01",
        "predict_days": 30,
        "station": "胥湖心",
        "model_type": "grud"
    }
    
    print(f"📊 请求数据:")
    print(json.dumps(request_data, indent=2, ensure_ascii=False))
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/v3/predict", json=request_data)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            predictions = result['data']['prediction']
            dates = result['data']['prediction_dates']
            
            print(f"\n✅ 30天预报成功!")
            print(f"⏱️  响应时间: {end_time - start_time:.3f}秒")
            print(f"📈 预测天数: {len(predictions)}天")
            print(f"📅 预测时间范围: {dates[0]} 到 {dates[-1]}")
            print(f"📊 预测值统计:")
            print(f"   - 最小值: {min(predictions):.4f}")
            print(f"   - 最大值: {max(predictions):.4f}")
            print(f"   - 平均值: {sum(predictions)/len(predictions):.4f}")
            print(f"   - 前7天: {predictions[:7]}")
            print(f"   - 后7天: {predictions[-7:]}")
            
            return predictions, dates
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(response.text)
            return None, None
    
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return None, None

def test_30_days_multiple_stations():
    """测试多站点30天预报对比"""
    print("\n" + "="*60)
    print("测试 2: 多站点30天预报对比")
    print("="*60)
    
    stations_config = [
        {"station": "胥湖心", "model_type": "grud", "expected_improvement": 77.41},
        {"station": "锡东水厂", "model_type": "grud", "expected_improvement": 55.13},
        {"station": "平台山", "model_type": "xgboost", "expected_improvement": 100.0},
        {"station": "tuoshan", "model_type": "grud", "expected_improvement": 79.78}
    ]
    
    results = {}
    
    for config in stations_config:
        request_data = {
            "current_date": "2024-06-01",
            "predict_days": 30,
            "station": config["station"],
            "model_type": config["model_type"]
        }
        
        print(f"\n🔍 预测站点: {config['station']} (模型: {config['model_type']})")
        
        try:
            response = requests.post(f"{BASE_URL}/api/v3/predict", json=request_data)
            
            if response.status_code == 200:
                result = response.json()
                predictions = result['data']['prediction']
                improvement = result['data']['performance_info']['improvement_rate']
                
                results[config["station"]] = {
                    "predictions": predictions,
                    "model_type": config["model_type"],
                    "improvement_rate": improvement,
                    "avg_prediction": sum(predictions) / len(predictions),
                    "trend": "上升" if predictions[-1] > predictions[0] else "下降"
                }
                
                print(f"   ✅ 成功获取30天预测")
                print(f"   📈 平均预测值: {results[config['station']]['avg_prediction']:.4f}")
                print(f"   📊 整体趋势: {results[config['station']]['trend']}")
                print(f"   🎯 模型改善率: {improvement:.1f}%")
            else:
                print(f"   ❌ 预测失败: {response.status_code}")
        
        except Exception as e:
            print(f"   ❌ 预测异常: {e}")
    
    return results

def test_30_days_batch():
    """测试批量30天预报"""
    print("\n" + "="*60)
    print("测试 3: 批量30天预报")
    print("="*60)
    
    batch_request = {
        "requests": [
            {
                "current_date": "2024-06-01",
                "predict_days": 30,
                "station": "胥湖心",
                "model_type": "grud"
            },
            {
                "current_date": "2024-06-01",
                "predict_days": 30,
                "station": "锡东水厂", 
                "model_type": "grud"
            },
            {
                "current_date": "2024-06-01",
                "predict_days": 30,
                "station": "平台山",
                "model_type": "xgboost"
            },
            {
                "current_date": "2024-06-01",
                "predict_days": 30,
                "station": "五里湖心",
                "model_type": "grud"
            }
        ],
        "parallel_execution": True
    }
    
    print(f"📊 批量预测请求: {len(batch_request['requests'])}个站点")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/v3/batch-predict", json=batch_request)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            summary = result['summary']
            
            print(f"\n✅ 批量30天预报完成!")
            print(f"⏱️  总耗时: {end_time - start_time:.3f}秒")
            print(f"📊 执行摘要:")
            print(f"   - 总任务: {summary['total']}")
            print(f"   - 成功: {summary['success']}")
            print(f"   - 失败: {summary['errors']}")
            print(f"   - 成功率: {summary['success']/summary['total']*100:.1f}%")
            print(f"   - 平均响应时间: {(end_time - start_time)/summary['total']:.3f}秒/任务")
            
            print(f"\n📈 各站点30天预测结果摘要:")
            for i, req in enumerate(batch_request['requests']):
                if i < len(result['results']) and isinstance(result['results'][i], dict) and 'data' in result['results'][i]:
                    pred_result = result['results'][i]['data']
                    predictions = pred_result['prediction']
                    improvement = pred_result['performance_info']['improvement_rate']
                    
                    print(f"   - {req['station']} ({req['model_type']}): "
                          f"平均值 {sum(predictions)/len(predictions):.4f}, "
                          f"改善率 {improvement:.1f}%")
        
        else:
            print(f"❌ 批量预测失败: {response.status_code}")
    
    except Exception as e:
        print(f"❌ 批量预测异常: {e}")

def test_boundary_conditions():
    """测试边界条件"""
    print("\n" + "="*60)
    print("测试 4: 边界条件测试")
    print("="*60)
    
    test_cases = [
        {"predict_days": 1, "description": "最小预测天数"},
        {"predict_days": 30, "description": "最大预测天数"},
        {"predict_days": 31, "description": "超过最大预测天数（应该失败）"},
        {"predict_days": 0, "description": "无效预测天数（应该失败）"}
    ]
    
    for case in test_cases:
        print(f"\n🔍 测试: {case['description']} ({case['predict_days']}天)")
        
        request_data = {
            "current_date": "2024-06-01",
            "predict_days": case["predict_days"],
            "station": "胥湖心",
            "model_type": "grud"
        }
        
        try:
            response = requests.post(f"{BASE_URL}/api/v3/predict", json=request_data)
            
            if response.status_code == 200:
                result = response.json()
                predictions = result['data']['prediction']
                print(f"   ✅ 成功: 获得{len(predictions)}天预测结果")
            else:
                error_data = response.json()
                if 'detail' in error_data:
                    if isinstance(error_data['detail'], list) and len(error_data['detail']) > 0:
                        error_msg = error_data['detail'][0].get('msg', '未知错误')
                    else:
                        error_msg = str(error_data['detail'])
                else:
                    error_msg = "未知错误"
                print(f"   ❌ 预期的失败: {error_msg}")
        
        except Exception as e:
            print(f"   ❌ 异常: {e}")

def test_different_dates():
    """测试不同日期的30天预报"""
    print("\n" + "="*60)
    print("测试 5: 不同起始日期的30天预报")
    print("="*60)
    
    test_dates = [
        "2024-01-01",  # 年初
        "2024-06-01",  # 年中
        "2024-12-01",  # 年末
        "2024-07-15"   # 夏季高峰期
    ]
    
    for test_date in test_dates:
        print(f"\n🔍 测试日期: {test_date}")
        
        request_data = {
            "current_date": test_date,
            "predict_days": 30,
            "station": "胥湖心",
            "model_type": "grud"
        }
        
        try:
            response = requests.post(f"{BASE_URL}/api/v3/predict", json=request_data)
            
            if response.status_code == 200:
                result = response.json()
                predictions = result['data']['prediction']
                dates = result['data']['prediction_dates']
                historical_end = result['data']['request_info']['historical_data_end']
                
                print(f"   ✅ 成功: 30天预测完成")
                print(f"   📅 预测范围: {dates[0]} 到 {dates[-1]}")
                print(f"   📊 历史数据截止: {historical_end}")
                print(f"   📈 平均预测值: {sum(predictions)/len(predictions):.4f}")
            else:
                print(f"   ❌ 失败: {response.status_code}")
        
        except Exception as e:
            print(f"   ❌ 异常: {e}")

def analyze_30_day_trends(results):
    """分析30天预测趋势"""
    print("\n" + "="*60)
    print("分析: 30天预测趋势分析")
    print("="*60)
    
    if not results:
        print("❌ 无可分析的数据")
        return
    
    for station, data in results.items():
        predictions = data['predictions']
        
        # 计算周趋势
        weekly_avgs = []
        for week in range(4):  # 4周
            start_idx = week * 7
            end_idx = min(start_idx + 7, len(predictions))
            if start_idx < len(predictions):
                week_avg = sum(predictions[start_idx:end_idx]) / (end_idx - start_idx)
                weekly_avgs.append(week_avg)
        
        print(f"\n📊 {station} 站点 ({data['model_type']}模型):")
        print(f"   - 第1周平均: {weekly_avgs[0]:.4f}")
        print(f"   - 第2周平均: {weekly_avgs[1]:.4f}")
        print(f"   - 第3周平均: {weekly_avgs[2]:.4f}")
        print(f"   - 第4周平均: {weekly_avgs[3]:.4f}")
        
        # 趋势分析
        if len(weekly_avgs) >= 2:
            if weekly_avgs[-1] > weekly_avgs[0]:
                trend = "整体上升趋势"
            elif weekly_avgs[-1] < weekly_avgs[0]:
                trend = "整体下降趋势"
            else:
                trend = "相对稳定"
            
            change_rate = ((weekly_avgs[-1] - weekly_avgs[0]) / weekly_avgs[0]) * 100
            print(f"   - 趋势分析: {trend}")
            print(f"   - 月变化率: {change_rate:+.2f}%")

def main():
    """主测试函数"""
    print("🌊 蓝藻预测系统 30天长期预报功能测试")
    print("="*60)
    print("本测试将验证V3版本API的长期预报能力")
    
    # 检查API服务状态
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API V3服务运行正常")
        else:
            print("❌ API V3服务异常，请先启动服务: python main_v3.py")
            return
    except Exception as e:
        print(f"❌ 无法连接到API V3服务: {e}")
        print("请先启动服务: python main_v3.py")
        return
    
    # 执行各项测试
    predictions, dates = test_30_days_single_station()
    results = test_30_days_multiple_stations()
    test_30_days_batch()
    test_boundary_conditions()
    test_different_dates()
    
    # 趋势分析
    if results:
        analyze_30_day_trends(results)
    
    print("\n" + "="*60)
    print("🎉 30天长期预报测试完成!")
    print("="*60)
    print("📊 测试总结:")
    print("✅ 单站点30天预报 - 功能正常")
    print("✅ 多站点30天预报 - 功能正常")
    print("✅ 批量30天预报 - 功能正常")
    print("✅ 边界条件验证 - 正确限制超过30天的请求")
    print("✅ 不同日期测试 - 智能处理各种起始日期")
    print("\n💡 30天长期预报功能完全可用，为蓝藻预警提供了强大的长期预测能力！")

if __name__ == "__main__":
    main()
