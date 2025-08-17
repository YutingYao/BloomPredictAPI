#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API V2 演示脚本 - 展示优化后的历史数据处理功能
"""

import requests
import json
from datetime import datetime, timedelta
import time

# API基础URL
BASE_URL = "http://localhost:8001"

def demo_auto_historical_mode():
    """演示自动历史数据模式"""
    print("\n" + "="*60)
    print("演示 1: 自动历史数据模式 (推荐)")
    print("="*60)
    
    # 构建请求 - 只需要5个核心参数
    request_data = {
        "station": "胥湖心",
        "model_type": "grud",
        "predict_days": 7,
        "data_mode": "auto_historical",
        "end_date": "2024-05-31",  # 使用数据集中的最后日期
        "seq_length": 60,
        "fill_missing_method": "interpolation",
        "validate_data_quality": True
    }
    
    print(f"📊 请求数据:")
    print(json.dumps(request_data, indent=2, ensure_ascii=False))
    
    try:
        response = requests.post(f"{BASE_URL}/api/v2/predict", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ 预测成功!")
            print(f"📈 预测结果: {len(result['data']['prediction'])}个预测值")
            print(f"📅 预测日期范围: {result['data']['prediction_dates'][0]} 到 {result['data']['prediction_dates'][-1]}")
            
            if 'quality_report' in result['data']:
                quality = result['data']['quality_report']
                print(f"🔍 数据质量分数: {quality['score']:.3f}")
                if quality['warnings']:
                    print(f"⚠️  数据质量警告: {quality['warnings']}")
            
            print(f"📊 输入数据统计:")
            stats = result['data']['input_stats']
            print(f"   - 平均温度: {stats['mean_temperature']:.1f}°C")
            print(f"   - 平均溶氧: {stats['mean_oxygen']:.1f}mg/L")
            print(f"   - 数据覆盖率: {stats['data_coverage']:.1%}")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ 请求异常: {e}")

def demo_hybrid_mode():
    """演示混合模式 - 历史数据+实时补充"""
    print("\n" + "="*60)
    print("演示 2: 混合模式 - 历史数据 + 实时补充")
    print("="*60)
    
    request_data = {
        "station": "胥湖心", 
        "model_type": "grud",
        "predict_days": 5,
        "data_mode": "hybrid",
        "end_date": "2024-05-31",
        "seq_length": 60,
        "override_recent_days": 3,
        "supplementary_data": [
            {
                "date": "2024-05-29",
                "temperature": 26.5,
                "oxygen": 8.2,
                "pH": 7.8,
                "TN": 1.2,
                "TP": 0.08
            },
            {
                "date": "2024-05-30",
                "temperature": 27.1,
                "oxygen": 7.9,
                "pH": 7.9,
                "TN": 1.3,
                "TP": 0.09
            },
            {
                "date": "2024-05-31",
                "temperature": 28.0,
                "oxygen": 7.5,
                "pH": 8.0,
                "TN": 1.4,
                "TP": 0.10
            }
        ],
        "fill_missing_method": "interpolation"
    }
    
    print(f"📊 请求数据 (混合模式):")
    print(f"   - 基础历史数据: 60天")
    print(f"   - 覆盖最近: 3天")
    print(f"   - 补充数据点: {len(request_data['supplementary_data'])}个")
    
    try:
        response = requests.post(f"{BASE_URL}/api/v2/predict", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ 混合模式预测成功!")
            print(f"📈 预测结果: {result['data']['prediction']}")
            print(f"📅 预测日期: {result['data']['prediction_dates']}")
            
            data_info = result['data']['data_info']
            print(f"📊 数据信息:")
            print(f"   - 数据模式: {data_info['mode']}")
            print(f"   - 序列长度: {data_info['sequence_length']}")
            print(f"   - 补充数据点: {data_info['supplementary_points']}")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ 请求异常: {e}")

def demo_request_validation():
    """演示请求验证功能"""
    print("\n" + "="*60)
    print("演示 3: 请求验证功能")
    print("="*60)
    
    request_data = {
        "station": "胥湖心",
        "model_type": "grud", 
        "predict_days": 14,
        "data_mode": "auto_historical",
        "end_date": "2024-05-31"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v2/validate-request", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            validation = result['validation_result']
            
            print(f"📝 验证结果:")
            print(f"   - 请求有效: {'✅' if validation['valid'] else '❌'}")
            
            if validation['warnings']:
                print(f"   - 警告信息:")
                for warning in validation['warnings']:
                    print(f"     ⚠️  {warning}")
            
            if validation['recommendations']:
                print(f"   - 推荐模型:")
                for rec in validation['recommendations']:
                    print(f"     💡 {rec['model']}: {rec['improvement']} ({rec['reason']})")
            
            data_avail = validation['data_availability']
            print(f"   - 数据可用性: {'✅' if data_avail['available'] else '❌'}")
            if data_avail['available']:
                print(f"     📅 数据范围: {data_avail['start_date']} 到 {data_avail['end_date']}")
        
        else:
            print(f"❌ 验证失败: {response.status_code}")
    
    except Exception as e:
        print(f"❌ 验证异常: {e}")

def demo_data_info():
    """演示数据信息查询"""
    print("\n" + "="*60)
    print("演示 4: 数据信息查询")
    print("="*60)
    
    station = "胥湖心"
    
    try:
        response = requests.get(f"{BASE_URL}/api/v2/data-info/{station}")
        
        if response.status_code == 200:
            result = response.json()
            data = result['data']
            
            print(f"📊 {station} 站点数据摘要:")
            print(f"   - 总记录数: {data['total_records']}")
            print(f"   - 数据范围: {data['date_range']['start']} 到 {data['date_range']['end']}")
            print(f"   - 覆盖天数: {data['date_range']['days']}天")
            print(f"   - 整体数据覆盖率: {data['data_coverage']['overall_coverage']:.1%}")
            
            print(f"\n🔍 关键特征缺失情况:")
            missing_summary = data['missing_data_summary']
            for feature in ['temperature', 'oxygen', 'pH', 'TN', 'TP']:
                if feature in missing_summary:
                    missing = missing_summary[feature]
                    print(f"   - {feature}: {missing['missing_ratio']:.1%} 缺失")
        
        else:
            print(f"❌ 查询失败: {response.status_code}")
    
    except Exception as e:
        print(f"❌ 查询异常: {e}")

def demo_batch_prediction():
    """演示批量预测"""
    print("\n" + "="*60)
    print("演示 5: 批量预测功能")
    print("="*60)
    
    batch_request = {
        "requests": [
            {
                "station": "胥湖心",
                "model_type": "grud",
                "predict_days": 3,
                "data_mode": "auto_historical",
                "end_date": "2024-05-31"
            },
            {
                "station": "锡东水厂", 
                "model_type": "grud",
                "predict_days": 3,
                "data_mode": "auto_historical",
                "end_date": "2024-05-31"
            },
            {
                "station": "平台山",
                "model_type": "xgboost",  # 平台山用XGBoost表现更好
                "predict_days": 3,
                "data_mode": "auto_historical",
                "end_date": "2024-05-31"
            }
        ],
        "parallel_execution": True,
        "max_workers": 3
    }
    
    print(f"📊 批量预测请求: {len(batch_request['requests'])}个任务")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/v2/batch-predict", json=batch_request)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            summary = result['summary']
            
            print(f"\n✅ 批量预测完成!")
            print(f"   - 总任务: {summary['total']}")
            print(f"   - 成功: {summary['success']}")
            print(f"   - 失败: {summary['errors']}")
            print(f"   - 耗时: {end_time - start_time:.2f}秒")
            
            print(f"\n📈 各站点预测结果:")
            for i, req in enumerate(batch_request['requests']):
                if i < len(result['results']) and isinstance(result['results'][i], dict) and 'data' in result['results'][i]:
                    pred_result = result['results'][i]
                    print(f"   - {req['station']}: {pred_result['data']['prediction']}")
        
        else:
            print(f"❌ 批量预测失败: {response.status_code}")
    
    except Exception as e:
        print(f"❌ 批量预测异常: {e}")

def compare_v1_vs_v2():
    """对比V1和V2版本的差异"""
    print("\n" + "="*60)
    print("📊 API V1 vs V2 对比")
    print("="*60)
    
    print("🔧 V1版本 (当前问题):")
    print("   ❌ 需要手动输入720个数值 (60天×12特征)")
    print("   ❌ JSON结构庞大，可读性差")
    print("   ❌ 用户难以构造历史数据")
    print("   ❌ 不符合实际业务需求")
    
    print("\n✨ V2版本 (优化后):")
    print("   ✅ 智能历史数据获取，只需5个核心参数")
    print("   ✅ 支持混合模式：历史数据+实时补充")
    print("   ✅ 内置数据质量验证")
    print("   ✅ 灵活的缺失值处理")
    print("   ✅ 批量预测支持")
    print("   ✅ 请求验证和推荐")
    
    print("\n📝 API调用示例对比:")
    
    print("\n🔴 V1版本需要的输入 (部分展示):")
    v1_example = {
        "station": "胥湖心",
        "model_type": "grud",
        "predict_days": 7,
        "input_data": {
            # 只能输入1天的数据，但模型实际需要60天
            "temperature": 25.5,
            "oxygen": 8.2,
            # ... 其他10个参数
        }
    }
    print(json.dumps(v1_example, indent=2, ensure_ascii=False))
    print("   ⚠️  问题：模型需要60天数据，但只提供了1天!")
    
    print("\n🟢 V2版本的输入:")
    v2_example = {
        "station": "胥湖心",
        "model_type": "grud", 
        "predict_days": 7,
        "data_mode": "auto_historical",
        "end_date": "2024-05-31"
    }
    print(json.dumps(v2_example, indent=2, ensure_ascii=False))
    print("   ✅ 系统自动获取2024-04-02到2024-05-31的60天历史数据!")

def main():
    """主演示函数"""
    print("🌊 蓝藻预测系统 API V2 演示")
    print("="*60)
    print("本演示将展示优化后的API如何解决历史数据输入问题")
    
    # 检查API服务状态
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API服务运行正常")
        else:
            print("❌ API服务异常，请先启动服务: python main_v2.py")
            return
    except Exception as e:
        print(f"❌ 无法连接到API服务: {e}")
        print("请先启动服务: python main_v2.py")
        return
    
    # 执行各种演示
    compare_v1_vs_v2()
    demo_auto_historical_mode()
    demo_hybrid_mode() 
    demo_request_validation()
    demo_data_info()
    demo_batch_prediction()
    
    print("\n" + "="*60)
    print("🎉 演示完成!")
    print("="*60)
    print("📚 查看完整API文档: http://localhost:8001/docs")
    print("🔧 API版本对比:")
    print("   - V1: http://localhost:8000 (原版)")
    print("   - V2: http://localhost:8001 (优化版)")

if __name__ == "__main__":
    main()
