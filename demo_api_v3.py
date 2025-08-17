#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API V3 演示脚本 - 展示极简的4参数预测接口
"""

import requests
import json
from datetime import datetime, timedelta
import time

# API基础URL
BASE_URL = "http://localhost:8002"

def demo_simple_prediction():
    """演示简化预测 - 只需4个核心参数"""
    print("\n" + "="*60)
    print("演示 1: 简化预测接口 (只需4个核心参数)")
    print("="*60)
    
    # 构建请求 - 只需要4个核心参数
    request_data = {
        "current_date": "2024-06-01",  # 当前日期
        "predict_days": 7,            # 预测天数
        "station": "胥湖心",          # 预测点位
        "model_type": "grud"          # 模型类型
    }
    
    print(f"📊 请求数据 (极简版):")
    print(json.dumps(request_data, indent=2, ensure_ascii=False))
    
    print(f"\n🤖 系统自动处理:")
    print(f"   - 历史数据截止: {request_data['current_date']} 的前一天")
    print(f"   - 自动获取: 60天历史数据")
    print(f"   - 预测时间: 从 {request_data['current_date']} 开始未来 {request_data['predict_days']} 天")
    
    try:
        response = requests.post(f"{BASE_URL}/api/v3/predict", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ 预测成功!")
            print(f"📈 预测结果: {result['data']['prediction']}")
            print(f"📅 预测日期: {result['data']['prediction_dates']}")
            
            request_info = result['data']['request_info']
            print(f"\n📊 请求信息:")
            print(f"   - 当前日期: {request_info['current_date']}")
            print(f"   - 历史数据截止: {request_info['historical_data_end']}")
            print(f"   - 历史数据长度: {request_info['sequence_length']} 天")
            print(f"   - 预测天数: {request_info['predict_days']} 天")
            
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ 请求异常: {e}")

def demo_batch_simple_prediction():
    """演示批量简化预测"""
    print("\n" + "="*60)
    print("演示 2: 批量简化预测")
    print("="*60)
    
    batch_request = {
        "requests": [
            {
                "current_date": "2024-06-01",
                "predict_days": 3,
                "station": "胥湖心",
                "model_type": "grud"
            },
            {
                "current_date": "2024-06-01",
                "predict_days": 3,
                "station": "锡东水厂",
                "model_type": "grud"
            },
            {
                "current_date": "2024-06-01",
                "predict_days": 3,
                "station": "平台山",
                "model_type": "xgboost"  # 平台山用XGBoost表现更好
            }
        ],
        "parallel_execution": True
    }
    
    print(f"📊 批量预测请求: {len(batch_request['requests'])}个任务")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/v3/batch-predict", json=batch_request)
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

def demo_request_validation():
    """演示请求验证功能"""
    print("\n" + "="*60)
    print("演示 3: 简化请求验证")
    print("="*60)
    
    request_data = {
        "current_date": "2024-06-01",
        "predict_days": 14,
        "station": "胥湖心",
        "model_type": "grud"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v3/validate", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            validation = result['validation_result']
            
            print(f"📝 验证结果:")
            print(f"   - 请求有效: {'✅' if validation['valid'] else '❌'}")
            print(f"   - 自动截止日期: {validation['auto_end_date']}")
            
            if validation['warnings']:
                print(f"   - 警告信息:")
                for warning in validation['warnings']:
                    print(f"     ⚠️  {warning}")
            
            data_avail = validation['data_availability']
            print(f"   - 数据可用性: {'✅' if data_avail['available'] else '❌'}")
            if data_avail['available']:
                print(f"     📅 数据范围: {data_avail['start_date']} 到 {data_avail['end_date']}")
        
        else:
            print(f"❌ 验证失败: {response.status_code}")
    
    except Exception as e:
        print(f"❌ 验证异常: {e}")

def demo_api_comparison():
    """对比不同版本的API"""
    print("\n" + "="*60)
    print("📊 API 版本对比")
    print("="*60)
    
    print("🔴 V1版本 (复杂):")
    print("   ❌ 需要手动输入720个数值 (60天×12特征)")
    print("   ❌ JSON结构庞大，难以理解")
    print("   ❌ 用户需要处理历史数据获取")
    
    print("\n🟡 V2版本 (改进):")
    print("   ✅ 自动获取历史数据")
    print("   ❓ 仍需多个配置参数 (data_mode, end_date, seq_length等)")
    print("   ❓ 接口相对复杂")
    
    print("\n🟢 V3版本 (极简):")
    print("   ✅ 只需4个核心参数")
    print("   ✅ 零配置，全自动处理")
    print("   ✅ 实时数据自动更新")
    print("   ✅ 用户无需感知数据管理")
    
    print("\n📝 接口调用对比:")
    
    print("\n🔴 V1版本:")
    v1_example = {
        "station": "胥湖心",
        "model_type": "grud",
        "predict_days": 7,
        "input_data": {
            "temperature": 25.5,
            "oxygen": 8.2,
            # "需要720个数值..."
        }
    }
    print(json.dumps(v1_example, indent=2, ensure_ascii=False))
    
    print("\n🟡 V2版本:")
    v2_example = {
        "station": "胥湖心",
        "model_type": "grud",
        "predict_days": 7,
        "data_mode": "auto_historical",
        "end_date": "2024-05-31",
        "seq_length": 60,
        "fill_missing_method": "interpolation"
    }
    print(json.dumps(v2_example, indent=2, ensure_ascii=False))
    
    print("\n🟢 V3版本 (极简):")
    v3_example = {
        "current_date": "2024-06-01",
        "predict_days": 7,
        "station": "胥湖心",
        "model_type": "grud"
    }
    print(json.dumps(v3_example, indent=2, ensure_ascii=False))

def demo_workflow_explanation():
    """演示V3版本的工作流程"""
    print("\n" + "="*60)
    print("🔄 V3版本工作流程说明")
    print("="*60)
    
    print("📊 用户输入:")
    print("   - current_date: 2024-06-01")
    print("   - predict_days: 7")
    print("   - station: 胥湖心")
    print("   - model_type: grud")
    
    print("\n🤖 系统自动处理:")
    print("   1. 计算历史数据截止日期: 2024-05-31 (current_date - 1天)")
    print("   2. 从CSV文件获取历史数据: 2024-04-02 到 2024-05-31 (60天)")
    print("   3. 数据预处理: 缺失值填充、标准化等")
    print("   4. 加载模型: 胥湖心站点的GRU-D模型")
    print("   5. 执行预测: 预测未来7天")
    print("   6. 返回结果: 2024-06-01 到 2024-06-07的预测值")
    
    print("\n💡 实时数据更新机制:")
    print("   - generate_fake_data.py 定期运行")
    print("   - 新的实时监测数据自动追加到CSV文件")
    print("   - 用户无需感知数据更新过程")
    print("   - 系统始终使用最新的历史数据")

def main():
    """主演示函数"""
    print("🌊 蓝藻预测系统 API V3 演示")
    print("="*60)
    print("V3版本特点：只需4个核心参数，零配置，全自动")
    
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
    
    # 执行各种演示
    demo_api_comparison()
    demo_workflow_explanation()
    demo_simple_prediction()
    demo_batch_simple_prediction()
    demo_request_validation()
    
    print("\n" + "="*60)
    print("🎉 演示完成!")
    print("="*60)
    print("📚 查看完整API文档: http://localhost:8002/docs")
    print("🔧 API版本对比:")
    print("   - V1: http://localhost:8000 (原版 - 720个参数)")
    print("   - V2: http://localhost:8001 (改进版 - 自动历史数据)")
    print("   - V3: http://localhost:8002 (极简版 - 4个参数)")

if __name__ == "__main__":
    main()
