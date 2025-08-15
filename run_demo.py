#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蓝藻预测API演示脚本 - 模拟API功能展示
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any

class MockPredictionAPI:
    """模拟蓝藻预测API类"""
    
    def __init__(self):
        self.stations = [
            {"name": "胥湖心", "name_en": "Xuhu Center", "zone": "重污染与高风险区"},
            {"name": "锡东水厂", "name_en": "Xidong Water Plant", "zone": "重污染与高风险区"},
            {"name": "平台山", "name_en": "Pingtai Mountain", "zone": "背景与参照区"},
            {"name": "tuoshan", "name_en": "Tuoshan Mountain", "zone": "背景与参照区"},
            {"name": "lanshanzui", "name_en": "Lanshan Cape", "zone": "边界条件区"},
            {"name": "五里湖心", "name_en": "Wulihu Center", "zone": "边界条件区"}
        ]
        
        self.models = {
            "lstm": {"name": "LSTM", "improvement": 0.0},
            "grud": {"name": "GRU-D", "improvement": 50.07},
            "tcn": {"name": "TCN", "improvement": -1.88},
            "xgboost": {"name": "XGBoost", "improvement": 16.43}
        }
        
        # 模拟性能数据
        self.performance_data = {
            "胥湖心": {"grud": 77.41, "tcn": -0.84, "xgboost": 17.90},
            "锡东水厂": {"grud": 55.13, "tcn": 1.56, "xgboost": -37.59},
            "平台山": {"grud": 16.32, "tcn": -20.09, "xgboost": 100.0},
            "tuoshan": {"grud": 79.78, "tcn": 0.07, "xgboost": 49.31},
            "lanshanzui": {"grud": 28.83, "tcn": 5.35, "xgboost": 9.32},
            "五里湖心": {"grud": 42.93, "tcn": 2.69, "xgboost": -40.37}
        }
    
    def predict(self, station: str, model_type: str, predict_days: int, input_data: Dict) -> Dict:
        """模拟预测功能"""
        print(f"🔮 正在使用{model_type.upper()}模型预测{station}站点未来{predict_days}天的蓝藻密度...")
        time.sleep(1)  # 模拟计算时间
        
        # 生成模拟预测结果
        import random
        random.seed(42)  # 确保结果可重现
        
        base_value = 0.1
        predictions = []
        for i in range(predict_days):
            # 模拟增长率变化
            variation = random.uniform(-0.05, 0.1)
            value = base_value + variation + i * 0.02
            predictions.append(round(value, 4))
        
        # 计算置信度
        improvement = self.performance_data.get(station, {}).get(model_type, 0)
        confidence = min(0.95, max(0.3, (improvement + 50) / 150))
        
        return {
            "success": True,
            "data": {
                "station": station,
                "model_type": model_type,
                "predict_days": predict_days,
                "prediction": predictions,
                "confidence": round(confidence, 3),
                "rmse": round(random.uniform(0.1, 2.0), 4),
                "model_info": {
                    "name": self.models[model_type]["name"],
                    "improvement_over_lstm": f"{improvement:.2f}%"
                }
            },
            "message": "预测成功",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_performance(self, station: str = None) -> Dict:
        """获取模型性能对比"""
        print(f"📊 获取{'全部站点' if not station else station}的模型性能对比数据...")
        time.sleep(0.5)
        
        target_stations = [station] if station else [s["name"] for s in self.stations]
        
        performance_list = []
        for station_name in target_stations:
            if station_name in self.performance_data:
                station_info = next(s for s in self.stations if s["name"] == station_name)
                models_perf = []
                
                for model_type, improvement in self.performance_data[station_name].items():
                    models_perf.append({
                        "model_type": model_type,
                        "model_name": self.models[model_type]["name"],
                        "improvement_over_lstm": improvement,
                        "status": "优秀" if improvement > 30 else "良好" if improvement > 0 else "一般"
                    })
                
                performance_list.append({
                    "station": station_name,
                    "zone": station_info["zone"],
                    "models": models_perf
                })
        
        return {
            "success": True,
            "data": {
                "station_performance": performance_list,
                "summary": {
                    "best_model": "grud",
                    "overall_improvement": 33.6,
                    "recommended_model": "grud"
                }
            },
            "message": "性能数据获取成功"
        }

def demo_prediction():
    """演示预测功能"""
    print("=" * 60)
    print("🌊 蓝藻预测系统API演示")
    print("=" * 60)
    
    api = MockPredictionAPI()
    
    # 展示支持的站点
    print("\n📍 支持的监测站点:")
    for i, station in enumerate(api.stations, 1):
        print(f"  {i}. {station['name']} ({station['name_en']}) - {station['zone']}")
    
    # 展示支持的模型
    print("\n🤖 支持的预测模型:")
    for model_type, info in api.models.items():
        recommended = " ⭐" if model_type == "grud" else ""
        print(f"  - {info['name']} ({model_type.upper()}): 平均改善率 {info['improvement']:.2f}%{recommended}")
    
    print("\n" + "=" * 60)
    print("🧪 预测演示")
    print("=" * 60)
    
    # 演示预测
    test_cases = [
        {"station": "胥湖心", "model": "grud", "days": 7},
        {"station": "平台山", "model": "xgboost", "days": 3},
        {"station": "tuoshan", "model": "grud", "days": 14}
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔬 测试案例 {i}: {case['station']} - {case['model'].upper()}模型 - {case['days']}天预测")
        
        # 模拟输入数据
        input_data = {
            "temperature": 25.5,
            "oxygen": 8.2,
            "TN": 1.5,
            "TP": 0.08,
            "pH": 7.8,
            "turbidity": 15.2
        }
        
        print("  输入数据:")
        for key, value in input_data.items():
            print(f"    {key}: {value}")
        
        # 执行预测
        result = api.predict(case["station"], case["model"], case["days"], input_data)
        
        if result["success"]:
            data = result["data"]
            print(f"  ✅ 预测成功!")
            print(f"    置信度: {data['confidence']:.1%}")
            print(f"    模型改善率: {data['model_info']['improvement_over_lstm']}")
            predictions = data['prediction']
            print(f"    预测结果: {predictions[:3]}{'...' if len(predictions) > 3 else ''}")
        else:
            print(f"  ❌ 预测失败: {result.get('message', '未知错误')}")

def demo_performance():
    """演示性能对比功能"""
    print("\n" + "=" * 60)
    print("📈 模型性能对比演示")
    print("=" * 60)
    
    api = MockPredictionAPI()
    
    # 获取性能数据
    result = api.get_performance()
    
    if result["success"]:
        data = result["data"]
        print("\n🏆 各站点最佳模型表现:")
        
        for station_perf in data["station_performance"]:
            station = station_perf["station"]
            zone = station_perf["zone"]
            models = station_perf["models"]
            
            # 找出最佳模型
            best_model = max(models, key=lambda x: x["improvement_over_lstm"])
            
            print(f"\n  📍 {station} ({zone}):")
            print(f"    🥇 最佳模型: {best_model['model_name']} (改善率: {best_model['improvement_over_lstm']:.2f}%)")
            
            # 显示所有模型表现
            for model in sorted(models, key=lambda x: x["improvement_over_lstm"], reverse=True):
                status_icon = "🟢" if model["status"] == "优秀" else "🟡" if model["status"] == "良好" else "🔴"
                print(f"      {status_icon} {model['model_name']}: {model['improvement_over_lstm']:+.2f}%")
        
        # 显示总结
        summary = data["summary"]
        print(f"\n🎯 总体分析:")
        print(f"  • 推荐模型: {summary['recommended_model'].upper()}")
        print(f"  • 平均改善率: {summary['overall_improvement']:.2f}%")
        print(f"  • 最佳模型: {summary['best_model'].upper()}")

def demo_api_endpoints():
    """演示API接口"""
    print("\n" + "=" * 60)
    print("🔗 API接口演示")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    endpoints = [
        {
            "method": "GET",
            "path": "/health",
            "description": "健康检查",
            "example": f"curl {base_url}/health"
        },
        {
            "method": "GET", 
            "path": "/api/stations",
            "description": "获取支持的监测站点",
            "example": f"curl {base_url}/api/stations"
        },
        {
            "method": "GET",
            "path": "/api/models", 
            "description": "获取支持的预测模型",
            "example": f"curl {base_url}/api/models"
        },
        {
            "method": "POST",
            "path": "/api/predict",
            "description": "蓝藻密度预测",
            "example": f"""curl -X POST {base_url}/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"station": "胥湖心", "model_type": "grud", "predict_days": 7, "input_data": {{"temperature": 25.5, "oxygen": 8.2, "TN": 1.5, "TP": 0.08}}}}'"""
        },
        {
            "method": "POST",
            "path": "/api/model-performance",
            "description": "模型性能对比",
            "example": f"""curl -X POST {base_url}/api/model-performance \\
  -H "Content-Type: application/json" \\
  -d '{{"station": "胥湖心", "model_types": ["lstm", "grud"]}}'"""
        }
    ]
    
    print("\n📋 可用的API接口:")
    for endpoint in endpoints:
        print(f"\n  🔸 {endpoint['method']} {endpoint['path']}")
        print(f"    描述: {endpoint['description']}")
        print(f"    示例: {endpoint['example'][:80]}{'...' if len(endpoint['example']) > 80 else ''}")
    
    print(f"\n📚 完整API文档: {base_url}/docs")
    print(f"🔍 Redoc文档: {base_url}/redoc")

def main():
    """主演示函数"""
    try:
        # 预测功能演示
        demo_prediction()
        
        # 性能对比演示
        demo_performance()
        
        # API接口演示
        demo_api_endpoints()
        
        print("\n" + "=" * 60)
        print("🎉 演示完成!")
        print("=" * 60)
        print("\n要启动真实的API服务器，请运行:")
        print("  python start_server.py")
        print("\n或者:")
        print("  python main.py")
        print("\n然后访问 http://localhost:8000/docs 查看完整API文档")
        
    except KeyboardInterrupt:
        print("\n\n👋 演示已停止")

if __name__ == "__main__":
    main()
