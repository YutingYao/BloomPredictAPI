#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蓝藻预测系统API V2 综合测试脚本
"""

import requests
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any
import concurrent.futures

# 测试配置
LOCAL_URL = "http://localhost:8001"
PROD_URL = "https://enfhoccrrryd.sealoshzh.site"

# 使用本地环境进行测试
BASE_URL = LOCAL_URL

# 测试数据
TEST_STATIONS = ["胥湖心", "锡东水厂", "平台山", "tuoshan", "lanshanzui", "五里湖心"]
TEST_MODELS = ["lstm", "grud", "tcn", "xgboost"]
TEST_END_DATE = "2024-05-31"

class APITester:
    """API测试类"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.test_results = {}
        self.failed_tests = []
        self.total_tests = 0
        self.passed_tests = 0
    
    def log_test(self, test_name: str, success: bool, details: str = "", response_time: float = 0):
        """记录测试结果"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            self.failed_tests.append({"test": test_name, "details": details})
        
        print(f"{status} | {test_name} | {response_time:.2f}ms | {details}")
        
        self.test_results[test_name] = {
            "success": success,
            "details": details,
            "response_time": response_time
        }
    
    def test_service_health(self):
        """测试服务健康状态"""
        print("\n" + "="*60)
        print("🏥 服务健康检查测试")
        print("="*60)
        
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_test("健康检查", True, "服务运行正常", response_time)
                else:
                    self.log_test("健康检查", False, f"服务状态异常: {data}", response_time)
            else:
                self.log_test("健康检查", False, f"HTTP {response.status_code}", response_time)
        
        except Exception as e:
            self.log_test("健康检查", False, f"异常: {e}")
    
    def test_basic_prediction(self):
        """测试基础预测功能"""
        print("\n" + "="*60)
        print("🔮 基础预测功能测试")
        print("="*60)
        
        test_cases = [
            {"station": "胥湖心", "model": "grud", "days": 7},
            {"station": "锡东水厂", "model": "grud", "days": 3},
            {"station": "平台山", "model": "xgboost", "days": 5},
        ]
        
        for case in test_cases:
            payload = {
                "station": case["station"],
                "model_type": case["model"],
                "predict_days": case["days"],
                "data_mode": "auto_historical",
                "end_date": TEST_END_DATE
            }
            
            try:
                start_time = time.time()
                response = requests.post(f"{self.base_url}/api/v2/predict", json=payload, timeout=30)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get("data", {}).get("prediction", [])
                    if len(predictions) == case["days"]:
                        self.log_test(
                            f"预测-{case['station']}-{case['model']}", 
                            True, 
                            f"预测{len(predictions)}天", 
                            response_time
                        )
                    else:
                        self.log_test(
                            f"预测-{case['station']}-{case['model']}", 
                            False, 
                            f"预测天数不匹配: {len(predictions)}/{case['days']}", 
                            response_time
                        )
                else:
                    self.log_test(
                        f"预测-{case['station']}-{case['model']}", 
                        False, 
                        f"HTTP {response.status_code}: {response.text[:100]}", 
                        response_time
                    )
            
            except Exception as e:
                self.log_test(f"预测-{case['station']}-{case['model']}", False, f"异常: {e}")
    
    def test_all_stations_models(self):
        """测试所有站点和模型组合"""
        print("\n" + "="*60)
        print("🌐 全站点全模型测试")
        print("="*60)
        
        # 每个站点使用推荐的模型
        recommended_combinations = [
            {"station": "胥湖心", "model": "grud"},
            {"station": "锡东水厂", "model": "grud"},
            {"station": "平台山", "model": "xgboost"},
            {"station": "tuoshan", "model": "grud"},
            {"station": "lanshanzui", "model": "grud"},
            {"station": "五里湖心", "model": "grud"},
        ]
        
        for combo in recommended_combinations:
            payload = {
                "station": combo["station"],
                "model_type": combo["model"],
                "predict_days": 3,
                "data_mode": "auto_historical",
                "end_date": TEST_END_DATE
            }
            
            try:
                start_time = time.time()
                response = requests.post(f"{self.base_url}/api/v2/predict", json=payload, timeout=30)
                response_time = (time.time() - start_time) * 1000
                
                success = response.status_code == 200
                details = "成功" if success else f"HTTP {response.status_code}"
                
                self.log_test(
                    f"全量测试-{combo['station']}-{combo['model']}", 
                    success, 
                    details, 
                    response_time
                )
            
            except Exception as e:
                self.log_test(f"全量测试-{combo['station']}-{combo['model']}", False, f"异常: {e}")
    
    def test_hybrid_mode(self):
        """测试混合模式"""
        print("\n" + "="*60)
        print("🔀 混合模式测试")
        print("="*60)
        
        payload = {
            "station": "胥湖心",
            "model_type": "grud",
            "predict_days": 5,
            "data_mode": "hybrid",
            "end_date": TEST_END_DATE,
            "supplementary_data": [
                {
                    "date": "2024-05-29",
                    "temperature": 26.5,
                    "oxygen": 8.2,
                    "pH": 7.8
                },
                {
                    "date": "2024-05-30",
                    "temperature": 27.1,
                    "oxygen": 7.9,
                    "pH": 7.9
                },
                {
                    "date": "2024-05-31",
                    "temperature": 28.0,
                    "oxygen": 7.5,
                    "pH": 8.0
                }
            ]
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{self.base_url}/api/v2/predict", json=payload, timeout=30)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                supplementary_points = data.get("data", {}).get("data_info", {}).get("supplementary_points", 0)
                self.log_test("混合模式", True, f"补充数据点: {supplementary_points}", response_time)
            else:
                self.log_test("混合模式", False, f"HTTP {response.status_code}", response_time)
        
        except Exception as e:
            self.log_test("混合模式", False, f"异常: {e}")
    
    def test_batch_prediction(self):
        """测试批量预测"""
        print("\n" + "="*60)
        print("📦 批量预测测试")
        print("="*60)
        
        payload = {
            "requests": [
                {
                    "station": "胥湖心",
                    "model_type": "grud",
                    "predict_days": 3,
                    "data_mode": "auto_historical",
                    "end_date": TEST_END_DATE
                },
                {
                    "station": "锡东水厂",
                    "model_type": "grud", 
                    "predict_days": 3,
                    "data_mode": "auto_historical",
                    "end_date": TEST_END_DATE
                },
                {
                    "station": "平台山",
                    "model_type": "xgboost",
                    "predict_days": 3,
                    "data_mode": "auto_historical",
                    "end_date": TEST_END_DATE
                }
            ],
            "parallel_execution": True
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{self.base_url}/api/v2/batch-predict", json=payload, timeout=60)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                summary = data.get("summary", {})
                success_count = summary.get("success", 0)
                total_count = summary.get("total", 0)
                
                if success_count == total_count:
                    self.log_test("批量预测", True, f"成功 {success_count}/{total_count}", response_time)
                else:
                    self.log_test("批量预测", False, f"部分失败 {success_count}/{total_count}", response_time)
            else:
                self.log_test("批量预测", False, f"HTTP {response.status_code}", response_time)
        
        except Exception as e:
            self.log_test("批量预测", False, f"异常: {e}")
    
    def test_data_validation(self):
        """测试数据验证功能"""
        print("\n" + "="*60)
        print("🔍 数据验证测试")
        print("="*60)
        
        # 测试请求验证
        payload = {
            "station": "胥湖心",
            "model_type": "grud",
            "predict_days": 7,
            "data_mode": "auto_historical",
            "end_date": TEST_END_DATE
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{self.base_url}/api/v2/validate-request", json=payload, timeout=15)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                validation_result = data.get("validation_result", {})
                is_valid = validation_result.get("valid", False)
                self.log_test("请求验证", is_valid, f"验证结果: {is_valid}", response_time)
            else:
                self.log_test("请求验证", False, f"HTTP {response.status_code}", response_time)
        
        except Exception as e:
            self.log_test("请求验证", False, f"异常: {e}")
    
    def test_data_info(self):
        """测试数据信息查询"""
        print("\n" + "="*60)
        print("📊 数据信息查询测试")
        print("="*60)
        
        for station in ["胥湖心", "锡东水厂", "平台山"]:
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}/api/v2/data-info/{station}", timeout=15)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    total_records = data.get("data", {}).get("total_records", 0)
                    self.log_test(f"数据信息-{station}", True, f"记录数: {total_records}", response_time)
                else:
                    self.log_test(f"数据信息-{station}", False, f"HTTP {response.status_code}", response_time)
            
            except Exception as e:
                self.log_test(f"数据信息-{station}", False, f"异常: {e}")
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n" + "="*60)
        print("🚨 错误处理测试")
        print("="*60)
        
        error_cases = [
            {
                "name": "无效站点",
                "payload": {
                    "station": "无效站点",
                    "model_type": "grud",
                    "predict_days": 7,
                    "data_mode": "auto_historical",
                    "end_date": TEST_END_DATE
                },
                "expected_status": 400
            },
            {
                "name": "无效模型",
                "payload": {
                    "station": "胥湖心",
                    "model_type": "invalid_model",
                    "predict_days": 7,
                    "data_mode": "auto_historical",
                    "end_date": TEST_END_DATE
                },
                "expected_status": 400
            },
            {
                "name": "超范围预测天数",
                "payload": {
                    "station": "胥湖心",
                    "model_type": "grud",
                    "predict_days": 50,
                    "data_mode": "auto_historical",
                    "end_date": TEST_END_DATE
                },
                "expected_status": 422
            }
        ]
        
        for case in error_cases:
            try:
                start_time = time.time()
                response = requests.post(f"{self.base_url}/api/v2/predict", json=case["payload"], timeout=15)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == case["expected_status"]:
                    self.log_test(f"错误处理-{case['name']}", True, f"正确返回 {response.status_code}", response_time)
                else:
                    self.log_test(f"错误处理-{case['name']}", False, f"期望 {case['expected_status']}, 实际 {response.status_code}", response_time)
            
            except Exception as e:
                self.log_test(f"错误处理-{case['name']}", False, f"异常: {e}")
    
    def test_performance_benchmarks(self):
        """测试性能基准"""
        print("\n" + "="*60)
        print("⚡ 性能基准测试")
        print("="*60)
        
        # 单次预测性能测试
        payload = {
            "station": "胥湖心",
            "model_type": "grud",
            "predict_days": 7,
            "data_mode": "auto_historical",
            "end_date": TEST_END_DATE
        }
        
        # 运行5次取平均值
        response_times = []
        for i in range(5):
            try:
                start_time = time.time()
                response = requests.post(f"{self.base_url}/api/v2/predict", json=payload, timeout=30)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    response_times.append(response_time)
                
            except Exception as e:
                print(f"性能测试第{i+1}次失败: {e}")
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            
            # 基准：平均响应时间应该小于10秒
            performance_ok = avg_time < 10000
            
            self.log_test(
                "性能基准", 
                performance_ok, 
                f"平均: {avg_time:.0f}ms, 最大: {max_time:.0f}ms, 最小: {min_time:.0f}ms", 
                avg_time
            )
        else:
            self.log_test("性能基准", False, "所有性能测试都失败了")
    
    def test_configuration_endpoints(self):
        """测试配置类端点"""
        print("\n" + "="*60)
        print("⚙️ 配置端点测试")
        print("="*60)
        
        endpoints = [
            {"path": "/api/stations", "name": "站点列表"},
            {"path": "/api/models", "name": "模型列表"},
            {"path": "/api/v2/input-schema", "name": "输入格式说明"},
        ]
        
        for endpoint in endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}{endpoint['path']}", timeout=10)
                response_time = (time.time() - start_time) * 1000
                
                success = response.status_code == 200
                details = "成功" if success else f"HTTP {response.status_code}"
                
                self.log_test(f"配置-{endpoint['name']}", success, details, response_time)
            
            except Exception as e:
                self.log_test(f"配置-{endpoint['name']}", False, f"异常: {e}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 蓝藻预测系统API V2 综合测试")
        print("="*60)
        print(f"测试目标: {self.base_url}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 执行所有测试
        self.test_service_health()
        self.test_configuration_endpoints()
        self.test_basic_prediction()
        self.test_hybrid_mode()
        self.test_data_validation()
        self.test_data_info()
        self.test_all_stations_models()
        self.test_batch_prediction()
        self.test_error_handling()
        self.test_performance_benchmarks()
        
        # 生成测试报告
        self.generate_test_report()
    
    def generate_test_report(self):
        """生成测试报告"""
        print("\n" + "="*60)
        print("📊 测试报告")
        print("="*60)
        
        # 统计信息
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"总测试数: {self.total_tests}")
        print(f"通过测试: {self.passed_tests}")
        print(f"失败测试: {len(self.failed_tests)}")
        print(f"成功率: {success_rate:.1f}%")
        
        # 性能统计
        response_times = [result["response_time"] for result in self.test_results.values() 
                         if result["response_time"] > 0]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            print(f"平均响应时间: {avg_response_time:.0f}ms")
            print(f"最大响应时间: {max_response_time:.0f}ms")
        
        # 失败测试详情
        if self.failed_tests:
            print(f"\n❌ 失败测试详情:")
            for failed in self.failed_tests:
                print(f"   - {failed['test']}: {failed['details']}")
        
        # 测试结论
        if success_rate >= 90:
            print(f"\n🎉 测试结论: 优秀 (成功率 {success_rate:.1f}%)")
            print("✅ API V2版本已准备好投入生产使用!")
        elif success_rate >= 70:
            print(f"\n⚠️  测试结论: 良好 (成功率 {success_rate:.1f}%)")
            print("🔧 建议修复失败的测试项后再投入使用")
        else:
            print(f"\n🚨 测试结论: 需要改进 (成功率 {success_rate:.1f}%)")
            print("❌ 建议修复主要问题后重新测试")
        
        # 保存测试结果
        self.save_test_results()
    
    def save_test_results(self):
        """保存测试结果到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"test_results/api_v2_test_results_{timestamp}.json"
        
        import os
        os.makedirs("test_results", exist_ok=True)
        
        test_summary = {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": len(self.failed_tests),
            "success_rate": (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0,
            "test_results": self.test_results,
            "failed_test_details": self.failed_tests
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(test_summary, f, indent=2, ensure_ascii=False)
            print(f"\n💾 测试结果已保存到: {filename}")
        except Exception as e:
            print(f"\n❌ 保存测试结果失败: {e}")

def main():
    """主函数"""
    print("🌊 蓝藻预测系统API V2综合测试")
    print("请确保API服务已启动在 http://localhost:8001")
    print()
    
    # 检查服务是否启动
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API服务未正常运行，请先启动服务:")
            print("   python main_v2.py")
            return
    except Exception as e:
        print(f"❌ 无法连接到API服务: {e}")
        print("请先启动服务: python main_v2.py")
        return
    
    # 创建测试器并运行测试
    tester = APITester(BASE_URL)
    tester.run_all_tests()

if __name__ == "__main__":
    main()
