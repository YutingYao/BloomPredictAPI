#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蓝藻预测系统API V2 生产环境测试脚本
测试公网地址: https://enfhoccrrryd.sealoshzh.site
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any

# 生产环境配置
PROD_URL = "https://enfhoccrrryd.sealoshzh.site"
TEST_TIMEOUT = 60  # 生产环境可能较慢，增加超时时间

class ProductionTester:
    """生产环境测试类"""
    
    def __init__(self):
        self.base_url = PROD_URL
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
    
    def log_result(self, test_name: str, success: bool, details: str = "", response_time: float = 0):
        """记录测试结果"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            status = "✅"
        else:
            status = "❌"
        
        result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }
        
        self.test_results.append(result)
        print(f"{status} {test_name} | {response_time:.0f}ms | {details}")
    
    def test_connectivity(self):
        """测试网络连接性"""
        print("\n🌐 网络连接性测试")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/", timeout=30)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                version = data.get("version", "unknown")
                self.log_result("网络连接", True, f"API版本: {version}", response_time)
            else:
                self.log_result("网络连接", False, f"HTTP {response.status_code}", response_time)
        
        except requests.exceptions.Timeout:
            self.log_result("网络连接", False, "连接超时")
        except requests.exceptions.ConnectionError:
            self.log_result("网络连接", False, "连接失败")
        except Exception as e:
            self.log_result("网络连接", False, f"异常: {e}")
    
    def test_https_security(self):
        """测试HTTPS安全性"""
        print("\n🔒 HTTPS安全性测试")
        print("-" * 40)
        
        try:
            # 测试HTTPS连接
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health", timeout=30, verify=True)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                self.log_result("HTTPS安全", True, "SSL证书验证通过", response_time)
            else:
                self.log_result("HTTPS安全", False, f"HTTP {response.status_code}", response_time)
        
        except requests.exceptions.SSLError:
            self.log_result("HTTPS安全", False, "SSL证书验证失败")
        except Exception as e:
            self.log_result("HTTPS安全", False, f"异常: {e}")
    
    def test_cors_configuration(self):
        """测试CORS配置"""
        print("\n🌍 CORS配置测试")
        print("-" * 40)
        
        headers = {
            'Origin': 'https://example.com',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type'
        }
        
        try:
            start_time = time.time()
            response = requests.options(f"{self.base_url}/api/v2/predict", headers=headers, timeout=30)
            response_time = (time.time() - start_time) * 1000
            
            cors_headers = {
                'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
                'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
            }
            
            if any(cors_headers.values()):
                self.log_result("CORS配置", True, "CORS头部存在", response_time)
            else:
                self.log_result("CORS配置", False, "CORS头部缺失", response_time)
        
        except Exception as e:
            self.log_result("CORS配置", False, f"异常: {e}")
    
    def test_production_prediction(self):
        """测试生产环境预测功能"""
        print("\n🔮 生产环境预测测试")
        print("-" * 40)
        
        test_cases = [
            {
                "name": "胥湖心-GRUD",
                "payload": {
                    "station": "胥湖心",
                    "model_type": "grud",
                    "predict_days": 7,
                    "data_mode": "auto_historical",
                    "end_date": "2024-05-31"
                }
            },
            {
                "name": "平台山-XGBoost",
                "payload": {
                    "station": "平台山",
                    "model_type": "xgboost",
                    "predict_days": 5,
                    "data_mode": "auto_historical",
                    "end_date": "2024-05-31"
                }
            }
        ]
        
        for case in test_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/v2/predict", 
                    json=case["payload"], 
                    timeout=TEST_TIMEOUT
                )
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get("data", {}).get("prediction", [])
                    expected_days = case["payload"]["predict_days"]
                    
                    if len(predictions) == expected_days:
                        self.log_result(
                            f"预测-{case['name']}", 
                            True, 
                            f"预测{len(predictions)}天", 
                            response_time
                        )
                    else:
                        self.log_result(
                            f"预测-{case['name']}", 
                            False, 
                            f"预测天数不匹配: {len(predictions)}/{expected_days}",
                            response_time
                        )
                else:
                    self.log_result(
                        f"预测-{case['name']}", 
                        False, 
                        f"HTTP {response.status_code}",
                        response_time
                    )
            
            except requests.exceptions.Timeout:
                self.log_result(f"预测-{case['name']}", False, "请求超时")
            except Exception as e:
                self.log_result(f"预测-{case['name']}", False, f"异常: {e}")
    
    def test_load_balancing(self):
        """测试负载均衡（通过多次请求）"""
        print("\n⚖️ 负载测试")
        print("-" * 40)
        
        payload = {
            "station": "胥湖心",
            "model_type": "grud",
            "predict_days": 3,
            "data_mode": "auto_historical",
            "end_date": "2024-05-31"
        }
        
        response_times = []
        success_count = 0
        
        for i in range(5):  # 发送5个请求
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/v2/predict", 
                    json=payload, 
                    timeout=TEST_TIMEOUT
                )
                response_time = (time.time() - start_time) * 1000
                response_times.append(response_time)
                
                if response.status_code == 200:
                    success_count += 1
                
                # 短暂等待避免过快请求
                time.sleep(1)
            
            except Exception as e:
                print(f"  请求{i+1}失败: {e}")
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            
            success_rate = (success_count / 5) * 100
            
            if success_rate >= 80:
                self.log_result(
                    "负载测试", 
                    True, 
                    f"成功率: {success_rate}%, 平均: {avg_time:.0f}ms",
                    avg_time
                )
            else:
                self.log_result(
                    "负载测试", 
                    False, 
                    f"成功率过低: {success_rate}%",
                    avg_time
                )
        else:
            self.log_result("负载测试", False, "所有请求都失败")
    
    def test_api_documentation(self):
        """测试API文档可访问性"""
        print("\n📚 API文档测试")
        print("-" * 40)
        
        doc_endpoints = [
            {"path": "/docs", "name": "Swagger文档"},
            {"path": "/redoc", "name": "ReDoc文档"},
            {"path": "/api/stations", "name": "站点列表"},
            {"path": "/api/models", "name": "模型列表"}
        ]
        
        for endpoint in doc_endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}{endpoint['path']}", timeout=30)
                response_time = (time.time() - start_time) * 1000
                
                success = response.status_code == 200
                details = "可访问" if success else f"HTTP {response.status_code}"
                
                self.log_result(f"文档-{endpoint['name']}", success, details, response_time)
            
            except Exception as e:
                self.log_result(f"文档-{endpoint['name']}", False, f"异常: {e}")
    
    def test_error_handling_production(self):
        """测试生产环境错误处理"""
        print("\n🚨 生产环境错误处理测试")
        print("-" * 40)
        
        # 测试各种错误情况
        error_cases = [
            {
                "name": "无效站点",
                "payload": {
                    "station": "不存在的站点",
                    "model_type": "grud",
                    "predict_days": 7,
                    "data_mode": "auto_historical",
                    "end_date": "2024-05-31"
                },
                "expected_status_range": [400, 422]
            },
            {
                "name": "无效日期格式",
                "payload": {
                    "station": "胥湖心",
                    "model_type": "grud", 
                    "predict_days": 7,
                    "data_mode": "auto_historical",
                    "end_date": "invalid-date"
                },
                "expected_status_range": [400, 422]
            }
        ]
        
        for case in error_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/v2/predict", 
                    json=case["payload"], 
                    timeout=30
                )
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code in case["expected_status_range"]:
                    self.log_result(
                        f"错误处理-{case['name']}", 
                        True, 
                        f"正确返回 {response.status_code}",
                        response_time
                    )
                else:
                    self.log_result(
                        f"错误处理-{case['name']}", 
                        False, 
                        f"期望 {case['expected_status_range']}, 实际 {response.status_code}",
                        response_time
                    )
            
            except Exception as e:
                self.log_result(f"错误处理-{case['name']}", False, f"异常: {e}")
    
    def test_client_compatibility(self):
        """测试客户端兼容性"""
        print("\n💻 客户端兼容性测试")
        print("-" * 40)
        
        # 测试不同的Content-Type和User-Agent
        headers_tests = [
            {
                "name": "标准JSON请求",
                "headers": {"Content-Type": "application/json"}
            },
            {
                "name": "浏览器请求",
                "headers": {
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            },
            {
                "name": "移动端请求",
                "headers": {
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X)"
                }
            }
        ]
        
        payload = {
            "station": "胥湖心",
            "model_type": "grud",
            "predict_days": 3,
            "data_mode": "auto_historical",
            "end_date": "2024-05-31"
        }
        
        for test in headers_tests:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/v2/predict",
                    json=payload,
                    headers=test["headers"],
                    timeout=TEST_TIMEOUT
                )
                response_time = (time.time() - start_time) * 1000
                
                success = response.status_code == 200
                details = "兼容" if success else f"HTTP {response.status_code}"
                
                self.log_result(f"兼容性-{test['name']}", success, details, response_time)
            
            except Exception as e:
                self.log_result(f"兼容性-{test['name']}", False, f"异常: {e}")
    
    def generate_production_report(self):
        """生成生产环境测试报告"""
        print("\n" + "="*60)
        print("📊 生产环境测试报告")
        print("="*60)
        
        # 统计信息
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"🌐 测试环境: {self.base_url}")
        print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 总测试数: {self.total_tests}")
        print(f"✅ 通过测试: {self.passed_tests}")
        print(f"❌ 失败测试: {self.total_tests - self.passed_tests}")
        print(f"📈 成功率: {success_rate:.1f}%")
        
        # 性能统计
        response_times = [r["response_time"] for r in self.test_results if r["response_time"] > 0]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            print(f"⚡ 平均响应时间: {avg_time:.0f}ms")
            print(f"⚡ 最大响应时间: {max_time:.0f}ms")
            print(f"⚡ 最小响应时间: {min_time:.0f}ms")
        
        # 失败测试详情
        failed_tests = [r for r in self.test_results if not r["success"]]
        if failed_tests:
            print(f"\n❌ 失败测试详情:")
            for failed in failed_tests:
                print(f"   - {failed['test_name']}: {failed['details']}")
        
        # 生产就绪评估
        print(f"\n🏭 生产就绪评估:")
        if success_rate >= 95:
            print("🟢 优秀 - 生产环境运行良好")
        elif success_rate >= 85:
            print("🟡 良好 - 建议关注失败项")
        elif success_rate >= 70:
            print("🟠 一般 - 需要修复部分问题")
        else:
            print("🔴 较差 - 需要重大修复")
        
        # 保存测试报告
        self.save_production_report()
        
        # 用户调用指南
        if success_rate >= 80:
            self.show_usage_guide()
    
    def save_production_report(self):
        """保存生产环境测试报告"""
        import os
        os.makedirs("test_results", exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"test_results/production_test_report_{timestamp}.json"
        
        report = {
            "test_environment": self.base_url,
            "test_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.total_tests - self.passed_tests,
                "success_rate": (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
            },
            "detailed_results": self.test_results
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n💾 生产环境测试报告已保存: {filename}")
        except Exception as e:
            print(f"\n❌ 保存报告失败: {e}")
    
    def show_usage_guide(self):
        """显示用户调用指南"""
        print(f"\n" + "="*60)
        print("🚀 API调用指南")
        print("="*60)
        
        print(f"📡 API地址: {self.base_url}")
        print(f"📚 API文档: {self.base_url}/docs")
        
        print(f"\n💡 基础调用示例:")
        print(f"```bash")
        print(f"curl -X POST '{self.base_url}/api/v2/predict' \\")
        print(f"  -H 'Content-Type: application/json' \\")
        print(f"  -d '{{")
        print(f"    \"station\": \"胥湖心\",")
        print(f"    \"model_type\": \"grud\",")
        print(f"    \"predict_days\": 7,")
        print(f"    \"data_mode\": \"auto_historical\",")
        print(f"    \"end_date\": \"2024-05-31\"")
        print(f"  }}'")
        print(f"```")
        
        print(f"\n🐍 Python调用示例:")
        print(f"```python")
        print(f"import requests")
        print(f"")
        print(f"url = '{self.base_url}/api/v2/predict'")
        print(f"payload = {{")
        print(f"    'station': '胥湖心',")
        print(f"    'model_type': 'grud',")
        print(f"    'predict_days': 7,")
        print(f"    'data_mode': 'auto_historical',")
        print(f"    'end_date': '2024-05-31'")
        print(f"}}")
        print(f"")
        print(f"response = requests.post(url, json=payload)")
        print(f"result = response.json()")
        print(f"```")
    
    def run_production_tests(self):
        """运行所有生产环境测试"""
        print("🌊 蓝藻预测系统API V2 生产环境测试")
        print("="*60)
        print(f"🌐 测试目标: {self.base_url}")
        print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 执行测试
        self.test_connectivity()
        self.test_https_security()
        self.test_cors_configuration()
        self.test_api_documentation()
        self.test_production_prediction()
        self.test_load_balancing()
        self.test_error_handling_production()
        self.test_client_compatibility()
        
        # 生成报告
        self.generate_production_report()

def main():
    """主函数"""
    print("🌐 蓝藻预测系统API V2 生产环境测试")
    print(f"目标地址: {PROD_URL}")
    print()
    
    # 创建测试器并运行测试
    tester = ProductionTester()
    tester.run_production_tests()

if __name__ == "__main__":
    main()
