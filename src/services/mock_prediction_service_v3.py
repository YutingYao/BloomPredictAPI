# -*- coding: utf-8 -*-
"""
模拟预测服务 V3 - 用于演示简化接口（生成模拟预测结果）
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

from src.schemas.response_schemas import PredictionResponse
from src.schemas.request_schemas_v3 import SimplePredictionRequest

logger = logging.getLogger(__name__)

class MockPredictionServiceV3:
    """模拟预测服务V3类 - 生成虚拟预测结果以演示接口"""
    
    def __init__(self):
        # 预定义的模拟性能数据
        self.mock_performance = {
            "胥湖心": {"grud": 77.41, "tcn": -0.84, "xgboost": 17.90, "lstm": 0.0},
            "锡东水厂": {"grud": 55.13, "tcn": 1.56, "xgboost": -37.59, "lstm": 0.0},
            "平台山": {"grud": 16.32, "tcn": -20.09, "xgboost": 100.00, "lstm": 0.0},
            "tuoshan": {"grud": 79.78, "tcn": 0.07, "xgboost": 49.31, "lstm": 0.0},
            "lanshanzui": {"grud": 28.83, "tcn": 5.35, "xgboost": 9.32, "lstm": 0.0},
            "五里湖心": {"grud": 42.93, "tcn": 2.69, "xgboost": -40.37, "lstm": 0.0}
        }
    
    async def predict_simple(self, request: SimplePredictionRequest) -> PredictionResponse:
        """
        执行模拟预测 - 生成基于算法的合理预测值
        """
        try:
            logger.info(f"开始执行模拟预测: {request.station} - {request.model_type} - {request.predict_days}天")
            
            # 生成模拟预测结果
            predictions = self._generate_mock_predictions(request)
            
            # 构建响应
            response = await self._build_mock_response(predictions, request)
            
            logger.info(f"模拟预测完成: {request.station} - {len(predictions)}个预测值")
            return response
            
        except Exception as e:
            logger.error(f"模拟预测执行失败: {e}")
            raise
    
    def _generate_mock_predictions(self, request: SimplePredictionRequest) -> List[float]:
        """生成模拟预测结果"""
        # 设置随机种子以确保结果可重现
        np.random.seed(hash(f"{request.station}_{request.model_type}_{request.current_date}") % 2**32)
        
        # 基于站点和模型类型生成不同的基础值
        base_performance = self.mock_performance.get(request.station, {}).get(request.model_type, 0.0)
        
        # 根据模型性能调整预测质量
        if base_performance > 50:
            # 高性能模型：变化较小，趋势明显
            trend = np.random.uniform(-0.1, 0.2)  # 轻微上升趋势
            noise_level = 0.05
        elif base_performance > 0:
            # 中等性能模型：中等变化
            trend = np.random.uniform(-0.2, 0.3)
            noise_level = 0.1
        else:
            # 低性能模型：变化较大
            trend = np.random.uniform(-0.3, 0.4)
            noise_level = 0.15
        
        predictions = []
        base_value = np.random.uniform(0.1, 0.5)  # 基础蓝藻密度增长率
        
        for i in range(request.predict_days):
            # 模拟时间序列的自相关性
            day_effect = trend * i + np.random.normal(0, noise_level)
            seasonal_effect = 0.1 * np.sin(2 * np.pi * i / 7)  # 周期性变化
            
            # 计算当天的预测值
            daily_prediction = base_value + day_effect + seasonal_effect
            
            # 确保值在合理范围内
            daily_prediction = np.clip(daily_prediction, -0.5, 2.0)
            predictions.append(round(daily_prediction, 4))
        
        return predictions
    
    async def _build_mock_response(
        self, 
        predictions: List[float], 
        request: SimplePredictionRequest
    ) -> PredictionResponse:
        """构建模拟响应"""
        try:
            # 构建预测时间序列
            current_dt = datetime.strptime(request.current_date, '%Y-%m-%d')
            prediction_dates = [
                (current_dt + timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range(len(predictions))
            ]
            
            # 计算历史数据截止日期
            end_date = (current_dt - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # 模拟模型信息
            model_info = {
                "lstm": {"name": "LSTM", "description": "长短期记忆网络，基准模型"},
                "grud": {"name": "GRU-D", "description": "专为缺失数据设计的GRU改进版"},
                "tcn": {"name": "TCN", "description": "时间卷积网络"},
                "xgboost": {"name": "XGBoost", "description": "梯度提升树算法"}
            }.get(request.model_type, {"name": request.model_type, "description": "未知模型"})
            
            # 构建响应数据
            response_data = {
                "prediction": predictions,
                "prediction_dates": prediction_dates,
                "model_info": {
                    "type": request.model_type,
                    "name": model_info["name"],
                    "description": model_info["description"],
                    "mode": "模拟模式 (演示用)"
                },
                "request_info": {
                    "current_date": request.current_date,
                    "historical_data_end": end_date,
                    "sequence_length": 60,
                    "predict_days": request.predict_days
                },
                "performance_info": {
                    "improvement_rate": self.mock_performance.get(request.station, {}).get(request.model_type, 0.0),
                    "note": "基于历史性能数据的模拟预测"
                }
            }
            
            return PredictionResponse(
                success=True,
                message=f"成功生成 {request.station} 从 {request.current_date} 开始未来 {request.predict_days} 天的模拟蓝藻密度预测",
                data=response_data,
                timestamp=datetime.now().isoformat(),
                request_id=f"mock_{request.station}_{request.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
        except Exception as e:
            logger.error(f"构建模拟响应失败: {e}")
            raise
    
    async def validate_simple_request(self, request: SimplePredictionRequest) -> Dict[str, Any]:
        """验证简化请求的可行性（模拟版本）"""
        try:
            validation_result = {
                "valid": True,
                "warnings": [],
                "data_availability": {},
                "auto_end_date": None,
                "mode": "模拟模式"
            }
            
            # 计算自动截止日期
            current_dt = datetime.strptime(request.current_date, '%Y-%m-%d')
            auto_end_date = (current_dt - timedelta(days=1)).strftime('%Y-%m-%d')
            validation_result["auto_end_date"] = auto_end_date
            
            # 模拟数据可用性检查
            validation_result["data_availability"] = {
                "start_date": "2021-01-01",
                "end_date": "2024-08-17",
                "available": True,
                "note": "模拟数据范围"
            }
            
            # 模拟模型可用性检查
            if request.station in self.mock_performance and request.model_type in self.mock_performance[request.station]:
                validation_result["model_available"] = True
                performance = self.mock_performance[request.station][request.model_type]
                if performance > 50:
                    validation_result["recommendation"] = f"推荐使用，改善率{performance:.1f}%"
                elif performance > 0:
                    validation_result["recommendation"] = f"可选模型，改善率{performance:.1f}%"
                else:
                    validation_result["recommendation"] = f"非推荐模型，改善率{performance:.1f}%"
            else:
                validation_result["model_available"] = False
                validation_result["warnings"].append(f"模型 {request.model_type} 在 {request.station} 不可用")
            
            validation_result["valid"] = validation_result["model_available"]
            
            return validation_result
            
        except Exception as e:
            logger.error(f"验证模拟请求失败: {e}")
            raise
