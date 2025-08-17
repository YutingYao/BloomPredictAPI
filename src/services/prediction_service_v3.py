# -*- coding: utf-8 -*-
"""
简化的预测服务 V3 - 只需4个核心参数
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

from src.models.model_manager import ModelManager
from src.services.historical_data_service import HistoricalDataService
from src.utils.data_processor import DataProcessor
from src.utils.validators import get_model_display_info, get_station_display_name
from src.schemas.response_schemas import PredictionResponse
from src.schemas.request_schemas_v3 import SimplePredictionRequest
from src.config.settings import Settings

logger = logging.getLogger(__name__)

class PredictionServiceV3:
    """简化的预测服务V3类 - 只需4个核心参数"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.historical_data_service = HistoricalDataService()
        self.data_processor = DataProcessor()
        self.settings = Settings()
    
    async def predict_simple(self, request: SimplePredictionRequest) -> PredictionResponse:
        """
        执行简化预测 - 只需4个核心参数
        
        逻辑：
        1. 根据current_date自动确定历史数据截止日期
        2. 从CSV文件自动获取60天历史数据
        3. 执行模型预测
        4. 返回结果
        """
        try:
            logger.info(f"开始执行简化预测: {request.station} - {request.model_type} - {request.predict_days}天")
            
            # 第一步：自动确定历史数据截止日期
            # 使用current_date的前一天作为历史数据的截止日期
            current_dt = datetime.strptime(request.current_date, '%Y-%m-%d')
            end_date = (current_dt - timedelta(days=1)).strftime('%Y-%m-%d')
            
            logger.info(f"自动确定历史数据截止日期: {end_date}")
            
            # 第二步：自动获取历史数据序列（固定60天）
            input_sequence = await self.historical_data_service.get_historical_sequence(
                station=request.station,
                end_date=end_date,
                seq_length=60,  # 固定使用60天
                fill_missing_method="interpolation"  # 默认插值填充
            )
            
            # 第三步：数据预处理
            processed_input = await self._preprocess_input(input_sequence, request.station)
            
            # 第四步：执行模型预测
            predictions = await self._execute_prediction(
                processed_input, request.station, request.model_type, request.predict_days
            )
            
            # 第五步：构建响应
            response = await self._build_simple_response(predictions, request, end_date)
            
            logger.info(f"简化预测完成: {request.station} - {len(predictions)}个预测值")
            return response
            
        except Exception as e:
            logger.error(f"简化预测执行失败: {e}")
            raise
    
    async def _preprocess_input(self, input_sequence: np.ndarray, station: str) -> np.ndarray:
        """预处理输入数据"""
        try:
            # 如果输入序列已经是正确的格式，直接返回
            if input_sequence.ndim == 3:
                logger.debug(f"输入数据预处理完成: {input_sequence.shape}")
                return input_sequence
            
            # 否则使用数据处理器创建序列数据
            loop = asyncio.get_event_loop()
            processed_data = await loop.run_in_executor(
                None, self.data_processor.create_sequence_data, input_sequence
            )
            
            logger.debug(f"输入数据预处理完成: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            logger.error(f"预处理输入数据失败: {e}")
            raise
    
    async def _execute_prediction(
        self, 
        processed_input: np.ndarray, 
        station: str, 
        model_type: str, 
        predict_days: int
    ) -> List[float]:
        """执行模型预测"""
        try:
            # 获取模型
            model = await self.model_manager.get_model(station, model_type, predict_days)
            if model is None:
                raise ValueError(f"无法获取模型: {station} - {model_type}")
            
            # 执行预测
            predictions = await self.model_manager.predict(
                model=model,
                input_data=processed_input,
                predict_days=predict_days,
                model_type=model_type
            )
            
            return predictions
            
        except Exception as e:
            logger.error(f"执行模型预测失败: {e}")
            raise
    
    async def _build_simple_response(
        self, 
        predictions: List[float], 
        request: SimplePredictionRequest,
        end_date: str
    ) -> PredictionResponse:
        """构建简化响应"""
        try:
            # 获取模型和站点显示信息
            model_info = get_model_display_info(request.model_type)
            station_display = get_station_display_name(request.station)
            
            # 构建预测时间序列
            current_dt = datetime.strptime(request.current_date, '%Y-%m-%d')
            prediction_dates = [
                (current_dt + timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range(len(predictions))
            ]
            
            # 构建响应数据
            response_data = {
                "prediction": predictions,
                "prediction_dates": prediction_dates,
                "model_info": {
                    "type": request.model_type,
                    "name": model_info["name"],
                    "description": model_info["description"]
                },
                "request_info": {
                    "current_date": request.current_date,
                    "historical_data_end": end_date,
                    "sequence_length": 60,
                    "predict_days": request.predict_days
                }
            }
            
            return PredictionResponse(
                success=True,
                message=f"成功预测 {station_display} 从 {request.current_date} 开始未来 {request.predict_days} 天的蓝藻密度变化",
                data=response_data,
                timestamp=datetime.now().isoformat(),
                request_id=f"{request.station}_{request.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
        except Exception as e:
            logger.error(f"构建简化响应失败: {e}")
            raise
    
    async def validate_simple_request(self, request: SimplePredictionRequest) -> Dict[str, Any]:
        """验证简化请求的可行性"""
        try:
            validation_result = {
                "valid": True,
                "warnings": [],
                "data_availability": {},
                "auto_end_date": None
            }
            
            # 计算自动截止日期
            current_dt = datetime.strptime(request.current_date, '%Y-%m-%d')
            auto_end_date = (current_dt - timedelta(days=1)).strftime('%Y-%m-%d')
            validation_result["auto_end_date"] = auto_end_date
            
            # 检查数据可用性
            try:
                min_date, max_date = await self.historical_data_service.get_available_date_range(request.station)
                validation_result["data_availability"] = {
                    "start_date": min_date,
                    "end_date": max_date,
                    "available": True
                }
                
                # 检查自动截止日期是否在可用范围内
                if auto_end_date > max_date:
                    validation_result["warnings"].append(f"自动截止日期 {auto_end_date} 超出数据范围 {max_date}")
                
            except Exception as e:
                validation_result["data_availability"]["available"] = False
                validation_result["warnings"].append(f"无法获取数据可用性: {str(e)}")
            
            # 检查模型可用性
            try:
                model = await self.model_manager.get_model(request.station, request.model_type, request.predict_days)
                if model is None:
                    validation_result["warnings"].append(f"模型 {request.model_type} 在 {request.station} 不可用")
            except Exception as e:
                validation_result["warnings"].append(f"模型检查失败: {str(e)}")
            
            validation_result["valid"] = len([w for w in validation_result["warnings"] if "不可用" in w]) == 0
            
            return validation_result
            
        except Exception as e:
            logger.error(f"验证简化请求失败: {e}")
            raise
