# -*- coding: utf-8 -*-
"""
预测服务 V2 - 集成历史数据处理的优化版本
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

from src.models.model_manager import ModelManager
from src.services.historical_data_service import HistoricalDataService
from src.utils.data_processor import DataProcessor
from src.utils.validators import get_model_display_info, get_station_display_name
from src.schemas.response_schemas import PredictionResponse, ModelPerformanceResponse
from src.schemas.request_schemas_v2 import PredictionRequestV2, SupplementaryDataPoint
from src.config.settings import Settings

logger = logging.getLogger(__name__)

class PredictionServiceV2:
    """预测服务V2类 - 支持优化的历史数据处理"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.historical_data_service = HistoricalDataService()
        self.data_processor = DataProcessor()
        self.settings = Settings()
        
        # 预定义的模型性能数据
        self.performance_data = self._load_performance_data()
    
    def _load_performance_data(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """加载预定义的模型性能数据"""
        return {
            "胥湖心": {
                "grud": {"improvement": 77.41, "rmse": 0.4379, "best_day": 25, "worst_day": 1},
                "tcn": {"improvement": -0.84, "rmse": 1.7755, "best_day": 11, "worst_day": 4},
                "xgboost": {"improvement": 17.90, "rmse": 1.3913, "best_day": 25, "worst_day": 4},
                "lstm": {"improvement": 0.0, "rmse": 1.6903, "best_day": 1, "worst_day": 30}
            },
            "锡东水厂": {
                "grud": {"improvement": 55.13, "rmse": 0.7094, "best_day": 30, "worst_day": 1},
                "tcn": {"improvement": 1.56, "rmse": 1.7335, "best_day": 20, "worst_day": 28},
                "xgboost": {"improvement": -37.59, "rmse": 2.4424, "best_day": 2, "worst_day": 24},
                "lstm": {"improvement": 0.0, "rmse": 1.8185, "best_day": 1, "worst_day": 30}
            },
            "平台山": {
                "grud": {"improvement": 16.32, "rmse": 2103.4, "best_day": 18, "worst_day": 11},
                "tcn": {"improvement": -20.09, "rmse": 2419.2, "best_day": 7, "worst_day": 25},
                "xgboost": {"improvement": 100.00, "rmse": 0.0548, "best_day": 1, "worst_day": 30},
                "lstm": {"improvement": 0.0, "rmse": 3102.8, "best_day": 11, "worst_day": 27}
            },
            "tuoshan": {
                "grud": {"improvement": 79.78, "rmse": 0.2028, "best_day": 5, "worst_day": 1},
                "tcn": {"improvement": 0.07, "rmse": 1.0829, "best_day": 12, "worst_day": 9},
                "xgboost": {"improvement": 49.31, "rmse": 0.5580, "best_day": 3, "worst_day": 28},
                "lstm": {"improvement": 0.0, "rmse": 1.1463, "best_day": 1, "worst_day": 25}
            },
            "lanshanzui": {
                "grud": {"improvement": 28.83, "rmse": 0.2570, "best_day": 14, "worst_day": 26},
                "tcn": {"improvement": 5.35, "rmse": 0.3461, "best_day": 2, "worst_day": 26},
                "xgboost": {"improvement": 9.32, "rmse": 0.3446, "best_day": 2, "worst_day": 26},
                "lstm": {"improvement": 0.0, "rmse": 0.3611, "best_day": 1, "worst_day": 21}
            },
            "五里湖心": {
                "grud": {"improvement": 42.93, "rmse": 0.5607, "best_day": 22, "worst_day": 1},
                "tcn": {"improvement": 2.69, "rmse": 1.0730, "best_day": 8, "worst_day": 26},
                "xgboost": {"improvement": -40.37, "rmse": 1.6478, "best_day": 9, "worst_day": 18},
                "lstm": {"improvement": 0.0, "rmse": 1.0844, "best_day": 1, "worst_day": 29}
            }
        }
    
    async def predict_v2(self, request: PredictionRequestV2) -> PredictionResponse:
        """
        执行预测V2 - 支持优化的历史数据处理
        """
        try:
            logger.info(f"开始执行预测V2: {request.station} - {request.model_type} - {request.predict_days}天")
            
            # 第一步：获取历史数据序列
            input_sequence = await self._prepare_input_sequence(request)
            
            # 第二步：数据质量验证
            if request.validate_data_quality:
                quality_report = await self.historical_data_service.validate_data_quality(input_sequence)
                logger.info(f"数据质量分数: {quality_report['quality_score']:.3f}")
                
                if quality_report['quality_score'] < 0.5:
                    logger.warning(f"数据质量较低: {quality_report['warnings']}")
            
            # 第三步：数据预处理
            processed_input = await self._preprocess_input(input_sequence, request.station)
            
            # 第四步：执行模型预测
            predictions = await self._execute_prediction(
                processed_input, request.station, request.model_type, request.predict_days
            )
            
            # 第五步：构建响应
            response = await self._build_response(
                predictions, request, input_sequence, quality_report if request.validate_data_quality else None
            )
            
            logger.info(f"预测V2完成: {request.station} - {len(predictions)}个预测值")
            return response
            
        except Exception as e:
            logger.error(f"预测V2执行失败: {e}")
            raise
    
    async def _prepare_input_sequence(self, request: PredictionRequestV2) -> np.ndarray:
        """准备输入序列数据"""
        try:
            if request.data_mode == "auto_historical":
                # 自动获取历史数据
                sequence = await self.historical_data_service.get_historical_sequence(
                    station=request.station,
                    end_date=request.end_date,
                    seq_length=request.seq_length,
                    fill_missing_method=request.fill_missing_method
                )
                
            elif request.data_mode == "hybrid":
                # 混合模式：基础历史数据 + 补充数据
                base_sequence = await self.historical_data_service.get_historical_sequence(
                    station=request.station,
                    end_date=request.end_date,
                    seq_length=request.seq_length,
                    fill_missing_method=request.fill_missing_method
                )
                
                # 应用补充数据
                if request.supplementary_data:
                    supplementary_dict = [point.dict() for point in request.supplementary_data]
                    sequence = await self.historical_data_service.apply_supplementary_data(
                        base_sequence=base_sequence,
                        supplementary_data=supplementary_dict,
                        end_date=request.end_date,
                        seq_length=request.seq_length
                    )
                else:
                    sequence = base_sequence
                    
            elif request.data_mode == "manual_upload":
                # 手动上传模式（暂未实现）
                raise NotImplementedError("手动上传模式尚未实现")
            
            else:
                raise ValueError(f"不支持的数据模式: {request.data_mode}")
            
            return sequence
            
        except Exception as e:
            logger.error(f"准备输入序列失败: {e}")
            raise
    
    async def _preprocess_input(self, input_sequence: np.ndarray, station: str) -> np.ndarray:
        """预处理输入数据"""
        try:
            # 使用现有的数据处理器
            loop = asyncio.get_event_loop()
            processed_data = await loop.run_in_executor(
                None, self.data_processor.prepare_sequence_data, input_sequence
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
            # 加载模型
            model = await self.model_manager.load_model(station, model_type)
            if model is None:
                raise ValueError(f"无法加载模型: {station} - {model_type}")
            
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
    
    async def _build_response(
        self, 
        predictions: List[float], 
        request: PredictionRequestV2,
        input_sequence: np.ndarray,
        quality_report: Optional[Dict[str, Any]] = None
    ) -> PredictionResponse:
        """构建响应"""
        try:
            # 获取模型和站点显示信息
            model_info = get_model_display_info(request.model_type)
            station_display = get_station_display_name(request.station)
            
            # 构建预测时间序列
            from datetime import datetime, timedelta
            end_dt = datetime.strptime(request.end_date, '%Y-%m-%d')
            prediction_dates = [
                (end_dt + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                for i in range(len(predictions))
            ]
            
            # 计算输入数据统计
            input_stats = {
                "mean_temperature": float(np.mean(input_sequence[:, 0])),
                "mean_oxygen": float(np.mean(input_sequence[:, 2])),
                "mean_pH": float(np.mean(input_sequence[:, 1])),
                "data_coverage": 1.0 - (np.isnan(input_sequence).sum() / input_sequence.size)
            }
            
            # 构建响应数据
            response_data = {
                "prediction": predictions,
                "prediction_dates": prediction_dates,
                "input_stats": input_stats,
                "model_info": {
                    "type": request.model_type,
                    "name": model_info["name"],
                    "description": model_info["description"]
                },
                "data_info": {
                    "mode": request.data_mode,
                    "sequence_length": request.seq_length,
                    "end_date": request.end_date,
                    "supplementary_points": len(request.supplementary_data) if request.supplementary_data else 0
                }
            }
            
            # 添加数据质量报告
            if quality_report:
                response_data["quality_report"] = {
                    "score": quality_report["quality_score"],
                    "warnings": quality_report["warnings"],
                    "missing_ratio": quality_report["missing_values"] / input_sequence.size
                }
            
            return PredictionResponse(
                success=True,
                message=f"成功预测 {station_display} 未来 {request.predict_days} 天的蓝藻密度变化",
                data=response_data,
                timestamp=datetime.now().isoformat(),
                request_id=f"{request.station}_{request.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
        except Exception as e:
            logger.error(f"构建响应失败: {e}")
            raise
    
    async def get_model_performance(
        self, 
        station: Optional[str] = None, 
        model_types: Optional[List[str]] = None
    ) -> ModelPerformanceResponse:
        """获取模型性能对比信息"""
        try:
            logger.info(f"获取模型性能: 站点={station}, 模型={model_types}")
            
            # 确定要返回的站点
            target_stations = [station] if station else list(self.performance_data.keys())
            
            # 确定要返回的模型类型
            target_models = model_types if model_types else ["lstm", "grud", "tcn", "xgboost"]
            
            performance_data = {}
            
            for station_name in target_stations:
                if station_name in self.performance_data:
                    station_data = {}
                    for model_type in target_models:
                        if model_type in self.performance_data[station_name]:
                            model_data = self.performance_data[station_name][model_type]
                            station_data[model_type] = {
                                "improvement_rate": model_data["improvement"],
                                "rmse": model_data["rmse"],
                                "best_performance_day": model_data["best_day"],
                                "worst_performance_day": model_data["worst_day"],
                                "recommended": model_data["improvement"] > 10.0
                            }
                    
                    if station_data:
                        performance_data[station_name] = station_data
            
            # 生成推荐
            recommendations = self._generate_recommendations(performance_data)
            
            return ModelPerformanceResponse(
                success=True,
                message="模型性能数据获取成功",
                data={
                    "performance_comparison": performance_data,
                    "recommendations": recommendations,
                    "metadata": {
                        "comparison_stations": list(performance_data.keys()),
                        "comparison_models": target_models,
                        "metrics_explanation": {
                            "improvement_rate": "相对于LSTM基准模型的改善率(%)",
                            "rmse": "均方根误差",
                            "recommended": "是否推荐使用该模型"
                        }
                    }
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"获取模型性能失败: {e}")
            raise
    
    def _generate_recommendations(self, performance_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """生成模型推荐"""
        recommendations = []
        
        for station, models in performance_data.items():
            # 找出最佳模型
            best_model = max(models.keys(), key=lambda m: models[m]["improvement_rate"])
            best_improvement = models[best_model]["improvement_rate"]
            
            if best_improvement > 50:
                level = "强烈推荐"
            elif best_improvement > 20:
                level = "推荐"
            elif best_improvement > 0:
                level = "可选"
            else:
                level = "不推荐"
            
            recommendation = {
                "station": station,
                "recommended_model": best_model,
                "improvement": f"{best_improvement:.1f}%",
                "level": level,
                "reason": f"在{station}站点表现最优，相比LSTM提升{best_improvement:.1f}%"
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    async def get_data_info(self, station: str) -> Dict[str, Any]:
        """获取站点数据信息"""
        try:
            return await self.historical_data_service.get_data_summary(station)
        except Exception as e:
            logger.error(f"获取数据信息失败: {e}")
            raise
    
    async def validate_prediction_request(self, request: PredictionRequestV2) -> Dict[str, Any]:
        """验证预测请求的可行性"""
        try:
            validation_result = {
                "valid": True,
                "warnings": [],
                "data_availability": {},
                "recommendations": []
            }
            
            # 检查数据可用性
            try:
                min_date, max_date = await self.historical_data_service.get_available_date_range(request.station)
                validation_result["data_availability"] = {
                    "start_date": min_date,
                    "end_date": max_date,
                    "available": True
                }
                
                # 检查请求的日期是否在可用范围内
                if request.end_date and request.end_date > max_date:
                    validation_result["warnings"].append(f"请求日期 {request.end_date} 超出数据范围 {max_date}")
                
            except Exception as e:
                validation_result["data_availability"]["available"] = False
                validation_result["warnings"].append(f"无法获取数据可用性: {str(e)}")
            
            # 检查模型可用性
            try:
                model = await self.model_manager.load_model(request.station, request.model_type)
                if model is None:
                    validation_result["warnings"].append(f"模型 {request.model_type} 在 {request.station} 不可用")
            except Exception as e:
                validation_result["warnings"].append(f"模型加载检查失败: {str(e)}")
            
            # 生成推荐
            if request.station in self.performance_data:
                best_models = sorted(
                    self.performance_data[request.station].items(),
                    key=lambda x: x[1]["improvement"],
                    reverse=True
                )[:2]
                
                for model_name, performance in best_models:
                    if performance["improvement"] > 0:
                        validation_result["recommendations"].append({
                            "model": model_name,
                            "improvement": f"{performance['improvement']:.1f}%",
                            "reason": f"在该站点表现优异"
                        })
            
            validation_result["valid"] = len([w for w in validation_result["warnings"] if "不可用" in w]) == 0
            
            return validation_result
            
        except Exception as e:
            logger.error(f"验证预测请求失败: {e}")
            raise
