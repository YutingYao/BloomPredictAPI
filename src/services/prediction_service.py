# -*- coding: utf-8 -*-
"""
预测服务 - 核心业务逻辑
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

from src.models.model_manager import ModelManager
from src.utils.data_processor import DataProcessor
from src.utils.validators import get_model_display_info, get_station_display_name
from src.schemas.response_schemas import PredictionResponse, ModelPerformanceResponse
from src.config.settings import Settings

logger = logging.getLogger(__name__)

class PredictionService:
    """预测服务类"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.data_processor = DataProcessor()
        self.settings = Settings()
        
        # 预定义的模型性能数据（基于之前的分析结果）
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
                "grud": {"improvement": 79.78, "rmse": 0.2063, "best_day": 5, "worst_day": 1},
                "tcn": {"improvement": 0.07, "rmse": 1.1149, "best_day": 12, "worst_day": 9},
                "xgboost": {"improvement": 49.31, "rmse": 0.5681, "best_day": 1, "worst_day": 30},
                "lstm": {"improvement": 0.0, "rmse": 1.1146, "best_day": 1, "worst_day": 25}
            },
            "lanshanzui": {
                "grud": {"improvement": 28.83, "rmse": 0.2348, "best_day": 14, "worst_day": 26},
                "tcn": {"improvement": 5.35, "rmse": 0.3321, "best_day": 2, "worst_day": 26},
                "xgboost": {"improvement": 9.32, "rmse": 0.3178, "best_day": 21, "worst_day": 26},
                "lstm": {"improvement": 0.0, "rmse": 0.3429, "best_day": 26, "worst_day": 21}
            },
            "五里湖心": {
                "grud": {"improvement": 42.93, "rmse": 0.5365, "best_day": 22, "worst_day": 1},
                "tcn": {"improvement": 2.69, "rmse": 0.9902, "best_day": 8, "worst_day": 26},
                "xgboost": {"improvement": -40.37, "rmse": 1.7718, "best_day": 9, "worst_day": 18},
                "lstm": {"improvement": 0.0, "rmse": 1.0012, "best_day": 25, "worst_day": 21}
            }
        }
    
    async def predict(
        self,
        station: str,
        model_type: str,
        predict_days: int,
        input_data: Dict[str, Any]
    ) -> PredictionResponse:
        """
        执行蓝藻密度预测
        
        Args:
            station: 监测站点
            model_type: 模型类型
            predict_days: 预测天数
            input_data: 输入数据
            
        Returns:
            预测响应结果
        """
        try:
            logger.info(f"开始预测: {station}, {model_type}, {predict_days}天")
            
            # 数据预处理
            input_dict = input_data.dict() if hasattr(input_data, 'dict') else input_data
            cleaned_data = self.data_processor.validate_and_clean_data(input_dict)
            normalized_data = self.data_processor.normalize_input_data(cleaned_data, station)
            
            # 根据模型类型准备输入数据
            if model_type.lower() == 'xgboost':
                model_input = self.data_processor.create_xgb_features(normalized_data)
            else:
                model_input = self.data_processor.create_sequence_data(normalized_data)
                
            logger.info(f"输入序列形状: {model_input.shape}, 序列长度: {self.settings.SEQ_LENGTH}")
            
            # 执行预测
            raw_prediction = await self.model_manager.predict_with_model(
                station, model_type, predict_days, model_input
            )
            
            if raw_prediction is None:
                raise ValueError(f"模型预测失败，可能是模型未加载或输入数据有误")
            
            # 后处理预测结果
            prediction = self.data_processor.post_process_prediction(raw_prediction, model_type)
            
            # 获取模型信息
            model_info = get_model_display_info(model_type)
            
            # 获取性能信息
            performance_info = self.performance_data.get(station, {}).get(model_type.lower(), {})
            
            # 计算置信度（基于模型性能）
            confidence = self._calculate_confidence(station, model_type, predict_days)
            
            # 构建响应数据
            response_data = {
                "station": station,
                "station_display": get_station_display_name(station),
                "model_type": model_type,
                "predict_days": predict_days,
                "prediction": prediction,
                "confidence": confidence,
                "rmse": performance_info.get("rmse", 0.0),
                "model_info": {
                    "name": model_info["name"],
                    "description": model_info["description"],
                    "features": model_info["features"],
                    "improvement_over_lstm": f"{performance_info.get('improvement', 0):.2f}%"
                },
                "input_summary": self._create_input_summary(cleaned_data)
            }
            
            logger.info(f"预测完成: {station}, 预测值数量={len(prediction)}")
            
            return PredictionResponse(
                success=True,
                data=response_data,
                message="预测成功",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return PredictionResponse(
                success=False,
                data=None,
                message=f"预测失败: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _calculate_confidence(self, station: str, model_type: str, predict_days: int) -> float:
        """计算预测置信度"""
        try:
            performance_info = self.performance_data.get(station, {}).get(model_type.lower(), {})
            
            if not performance_info:
                return 0.5  # 默认置信度
            
            # 基于改善率计算基础置信度
            improvement = performance_info.get("improvement", 0)
            base_confidence = min(0.9, max(0.3, (improvement + 50) / 150))
            
            # 根据预测天数调整置信度（短期预测通常更可靠）
            day_factor = max(0.5, 1.0 - (predict_days - 1) * 0.02)
            
            # 根据RMSE调整置信度
            rmse = performance_info.get("rmse", 1.0)
            rmse_factor = max(0.6, 1.0 - min(2.0, rmse) * 0.2)
            
            final_confidence = base_confidence * day_factor * rmse_factor
            return round(final_confidence, 3)
            
        except Exception as e:
            logger.warning(f"计算置信度失败: {e}")
            return 0.5
    
    def _create_input_summary(self, data: Dict[str, float]) -> Dict[str, Any]:
        """创建输入数据摘要"""
        return {
            "水质指标": {
                "温度": f"{data.get('temperature', 0):.1f}°C",
                "pH": f"{data.get('pH', 0):.1f}",
                "溶解氧": f"{data.get('oxygen', 0):.1f}mg/L",
                "总氮": f"{data.get('TN', 0):.2f}mg/L",
                "总磷": f"{data.get('TP', 0):.3f}mg/L"
            },
            "气象条件": {
                "降雨量": f"{data.get('rain_sum', 0):.1f}mm",
                "风速": f"{data.get('wind_speed_10m_max', 0):.1f}m/s",
                "短波辐射": f"{data.get('shortwave_radiation_sum', 0):.1f}MJ/m²"
            }
        }
    
    async def get_model_performance(
        self,
        station: Optional[str] = None,
        model_types: Optional[List[str]] = None
    ) -> ModelPerformanceResponse:
        """
        获取模型性能对比信息
        
        Args:
            station: 指定站点（可选）
            model_types: 指定模型类型列表（可选）
            
        Returns:
            模型性能响应结果
        """
        try:
            logger.info(f"获取模型性能信息: station={station}, models={model_types}")
            
            # 确定要分析的站点
            target_stations = [station] if station else self.settings.SUPPORTED_STATIONS
            
            # 确定要分析的模型类型
            target_models = model_types if model_types else self.settings.SUPPORTED_MODELS
            
            # 构建性能数据
            station_performance = []
            
            for station_name in target_stations:
                if station_name not in self.performance_data:
                    continue
                
                station_data = self._get_station_info(station_name)
                models_info = []
                
                for model_type in target_models:
                    if model_type.lower() in self.performance_data[station_name]:
                        model_perf = self.performance_data[station_name][model_type.lower()]
                        model_info = get_model_display_info(model_type)
                        
                        models_info.append({
                            "model_type": model_type,
                            "model_name": model_info["name"],
                            "rmse": model_perf.get("rmse", 0.0),
                            "improvement_over_lstm": model_perf.get("improvement", 0.0),
                            "best_performance_day": model_perf.get("best_day"),
                            "worst_performance_day": model_perf.get("worst_day"),
                            "description": model_info["description"]
                        })
                
                if models_info:
                    station_performance.append({
                        "station": station_name,
                        "station_en": station_data["station_en"],
                        "zone": station_data["zone"],
                        "zone_en": station_data["zone_en"],
                        "models": models_info
                    })
            
            # 计算总体统计
            summary = self._calculate_performance_summary(station_performance, target_models)
            
            response_data = {
                "station_performance": station_performance,
                "summary": summary,
                "analysis_info": {
                    "total_stations": len(station_performance),
                    "total_models": len(target_models),
                    "data_source": "预训练模型性能评估结果",
                    "evaluation_period": "2023年11月-2024年5月"
                }
            }
            
            return ModelPerformanceResponse(
                success=True,
                data=response_data,
                message="模型性能信息获取成功",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"获取模型性能信息失败: {e}")
            return ModelPerformanceResponse(
                success=False,
                data=None,
                message=f"获取模型性能信息失败: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _get_station_info(self, station: str) -> Dict[str, str]:
        """获取站点信息"""
        station_mapping = {
            "胥湖心": {
                "station_en": "Xuhu Center",
                "zone": "重污染与高风险区",
                "zone_en": "Heavily Polluted & High-Risk Area"
            },
            "锡东水厂": {
                "station_en": "Xidong Water Plant",
                "zone": "重污染与高风险区",
                "zone_en": "Heavily Polluted & High-Risk Area"
            },
            "平台山": {
                "station_en": "Pingtai Mountain",
                "zone": "背景与参照区",
                "zone_en": "Background & Reference Area"
            },
            "tuoshan": {
                "station_en": "Tuoshan Mountain",
                "zone": "背景与参照区",
                "zone_en": "Background & Reference Area"
            },
            "lanshanzui": {
                "station_en": "Lanshan Cape",
                "zone": "边界条件区",
                "zone_en": "Boundary Condition Area"
            },
            "五里湖心": {
                "station_en": "Wulihu Center",
                "zone": "边界条件区",
                "zone_en": "Boundary Condition Area"
            }
        }
        
        return station_mapping.get(station, {
            "station_en": station,
            "zone": "未知区域",
            "zone_en": "Unknown Area"
        })
    
    def _calculate_performance_summary(
        self, 
        station_performance: List[Dict],
        target_models: List[str]
    ) -> Dict[str, Any]:
        """计算性能总结"""
        try:
            # 统计各模型的平均改善率
            model_improvements = {}
            model_counts = {}
            
            for station_data in station_performance:
                for model_info in station_data["models"]:
                    model_type = model_info["model_type"]
                    improvement = model_info["improvement_over_lstm"]
                    
                    if model_type not in model_improvements:
                        model_improvements[model_type] = 0
                        model_counts[model_type] = 0
                    
                    model_improvements[model_type] += improvement
                    model_counts[model_type] += 1
            
            # 计算平均值
            avg_improvements = {}
            for model_type in model_improvements:
                if model_counts[model_type] > 0:
                    avg_improvements[model_type] = model_improvements[model_type] / model_counts[model_type]
            
            # 找出最佳模型
            best_model = max(avg_improvements.items(), key=lambda x: x[1]) if avg_improvements else ("grud", 0)
            
            # 推荐模型（基于综合表现）
            recommended_model = "grud"  # 基于分析结果，GRU-D在大多数情况下表现最佳
            
            return {
                "best_model": best_model[0],
                "best_model_improvement": round(best_model[1], 2),
                "overall_improvement": round(
                    sum(avg_improvements.values()) / len(avg_improvements) if avg_improvements else 0, 2
                ),
                "recommended_model": recommended_model,
                "model_rankings": sorted(
                    avg_improvements.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ),
                "analysis_notes": [
                    "GRU-D模型在大多数站点表现优异，平均改善率超过50%",
                    "XGBoost在部分站点（如平台山）表现突出",
                    "TCN模型整体改善有限，但在某些特定情况下有效",
                    "短期预测（1-7天）通常比长期预测更准确"
                ]
            }
            
        except Exception as e:
            logger.error(f"计算性能总结失败: {e}")
            return {
                "best_model": "grud",
                "overall_improvement": 0.0,
                "recommended_model": "grud"
            }
