# -*- coding: utf-8 -*-
"""
响应数据模型定义
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

class PredictionResponse(BaseModel):
    """预测响应模型"""
    success: bool = Field(..., description="预测是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="预测结果数据")
    message: str = Field(..., description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "data": {
                    "station": "胥湖心",
                    "model_type": "grud",
                    "predict_days": 7,
                    "prediction": [0.12, 0.15, 0.18, 0.21, 0.19, 0.17, 0.16],
                    "confidence": 0.85,
                    "rmse": 0.2567,
                    "model_info": {
                        "name": "GRU-D",
                        "description": "专为缺失数据设计的GRU改进版",
                        "improvement_over_lstm": "77.41%"
                    }
                },
                "message": "预测成功",
                "timestamp": "2024-01-20T10:30:00.000Z"
            }
        }

class ModelPerformanceInfo(BaseModel):
    """模型性能信息"""
    model_type: str = Field(..., description="模型类型")
    model_name: str = Field(..., description="模型名称")
    rmse: float = Field(..., description="均方根误差")
    improvement_over_lstm: Optional[float] = Field(None, description="相对LSTM的改善率(%)")
    best_performance_day: Optional[int] = Field(None, description="最佳表现天数")
    worst_performance_day: Optional[int] = Field(None, description="最差表现天数")

class StationPerformance(BaseModel):
    """站点性能信息"""
    station: str = Field(..., description="站点名称")
    station_en: str = Field(..., description="站点英文名称")
    zone: str = Field(..., description="生态区域")
    zone_en: str = Field(..., description="生态区域英文名称")
    models: List[ModelPerformanceInfo] = Field(..., description="模型性能列表")

class ModelPerformanceResponse(BaseModel):
    """模型性能响应模型"""
    success: bool = Field(..., description="请求是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="性能对比数据")
    message: str = Field(..., description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "data": {
                    "station_performance": [
                        {
                            "station": "胥湖心",
                            "station_en": "Xuhu Center",
                            "zone": "重污染与高风险区",
                            "zone_en": "Heavily Polluted & High-Risk Area",
                            "models": [
                                {
                                    "model_type": "lstm",
                                    "model_name": "LSTM",
                                    "rmse": 1.5234,
                                    "improvement_over_lstm": 0.0,
                                    "best_performance_day": 1,
                                    "worst_performance_day": 30
                                },
                                {
                                    "model_type": "grud",
                                    "model_name": "GRU-D", 
                                    "rmse": 0.3441,
                                    "improvement_over_lstm": 77.41,
                                    "best_performance_day": 25,
                                    "worst_performance_day": 1
                                }
                            ]
                        }
                    ],
                    "summary": {
                        "best_model": "grud",
                        "overall_improvement": 50.07,
                        "recommended_model": "grud"
                    }
                },
                "message": "模型性能获取成功",
                "timestamp": "2024-01-20T10:30:00.000Z"
            }
        }

class ErrorResponse(BaseModel):
    """错误响应模型"""
    success: bool = Field(False, description="请求是否成功")
    error: str = Field(..., description="错误类型")
    detail: str = Field(..., description="错误详情")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "ValidationError",
                "detail": "不支持的监测站点: 无效站点",
                "timestamp": "2024-01-20T10:30:00.000Z"
            }
        }
