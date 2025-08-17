# -*- coding: utf-8 -*-
"""
简化的请求数据模型定义 V3 - 只保留4个核心参数
"""

from pydantic import BaseModel, validator, Field
from typing import Optional
from datetime import date, datetime

class SimplePredictionRequest(BaseModel):
    """简化的预测请求模型 - 只需4个核心参数"""
    current_date: str = Field(..., description="当前日期 (YYYY-MM-DD)")
    predict_days: int = Field(..., description="预测天数", ge=1, le=30)
    station: str = Field(..., description="预测点位/监测站点名称")
    model_type: str = Field(..., description="模型类型")
    
    @validator('current_date')
    def validate_current_date(cls, v):
        """验证当前日期格式"""
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError("日期格式不正确，应为 YYYY-MM-DD")
        return v
    
    @validator('station')
    def validate_station(cls, v):
        """验证站点名称"""
        supported_stations = [
            "胥湖心", "锡东水厂", "平台山", "tuoshan", "lanshanzui", "五里湖心"
        ]
        if v not in supported_stations:
            raise ValueError(f"不支持的监测站点: {v}。支持的站点: {', '.join(supported_stations)}")
        return v
    
    @validator('model_type')
    def validate_model_type(cls, v):
        """验证模型类型"""
        supported_models = ["lstm", "grud", "tcn", "xgboost"]
        if v.lower() not in supported_models:
            raise ValueError(f"不支持的模型类型: {v}。支持的模型: {', '.join(supported_models)}")
        return v.lower()
    
    class Config:
        schema_extra = {
            "example": {
                "current_date": "2024-12-25",
                "predict_days": 7,
                "station": "胥湖心",
                "model_type": "grud"
            }
        }

class BatchSimplePredictionRequest(BaseModel):
    """批量简化预测请求模型"""
    requests: list[SimplePredictionRequest] = Field(..., description="批量预测请求列表")
    parallel_execution: bool = Field(True, description="是否并行执行")
    
    @validator('requests')
    def validate_requests_count(cls, v):
        """验证请求数量"""
        if len(v) > 10:
            raise ValueError("批量请求数量不能超过10个")
        return v