# -*- coding: utf-8 -*-
"""
请求数据模型定义
"""

from pydantic import BaseModel, validator, Field
from typing import Dict, List, Optional, Union
import re

class InputData(BaseModel):
    """输入数据模型"""
    temperature: float = Field(..., description="温度(°C)", ge=-5, le=40)
    oxygen: float = Field(..., description="溶解氧(mg/L)", ge=0, le=20)
    TN: float = Field(..., description="总氮(mg/L)", ge=0, le=10)
    TP: float = Field(..., description="总磷(mg/L)", ge=0, le=1)
    NH: float = Field(..., description="氨氮(mg/L)", ge=0, le=5)
    pH: float = Field(..., description="pH值", ge=5, le=10)
    turbidity: float = Field(..., description="浊度(NTU)", ge=0, le=200)
    conductivity: float = Field(..., description="电导率(μS/cm)", ge=0, le=2000)
    permanganate: float = Field(..., description="高锰酸盐指数(mg/L)", ge=0, le=20)
    rain_sum: float = Field(..., description="降雨量(mm)", ge=0, le=200)
    wind_speed_10m_max: float = Field(..., description="风速(m/s)", ge=0, le=30)
    shortwave_radiation_sum: float = Field(..., description="短波辐射(MJ/m²)", ge=0, le=50)

    class Config:
        schema_extra = {
            "example": {
                "temperature": 25.5,
                "oxygen": 8.2,
                "TN": 1.5,
                "TP": 0.08,
                "NH": 0.5,
                "pH": 7.8,
                "turbidity": 15.2,
                "conductivity": 420.0,
                "permanganate": 3.5,
                "rain_sum": 0.0,
                "wind_speed_10m_max": 3.2,
                "shortwave_radiation_sum": 18.5
            }
        }

class PredictionRequest(BaseModel):
    """预测请求模型"""
    station: str = Field(..., description="监测站点名称")
    model_type: str = Field(..., description="预测模型类型")
    predict_days: int = Field(..., description="预测天数", ge=1, le=30)
    input_data: InputData = Field(..., description="输入的水质和气象数据")
    
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
                "station": "胥湖心",
                "model_type": "grud",
                "predict_days": 7,
                "input_data": {
                    "temperature": 25.5,
                    "oxygen": 8.2,
                    "TN": 1.5,
                    "TP": 0.08,
                    "NH": 0.5,
                    "pH": 7.8,
                    "turbidity": 15.2,
                    "conductivity": 420.0,
                    "permanganate": 3.5,
                    "rain_sum": 0.0,
                    "wind_speed_10m_max": 3.2,
                    "shortwave_radiation_sum": 18.5
                }
            }
        }

class ModelPerformanceRequest(BaseModel):
    """模型性能请求模型"""
    station: Optional[str] = Field(None, description="监测站点名称（可选）")
    model_types: Optional[List[str]] = Field(None, description="要对比的模型类型列表（可选）")
    
    @validator('station')
    def validate_station(cls, v):
        """验证站点名称"""
        if v is None:
            return v
        supported_stations = [
            "胥湖心", "锡东水厂", "平台山", "tuoshan", "lanshanzui", "五里湖心"
        ]
        if v not in supported_stations:
            raise ValueError(f"不支持的监测站点: {v}。支持的站点: {', '.join(supported_stations)}")
        return v
    
    @validator('model_types')
    def validate_model_types(cls, v):
        """验证模型类型列表"""
        if v is None:
            return v
        supported_models = ["lstm", "grud", "tcn", "xgboost"]
        invalid_models = [m for m in v if m.lower() not in supported_models]
        if invalid_models:
            raise ValueError(f"不支持的模型类型: {', '.join(invalid_models)}。支持的模型: {', '.join(supported_models)}")
        return [m.lower() for m in v]
    
    class Config:
        schema_extra = {
            "example": {
                "station": "胥湖心",
                "model_types": ["lstm", "grud", "tcn"]
            }
        }
