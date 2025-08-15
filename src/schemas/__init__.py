# -*- coding: utf-8 -*-
"""
数据模型定义模块
"""

from .request_schemas import PredictionRequest, ModelPerformanceRequest, InputData
from .response_schemas import (
    PredictionResponse, 
    ModelPerformanceResponse, 
    ModelPerformanceInfo,
    StationPerformance,
    ErrorResponse
)

__all__ = [
    "PredictionRequest",
    "ModelPerformanceRequest", 
    "InputData",
    "PredictionResponse",
    "ModelPerformanceResponse",
    "ModelPerformanceInfo",
    "StationPerformance",
    "ErrorResponse"
]
