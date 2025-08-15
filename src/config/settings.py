# -*- coding: utf-8 -*-
"""
应用配置设置
"""

from pydantic_settings import BaseSettings
from typing import List, Dict, Any
import os

class Settings(BaseSettings):
    """应用配置类"""
    
    # API设置
    API_TITLE: str = "蓝藻预测系统API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "基于多种机器学习模型的太湖流域蓝藻密度预测系统"
    
    # 服务器设置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS设置
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080"
    ]
    
    # 支持的监测站点
    SUPPORTED_STATIONS: List[str] = [
        "胥湖心", "锡东水厂", "平台山", "tuoshan", "lanshanzui", "五里湖心"
    ]
    
    # 支持的模型类型
    SUPPORTED_MODELS: List[str] = ["lstm", "grud", "tcn", "xgboost"]
    
    # 预测设置
    MAX_PREDICT_DAYS: int = 30
    MIN_PREDICT_DAYS: int = 1
    
    # 模型文件设置
    MODEL_BASE_PATH: str = "."
    MODEL_FILE_PATTERNS: Dict[str, str] = {
        "lstm": "models/00-lstm_model_data_{station}-去除负数.pkl",
        "grud": "models/00-GRUD_model_data_{station}-去除负数.pkl", 
        "tcn": "models/00-TCN_model_data_{station}-去除负数.pkl",
        "xgboost": "models/00-XGB_model_data_{station}-去除负数.pkl"
    }
    
    # 输入特征字段
    INPUT_FEATURES: List[str] = [
        'temperature', 'pH', 'oxygen', 'permanganate', 'TN', 'conductivity',
        'turbidity', 'rain_sum', 'wind_speed_10m_max', 'shortwave_radiation_sum',
        'TP', 'NH'
    ]
    
    # 数据验证范围
    FEATURE_RANGES: Dict[str, Dict[str, float]] = {
        "temperature": {"min": -5.0, "max": 40.0},
        "oxygen": {"min": 0.0, "max": 20.0},
        "TN": {"min": 0.0, "max": 10.0},
        "TP": {"min": 0.0, "max": 1.0},
        "NH": {"min": 0.0, "max": 5.0},
        "pH": {"min": 5.0, "max": 10.0},
        "turbidity": {"min": 0.0, "max": 200.0},
        "conductivity": {"min": 0.0, "max": 2000.0},
        "permanganate": {"min": 0.0, "max": 20.0},
        "rain_sum": {"min": 0.0, "max": 200.0},
        "wind_speed_10m_max": {"min": 0.0, "max": 30.0},
        "shortwave_radiation_sum": {"min": 0.0, "max": 50.0}
    }
    
    # 日志设置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/api.log"
    
    # 模型性能缓存设置
    PERFORMANCE_CACHE_HOURS: int = 24
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
