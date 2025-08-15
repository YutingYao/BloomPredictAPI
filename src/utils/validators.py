# -*- coding: utf-8 -*-
"""
数据验证工具
"""

from typing import List, Dict, Any
from src.config.settings import Settings

settings = Settings()

def validate_station(station: str) -> bool:
    """验证监测站点是否支持"""
    return station in settings.SUPPORTED_STATIONS

def validate_model_type(model_type: str) -> bool:
    """验证模型类型是否支持"""
    return model_type.lower() in settings.SUPPORTED_MODELS

def validate_predict_days(days: int) -> bool:
    """验证预测天数是否在有效范围内"""
    return settings.MIN_PREDICT_DAYS <= days <= settings.MAX_PREDICT_DAYS

def validate_input_data(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    验证输入数据的完整性和有效性
    
    Args:
        data: 输入数据字典
        
    Returns:
        Dict containing validation errors, empty if all valid
    """
    errors = {"missing_fields": [], "invalid_values": [], "out_of_range": []}
    
    # 检查必需字段
    for field in settings.INPUT_FEATURES:
        if field not in data:
            errors["missing_fields"].append(field)
    
    # 检查数据类型和范围
    for field, value in data.items():
        if field in settings.FEATURE_RANGES:
            try:
                float_value = float(value)
                range_info = settings.FEATURE_RANGES[field]
                
                if float_value < range_info["min"] or float_value > range_info["max"]:
                    errors["out_of_range"].append(
                        f"{field}: {float_value} (范围: {range_info['min']}-{range_info['max']})"
                    )
            except (ValueError, TypeError):
                errors["invalid_values"].append(f"{field}: {value} (应为数值)")
    
    # 移除空的错误类别
    return {k: v for k, v in errors.items() if v}

def get_station_display_name(station: str) -> str:
    """获取站点的显示名称"""
    station_mapping = {
        "tuoshan": "拖山",
        "lanshanzui": "兰山嘴"
    }
    return station_mapping.get(station, station)

def get_model_display_info(model_type: str) -> Dict[str, str]:
    """获取模型的显示信息"""
    model_info = {
        "lstm": {
            "name": "LSTM",
            "description": "长短期记忆网络，基准模型",
            "features": "三门控机制，长期记忆能力强"
        },
        "grud": {
            "name": "GRU-D", 
            "description": "专为缺失数据设计的GRU改进版",
            "features": "处理缺失数据，训练速度快，预测精度高"
        },
        "tcn": {
            "name": "TCN",
            "description": "时间卷积网络",
            "features": "因果卷积，并行计算，大感受野"
        },
        "xgboost": {
            "name": "XGBoost",
            "description": "梯度提升树算法",
            "features": "梯度提升，可解释性强"
        }
    }
    return model_info.get(model_type.lower(), {
        "name": model_type.upper(),
        "description": "未知模型",
        "features": ""
    })
