# -*- coding: utf-8 -*-
"""
请求数据模型定义 V2 - 优化后的API设计
"""

from pydantic import BaseModel, validator, Field
from typing import Dict, List, Optional, Union, Literal
from datetime import date, datetime
import re

class HistoricalDataQuery(BaseModel):
    """历史数据查询模型"""
    station: str = Field(..., description="监测站点名称")
    end_date: str = Field(..., description="历史数据结束日期 (YYYY-MM-DD)")
    seq_length: int = Field(60, description="历史数据序列长度", ge=30, le=120)
    
    @validator('end_date')
    def validate_end_date(cls, v):
        """验证日期格式"""
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

class SupplementaryDataPoint(BaseModel):
    """补充数据点模型"""
    date: str = Field(..., description="数据日期 (YYYY-MM-DD)")
    temperature: Optional[float] = Field(None, description="温度(°C)", ge=-5, le=40)
    oxygen: Optional[float] = Field(None, description="溶解氧(mg/L)", ge=0, le=20)
    TN: Optional[float] = Field(None, description="总氮(mg/L)", ge=0, le=10)
    TP: Optional[float] = Field(None, description="总磷(mg/L)", ge=0, le=1)
    NH: Optional[float] = Field(None, description="氨氮(mg/L)", ge=0, le=5)
    pH: Optional[float] = Field(None, description="pH值", ge=5, le=10)
    turbidity: Optional[float] = Field(None, description="浊度(NTU)", ge=0, le=200)
    conductivity: Optional[float] = Field(None, description="电导率(μS/cm)", ge=0, le=2000)
    permanganate: Optional[float] = Field(None, description="高锰酸盐指数(mg/L)", ge=0, le=20)
    rain_sum: Optional[float] = Field(None, description="降雨量(mm)", ge=0, le=200)
    wind_speed_10m_max: Optional[float] = Field(None, description="风速(m/s)", ge=0, le=30)
    shortwave_radiation_sum: Optional[float] = Field(None, description="短波辐射(MJ/m²)", ge=0, le=50)
    
    @validator('date')
    def validate_date(cls, v):
        """验证日期格式"""
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError("日期格式不正确，应为 YYYY-MM-DD")
        return v

class PredictionRequestV2(BaseModel):
    """预测请求模型 V2 - 优化版本"""
    station: str = Field(..., description="监测站点名称")
    model_type: str = Field(..., description="预测模型类型")
    predict_days: int = Field(..., description="预测天数", ge=1, le=30)
    
    # 历史数据获取方式
    data_mode: Literal["auto_historical", "manual_upload", "hybrid"] = Field(
        "auto_historical", 
        description="数据获取模式: auto_historical=自动获取历史数据, manual_upload=手动上传, hybrid=混合模式"
    )
    
    # 自动历史数据模式参数
    end_date: Optional[str] = Field(None, description="历史数据结束日期 (YYYY-MM-DD)")
    seq_length: int = Field(60, description="历史数据序列长度", ge=30, le=120)
    
    # 混合模式参数 - 允许覆盖最近几天的数据
    override_recent_days: Optional[int] = Field(None, description="覆盖最近N天的数据", ge=1, le=30)
    supplementary_data: Optional[List[SupplementaryDataPoint]] = Field(None, description="补充或覆盖的数据点")
    
    # 数据处理选项
    fill_missing_method: Literal["interpolation", "forward_fill", "backward_fill", "mean"] = Field(
        "interpolation", 
        description="缺失数据填补方法"
    )
    validate_data_quality: bool = Field(True, description="是否进行数据质量验证")
    
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
    
    @validator('end_date')
    def validate_end_date(cls, v, values):
        """验证结束日期"""
        if v is None and values.get('data_mode') == 'auto_historical':
            raise ValueError("自动历史数据模式需要提供 end_date")
        if v is not None:
            try:
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError("日期格式不正确，应为 YYYY-MM-DD")
        return v
    
    @validator('override_recent_days')
    def validate_override_recent_days(cls, v, values):
        """验证覆盖天数"""
        if v is not None and values.get('data_mode') != 'hybrid':
            raise ValueError("只有 hybrid 模式支持 override_recent_days 参数")
        return v
    
    @validator('supplementary_data')
    def validate_supplementary_data(cls, v, values):
        """验证补充数据"""
        if v is not None and values.get('data_mode') == 'auto_historical':
            raise ValueError("auto_historical 模式不支持 supplementary_data")
        return v
    
    class Config:
        schema_extra = {
            "examples": [
                {
                    "title": "自动历史数据模式",
                    "description": "系统自动获取指定日期前60天的历史数据",
                    "value": {
                        "station": "胥湖心",
                        "model_type": "grud",
                        "predict_days": 7,
                        "data_mode": "auto_historical",
                        "end_date": "2024-12-25",
                        "seq_length": 60,
                        "fill_missing_method": "interpolation"
                    }
                },
                {
                    "title": "混合模式",
                    "description": "基于历史数据，并覆盖最近几天的实际监测数据",
                    "value": {
                        "station": "胥湖心", 
                        "model_type": "grud",
                        "predict_days": 7,
                        "data_mode": "hybrid",
                        "end_date": "2024-12-25",
                        "seq_length": 60,
                        "override_recent_days": 3,
                        "supplementary_data": [
                            {
                                "date": "2024-12-23",
                                "temperature": 24.1,
                                "oxygen": 8.5,
                                "pH": 7.6
                            },
                            {
                                "date": "2024-12-24", 
                                "temperature": 25.2,
                                "oxygen": 8.2,
                                "pH": 7.8
                            },
                            {
                                "date": "2024-12-25",
                                "temperature": 26.0,
                                "oxygen": 7.9,
                                "pH": 7.7
                            }
                        ]
                    }
                }
            ]
        }

class FileUploadRequest(BaseModel):
    """文件上传请求模型"""
    station: str = Field(..., description="监测站点名称")
    file_format: Literal["csv", "json"] = Field("csv", description="文件格式")
    has_header: bool = Field(True, description="CSV文件是否包含表头")
    date_column: str = Field("date", description="日期列名")
    date_format: str = Field("%Y-%m-%d", description="日期格式")
    
    @validator('station')
    def validate_station(cls, v):
        """验证站点名称"""
        supported_stations = [
            "胥湖心", "锡东水厂", "平台山", "tuoshan", "lanshanzui", "五里湖心"
        ]
        if v not in supported_stations:
            raise ValueError(f"不支持的监测站点: {v}。支持的站点: {', '.join(supported_stations)}")
        return v

class BatchPredictionRequest(BaseModel):
    """批量预测请求模型"""
    requests: List[PredictionRequestV2] = Field(..., description="批量预测请求列表")
    parallel_execution: bool = Field(True, description="是否并行执行")
    max_workers: int = Field(4, description="最大并行工作线程数", ge=1, le=8)
    
    @validator('requests')
    def validate_requests_count(cls, v):
        """验证请求数量"""
        if len(v) > 20:
            raise ValueError("批量请求数量不能超过20个")
        return v
