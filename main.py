#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蓝藻预测系统后端API
支持多种机器学习模型（LSTM、GRU-D、TCN、XGBoost）对太湖流域6个监测站点的蓝藻密度进行预测
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import Dict, List, Optional, Union
import logging
import uvicorn
import asyncio
from contextlib import asynccontextmanager

from src.models.model_manager import ModelManager
from src.services.prediction_service import PredictionService
from src.utils.validators import validate_station, validate_model_type, validate_predict_days
from src.schemas.request_schemas import PredictionRequest, ModelPerformanceRequest
from src.schemas.response_schemas import PredictionResponse, ModelPerformanceResponse
from src.config.settings import Settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局变量
model_manager = None
prediction_service = None
settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    global model_manager, prediction_service
    
    logger.info("正在初始化蓝藻预测API服务...")
    
    # 初始化模型管理器
    model_manager = ModelManager()
    await model_manager.initialize()
    
    # 初始化预测服务
    prediction_service = PredictionService(model_manager)
    
    logger.info("蓝藻预测API服务初始化完成")
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭蓝藻预测API服务...")

# 创建FastAPI应用
app = FastAPI(
    title="蓝藻预测系统API",
    description="基于多种机器学习模型的太湖流域蓝藻密度预测系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 依赖注入
async def get_prediction_service() -> PredictionService:
    """获取预测服务实例"""
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="预测服务未初始化")
    return prediction_service

# API路由
@app.get("/", tags=["基础"])
async def root():
    """根路径，返回API基本信息"""
    return {
        "message": "蓝藻预测系统API",
        "version": "1.0.0",
        "status": "运行中",
        "supported_stations": settings.SUPPORTED_STATIONS,
        "supported_models": settings.SUPPORTED_MODELS,
        "max_predict_days": settings.MAX_PREDICT_DAYS,
        "input_seq_length": settings.SEQ_LENGTH,
        "model_info": "基于60天历史数据预测未来1-30天蓝藻密度增长率"
    }

@app.get("/health", tags=["基础"])
async def health_check():
    """健康检查接口"""
    try:
        # 检查模型管理器状态
        if model_manager is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "模型管理器未初始化"}
            )
        
        # 检查预测服务状态
        if prediction_service is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "预测服务未初始化"}
            )
        
        # 检查模型加载状态
        loaded_models = await model_manager.get_loaded_models_status()
        
        return {
            "status": "healthy",
            "timestamp": str(asyncio.get_event_loop().time()),
            "loaded_models": loaded_models
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": str(e)}
        )

@app.post("/api/predict", response_model=PredictionResponse, tags=["预测"])
async def predict_algae_density(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    蓝藻密度预测接口
    
    - **station**: 监测站点名称
    - **model_type**: 预测模型类型 (lstm, grud, tcn, xgboost)
    - **predict_days**: 预测天数 (1-30)
    - **input_data**: 输入的水质和气象数据
    """
    try:
        logger.info(f"收到预测请求: 站点={request.station}, 模型={request.model_type}, 天数={request.predict_days}")
        
        # 执行预测
        result = await service.predict(
            station=request.station,
            model_type=request.model_type,
            predict_days=request.predict_days,
            input_data=request.input_data
        )
        
        prediction_count = len(result.data.get("prediction", [])) if result.data else 0
        logger.info(f"预测完成: 站点={request.station}, 预测值数量={prediction_count}")
        return result
        
    except ValueError as e:
        logger.warning(f"预测请求参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"预测过程发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.post("/api/model-performance", response_model=ModelPerformanceResponse, tags=["模型性能"])
async def get_model_performance(
    request: ModelPerformanceRequest,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    获取模型性能对比信息
    
    - **station**: 监测站点名称（可选，不指定则返回所有站点）
    - **model_types**: 要对比的模型类型列表（可选，不指定则返回所有模型）
    """
    try:
        logger.info(f"收到模型性能查询请求: 站点={request.station}, 模型={request.model_types}")
        
        result = await service.get_model_performance(
            station=request.station,
            model_types=request.model_types
        )
        
        return result
        
    except Exception as e:
        logger.error(f"获取模型性能信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型性能失败: {str(e)}")

@app.get("/api/stations", tags=["配置"])
async def get_supported_stations():
    """获取支持的监测站点列表"""
    return {
        "stations": [
            {
                "name": "胥湖心",
                "name_en": "Xuhu Center",
                "zone": "重污染与高风险区",
                "zone_en": "Heavily Polluted & High-Risk Area"
            },
            {
                "name": "锡东水厂",
                "name_en": "Xidong Water Plant", 
                "zone": "重污染与高风险区",
                "zone_en": "Heavily Polluted & High-Risk Area"
            },
            {
                "name": "平台山",
                "name_en": "Pingtai Mountain",
                "zone": "背景与参照区",
                "zone_en": "Background & Reference Area"
            },
            {
                "name": "tuoshan",
                "name_en": "Tuoshan Mountain",
                "name_cn": "拖山",
                "zone": "背景与参照区",
                "zone_en": "Background & Reference Area"
            },
            {
                "name": "lanshanzui",
                "name_en": "Lanshan Cape",
                "name_cn": "兰山嘴",
                "zone": "边界条件区",
                "zone_en": "Boundary Condition Area"
            },
            {
                "name": "五里湖心",
                "name_en": "Wulihu Center",
                "zone": "边界条件区",
                "zone_en": "Boundary Condition Area"
            }
        ]
    }

@app.get("/api/models", tags=["配置"])
async def get_supported_models():
    """获取支持的预测模型列表"""
    return {
        "models": [
            {
                "type": "lstm",
                "name": "LSTM",
                "description": "长短期记忆网络，基准模型",
                "features": ["三门控机制", "长期记忆能力强", "参数量大"]
            },
            {
                "type": "grud", 
                "name": "GRU-D",
                "description": "专为缺失数据设计的GRU改进版",
                "features": ["处理缺失数据", "训练速度快", "预测精度高"],
                "recommended": True
            },
            {
                "type": "tcn",
                "name": "TCN", 
                "description": "时间卷积网络",
                "features": ["因果卷积", "并行计算", "大感受野"]
            },
            {
                "type": "xgboost",
                "name": "XGBoost",
                "description": "梯度提升树算法", 
                "features": ["梯度提升", "可解释性强", "部分站点表现优异"]
            }
        ]
    }

@app.get("/api/input-schema", tags=["配置"])
async def get_input_schema():
    """获取输入数据格式说明"""
    return {
        "model_parameters": {
            "seq_length": {"description": "输入序列长度", "default": settings.SEQ_LENGTH, "notes": "模型使用的历史数据窗口大小"},
            "predict_days": {"description": "预测天数", "range": f"{settings.MIN_PREDICT_DAYS}-{settings.MAX_PREDICT_DAYS}", "notes": "可预测的未来天数范围"}
        },
        "required_fields": {
            "temperature": {"description": "温度(°C)", "symbol": "T", "type": "float"},
            "oxygen": {"description": "溶解氧(mg/L)", "symbol": "O₂", "type": "float"},
            "TN": {"description": "总氮(mg/L)", "symbol": "Nₜ", "type": "float"},
            "TP": {"description": "总磷(mg/L)", "symbol": "Pₜ", "type": "float"},
            "NH": {"description": "氨氮(mg/L)", "symbol": "N_NH", "type": "float"},
            "pH": {"description": "pH值", "symbol": "pH", "type": "float"},
            "turbidity": {"description": "浊度(NTU)", "symbol": "τ", "type": "float"},
            "conductivity": {"description": "电导率(μS/cm)", "symbol": "σ", "type": "float"},
            "permanganate": {"description": "高锰酸盐指数(mg/L)", "symbol": "COD_Mn", "type": "float"},
            "rain_sum": {"description": "降雨量(mm)", "symbol": "R", "type": "float"},
            "wind_speed_10m_max": {"description": "风速(m/s)", "symbol": "u", "type": "float"},
            "shortwave_radiation_sum": {"description": "短波辐射(MJ/m²)", "symbol": "I_s", "type": "float"}
        },
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

# 全局异常处理
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """参数错误处理"""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "ValueError"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "内部服务器错误", "type": "InternalServerError"}
    )

if __name__ == "__main__":
    # 创建日志目录
    import os
    os.makedirs("logs", exist_ok=True)
    
    # 启动服务
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
