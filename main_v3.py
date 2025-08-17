#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蓝藻预测系统后端API V3 - 简化版本（只需4个核心参数）
支持多种机器学习模型（LSTM、GRU-D、TCN、XGBoost）对太湖流域6个监测站点的蓝藻密度进行预测
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
import logging
import uvicorn
import asyncio
from contextlib import asynccontextmanager

from src.models.model_manager import ModelManager
from src.services.mock_prediction_service_v3 import MockPredictionServiceV3
from src.services.historical_data_service import HistoricalDataService
from src.schemas.request_schemas_v3 import SimplePredictionRequest, BatchSimplePredictionRequest
from src.schemas.response_schemas import PredictionResponse
from src.config.settings import Settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api_v3.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局变量
model_manager = None
prediction_service = None
historical_data_service = None
settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    global model_manager, prediction_service, historical_data_service
    
    logger.info("正在初始化蓝藻预测API V3服务...")
    
    # 初始化模型管理器
    model_manager = ModelManager()
    await model_manager.initialize()
    
    # 初始化历史数据服务
    historical_data_service = HistoricalDataService()
    
    # 初始化模拟预测服务V3（演示模式）
    prediction_service = MockPredictionServiceV3()
    
    logger.info("蓝藻预测API V3服务初始化完成")
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭蓝藻预测API V3服务...")
    if historical_data_service:
        historical_data_service.clear_cache()

# 创建FastAPI应用
app = FastAPI(
    title="蓝藻预测系统API V3",
    description="基于多种机器学习模型的太湖流域蓝藻密度预测系统 - 简化版本（只需4个核心参数）",
    version="3.0.0",
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
async def get_prediction_service() -> MockPredictionServiceV3:
    """获取预测服务实例"""
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="预测服务未初始化")
    return prediction_service

async def get_historical_data_service() -> HistoricalDataService:
    """获取历史数据服务实例"""
    if historical_data_service is None:
        raise HTTPException(status_code=503, detail="历史数据服务未初始化")
    return historical_data_service

# API路由
@app.get("/", tags=["基础"])
async def root():
    """根路径，返回API基本信息"""
    return {
        "message": "蓝藻预测系统API V3 - 简化版本",
        "version": "3.0.0",
        "status": "运行中",
        "supported_stations": settings.SUPPORTED_STATIONS,
        "supported_models": settings.SUPPORTED_MODELS,
        "max_predict_days": settings.MAX_PREDICT_DAYS,
        "features": [
            "只需4个核心参数：当前日期、预测天数、预测点位、模型类型",
            "自动获取60天历史数据",
            "实时数据通过generate_fake_data.py自动更新到CSV",
            "用户无需感知数据管理过程"
        ],
        "demo_mode": "当前运行在演示模式，生成模拟预测结果",
        "simplified_input": {
            "required_parameters": [
                "current_date: 当前日期 (YYYY-MM-DD)",
                "predict_days: 预测天数 (1-30)",
                "station: 预测点位",
                "model_type: 模型类型 (lstm/grud/tcn/xgboost)"
            ]
        },
        "data_update_mechanism": "实时数据通过generate_fake_data.py自动更新到历史数据CSV文件中"
    }

@app.get("/health", tags=["基础"])
async def health_check():
    """健康检查接口"""
    try:
        # 检查服务状态
        if model_manager is None or prediction_service is None or historical_data_service is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "服务未完全初始化"}
            )
        
        # 检查模型加载状态
        loaded_models = await model_manager.get_loaded_models_status()
        
        return {
            "status": "healthy",
            "version": "3.0.0",
            "loaded_models": loaded_models,
            "services": {
                "model_manager": "running",
                "prediction_service": "running", 
                "historical_data_service": "running"
            }
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": str(e)}
        )

@app.post("/api/v3/predict", response_model=PredictionResponse, tags=["简化预测V3"])
async def predict_simple(
    request: SimplePredictionRequest,
    service: MockPredictionServiceV3 = Depends(get_prediction_service)
):
    """
    简化的蓝藻密度预测接口V3 - 只需4个核心参数
    
    **简化特性：**
    - **只需4个参数**: current_date, predict_days, station, model_type
    - **自动数据处理**: 系统自动获取截止到current_date前一天的60天历史数据
    - **零配置**: 用户无需关心数据获取、预处理等细节
    - **实时数据更新**: 新的实时数据通过generate_fake_data.py自动更新到CSV文件
    
    **参数说明：**
    - **current_date**: 当前日期，系统将自动获取此日期前60天的历史数据
    - **predict_days**: 预测天数，从current_date开始预测未来N天
    - **station**: 预测点位/监测站点名称
    - **model_type**: 预测模型类型 (lstm, grud, tcn, xgboost)
    
    **工作流程：**
    1. 根据current_date自动确定历史数据截止日期（current_date - 1天）
    2. 从CSV文件自动获取60天历史数据
    3. 执行模型预测
    4. 返回从current_date开始的预测结果
    """
    try:
        logger.info(f"收到简化预测请求: {request.station} - {request.model_type} - {request.predict_days}天")
        
        # 执行简化预测
        result = await service.predict_simple(request)
        
        logger.info(f"简化预测完成: {request.station} - 成功")
        return result
        
    except ValueError as e:
        logger.warning(f"简化预测请求参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"简化预测过程发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.post("/api/v3/batch-predict", tags=["简化预测V3"])
async def batch_predict_simple(
    request: BatchSimplePredictionRequest,
    service: MockPredictionServiceV3 = Depends(get_prediction_service)
):
    """
    批量简化预测接口V3 - 支持多个简化预测请求的并行处理
    """
    try:
        logger.info(f"收到批量简化预测请求: {len(request.requests)}个任务")
        
        results = []
        
        if request.parallel_execution:
            # 并行执行
            tasks = [service.predict_simple(req) for req in request.requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # 串行执行
            for req in request.requests:
                try:
                    result = await service.predict_simple(req)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e), "request": req.dict()})
        
        # 统计成功/失败数量
        success_count = sum(1 for r in results if isinstance(r, PredictionResponse))
        error_count = len(results) - success_count
        
        return {
            "success": True,
            "message": f"批量简化预测完成: 成功{success_count}个, 失败{error_count}个",
            "results": results,
            "summary": {
                "total": len(request.requests),
                "success": success_count,
                "errors": error_count
            }
        }
        
    except Exception as e:
        logger.error(f"批量简化预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")

@app.post("/api/v3/validate", tags=["简化预测V3"])
async def validate_simple_request(
    request: SimplePredictionRequest,
    service: MockPredictionServiceV3 = Depends(get_prediction_service)
):
    """验证简化预测请求的可行性"""
    try:
        validation_result = await service.validate_simple_request(request)
        return {
            "success": True,
            "validation_result": validation_result
        }
    except Exception as e:
        logger.error(f"简化请求验证失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")

# 配置接口
@app.get("/api/stations", tags=["配置"])
async def get_supported_stations():
    """获取支持的监测站点列表"""
    return {
        "stations": [
            {"name": "胥湖心", "name_en": "Xuhu Center"},
            {"name": "锡东水厂", "name_en": "Xidong Water Plant"},
            {"name": "平台山", "name_en": "Pingtai Mountain"},
            {"name": "tuoshan", "name_en": "Tuoshan Mountain", "name_cn": "拖山"},
            {"name": "lanshanzui", "name_en": "Lanshan Cape", "name_cn": "兰山嘴"},
            {"name": "五里湖心", "name_en": "Wulihu Center"}
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
                "description": "长短期记忆网络，基准模型"
            },
            {
                "type": "grud", 
                "name": "GRU-D",
                "description": "专为缺失数据设计的GRU改进版",
                "recommended": True
            },
            {
                "type": "tcn",
                "name": "TCN", 
                "description": "时间卷积网络"
            },
            {
                "type": "xgboost",
                "name": "XGBoost",
                "description": "梯度提升树算法"
            }
        ]
    }

@app.get("/api/v3/input-schema", tags=["配置"])
async def get_input_schema_v3():
    """获取V3版本的输入数据格式说明"""
    return {
        "api_version": "3.0",
        "simplified_design": "只需4个核心参数，零配置",
        "required_parameters": {
            "current_date": {
                "description": "当前日期",
                "format": "YYYY-MM-DD",
                "example": "2024-12-25",
                "note": "系统将自动获取此日期前60天的历史数据"
            },
            "predict_days": {
                "description": "预测天数",
                "range": "1-30",
                "example": 7,
                "note": "从current_date开始预测未来N天"
            },
            "station": {
                "description": "预测点位/监测站点名称",
                "options": ["胥湖心", "锡东水厂", "平台山", "tuoshan", "lanshanzui", "五里湖心"],
                "example": "胥湖心"
            },
            "model_type": {
                "description": "预测模型类型",
                "options": ["lstm", "grud", "tcn", "xgboost"],
                "example": "grud",
                "recommendation": "grud模型在大多数站点表现最佳"
            }
        },
        "automatic_features": [
            "自动获取60天历史数据",
            "自动数据预处理",
            "自动缺失值填充",
            "实时数据自动更新（通过generate_fake_data.py）"
        ],
        "workflow": [
            "1. 接收4个核心参数",
            "2. 根据current_date自动确定历史数据截止日期",
            "3. 从CSV文件自动获取60天历史数据",
            "4. 执行数据预处理",
            "5. 调用机器学习模型进行预测",
            "6. 返回从current_date开始的预测结果"
        ]
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
        "main_v3:app",
        host="0.0.0.0",
        port=8002,  # 使用新端口避免冲突
        reload=True,
        log_level="info"
    )