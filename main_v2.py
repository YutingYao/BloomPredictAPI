#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蓝藻预测系统后端API V2 - 优化的历史数据处理版本
支持多种机器学习模型（LSTM、GRU-D、TCN、XGBoost）对太湖流域6个监测站点的蓝藻密度进行预测
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Union
import logging
import uvicorn
import asyncio
from contextlib import asynccontextmanager

from src.models.model_manager import ModelManager
from src.services.prediction_service_v2 import PredictionServiceV2
from src.services.historical_data_service import HistoricalDataService
from src.utils.validators import validate_station, validate_model_type, validate_predict_days
from src.schemas.request_schemas_v2 import (
    PredictionRequestV2, 
    HistoricalDataQuery,
    FileUploadRequest,
    BatchPredictionRequest
)
from src.schemas.response_schemas import PredictionResponse, ModelPerformanceResponse
from src.config.settings import Settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api_v2.log', encoding='utf-8'),
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
    
    logger.info("正在初始化蓝藻预测API V2服务...")
    
    # 初始化模型管理器
    model_manager = ModelManager()
    await model_manager.initialize()
    
    # 初始化历史数据服务
    historical_data_service = HistoricalDataService()
    
    # 初始化预测服务V2
    prediction_service = PredictionServiceV2(model_manager)
    
    logger.info("蓝藻预测API V2服务初始化完成")
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭蓝藻预测API V2服务...")
    if historical_data_service:
        historical_data_service.clear_cache()

# 创建FastAPI应用
app = FastAPI(
    title="蓝藻预测系统API V2",
    description="基于多种机器学习模型的太湖流域蓝藻密度预测系统 - 优化历史数据处理版本",
    version="2.0.0",
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
async def get_prediction_service() -> PredictionServiceV2:
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
        "message": "蓝藻预测系统API V2",
        "version": "2.0.0",
        "status": "运行中",
        "supported_stations": settings.SUPPORTED_STATIONS,
        "supported_models": settings.SUPPORTED_MODELS,
        "max_predict_days": settings.MAX_PREDICT_DAYS,
        "input_seq_length": settings.SEQ_LENGTH,
        "new_features": [
            "智能历史数据获取",
            "混合数据输入模式", 
            "数据质量验证",
            "批量预测支持",
            "详细数据摘要"
        ],
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
        
        # 检查历史数据服务状态
        if historical_data_service is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "历史数据服务未初始化"}
            )
        
        # 检查模型加载状态
        loaded_models = await model_manager.get_loaded_models_status()
        
        return {
            "status": "healthy",
            "timestamp": str(asyncio.get_event_loop().time()),
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

@app.post("/api/v2/predict", response_model=PredictionResponse, tags=["预测V2"])
async def predict_algae_density_v2(
    request: PredictionRequestV2,
    service: PredictionServiceV2 = Depends(get_prediction_service)
):
    """
    蓝藻密度预测接口V2 - 支持优化的历史数据处理
    
    **新特性：**
    - **智能历史数据获取**: 自动从数据库获取60天历史数据
    - **混合输入模式**: 支持历史数据+实时数据补充
    - **数据质量验证**: 自动检测和报告数据质量问题
    - **灵活的缺失值处理**: 多种插值和填充方法
    
    **参数说明：**
    - **station**: 监测站点名称
    - **model_type**: 预测模型类型 (lstm, grud, tcn, xgboost)
    - **predict_days**: 预测天数 (1-30)
    - **data_mode**: 数据获取模式
      - `auto_historical`: 自动获取历史数据
      - `hybrid`: 历史数据+补充数据
    - **end_date**: 历史数据结束日期
    - **seq_length**: 历史数据序列长度 (默认60天)
    """
    try:
        logger.info(f"收到预测V2请求: {request.station} - {request.model_type} - {request.predict_days}天")
        
        # 验证请求
        validation_result = await service.validate_prediction_request(request)
        if not validation_result["valid"]:
            raise HTTPException(status_code=400, detail=f"请求验证失败: {validation_result['warnings']}")
        
        # 执行预测
        result = await service.predict_v2(request)
        
        logger.info(f"预测V2完成: {request.station} - 成功")
        return result
        
    except ValueError as e:
        logger.warning(f"预测V2请求参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"预测V2过程发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.post("/api/v2/batch-predict", tags=["预测V2"])
async def batch_predict_algae_density_v2(
    request: BatchPredictionRequest,
    service: PredictionServiceV2 = Depends(get_prediction_service)
):
    """
    批量预测接口V2 - 支持多个预测请求的并行处理
    """
    try:
        logger.info(f"收到批量预测V2请求: {len(request.requests)}个任务")
        
        results = []
        
        if request.parallel_execution:
            # 并行执行
            tasks = [service.predict_v2(req) for req in request.requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # 串行执行
            for req in request.requests:
                try:
                    result = await service.predict_v2(req)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e), "request": req.dict()})
        
        # 统计成功/失败数量
        success_count = sum(1 for r in results if isinstance(r, PredictionResponse))
        error_count = len(results) - success_count
        
        return {
            "success": True,
            "message": f"批量预测完成: 成功{success_count}个, 失败{error_count}个",
            "results": results,
            "summary": {
                "total": len(request.requests),
                "success": success_count,
                "errors": error_count
            }
        }
        
    except Exception as e:
        logger.error(f"批量预测V2失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")

@app.post("/api/v2/validate-request", tags=["预测V2"])
async def validate_prediction_request_v2(
    request: PredictionRequestV2,
    service: PredictionServiceV2 = Depends(get_prediction_service)
):
    """验证预测请求的可行性"""
    try:
        validation_result = await service.validate_prediction_request(request)
        return {
            "success": True,
            "validation_result": validation_result
        }
    except Exception as e:
        logger.error(f"请求验证失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")

@app.get("/api/v2/data-info/{station}", tags=["数据管理V2"])
async def get_station_data_info(
    station: str,
    service: PredictionServiceV2 = Depends(get_prediction_service)
):
    """获取站点数据摘要信息"""
    try:
        data_info = await service.get_data_info(station)
        return {
            "success": True,
            "data": data_info
        }
    except Exception as e:
        logger.error(f"获取数据信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取数据信息失败: {str(e)}")

@app.post("/api/v2/historical-data", tags=["数据管理V2"])
async def get_historical_data(
    query: HistoricalDataQuery,
    data_service: HistoricalDataService = Depends(get_historical_data_service)
):
    """获取历史数据序列（用于调试和验证）"""
    try:
        sequence = await data_service.get_historical_sequence(
            station=query.station,
            end_date=query.end_date,
            seq_length=query.seq_length
        )
        
        return {
            "success": True,
            "data": {
                "sequence_shape": sequence.shape,
                "features": data_service.base_features,
                "date_range": {
                    "end_date": query.end_date,
                    "seq_length": query.seq_length
                },
                "sequence_data": sequence.tolist()  # 转换为列表便于JSON序列化
            }
        }
    except Exception as e:
        logger.error(f"获取历史数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取历史数据失败: {str(e)}")

@app.post("/api/model-performance", response_model=ModelPerformanceResponse, tags=["模型性能"])
async def get_model_performance(
    station: Optional[str] = None,
    model_types: Optional[List[str]] = None,
    service: PredictionServiceV2 = Depends(get_prediction_service)
):
    """
    获取模型性能对比信息（兼容V1接口）
    """
    try:
        result = await service.get_model_performance(station=station, model_types=model_types)
        return result
    except Exception as e:
        logger.error(f"获取模型性能信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型性能失败: {str(e)}")

# 保留原有的配置接口
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

@app.get("/api/v2/input-schema", tags=["配置"])
async def get_input_schema_v2():
    """获取V2版本的输入数据格式说明"""
    return {
        "api_version": "2.0",
        "input_modes": {
            "auto_historical": {
                "description": "自动获取历史数据模式",
                "required_fields": ["station", "model_type", "predict_days", "end_date"],
                "optional_fields": ["seq_length", "fill_missing_method", "validate_data_quality"]
            },
            "hybrid": {
                "description": "混合数据模式：历史数据+补充数据",
                "required_fields": ["station", "model_type", "predict_days", "end_date"],
                "optional_fields": ["seq_length", "override_recent_days", "supplementary_data"]
            }
        },
        "data_parameters": {
            "seq_length": {"description": "历史数据序列长度", "default": 60, "range": "30-120"},
            "predict_days": {"description": "预测天数", "range": "1-30"},
            "fill_missing_method": {
                "description": "缺失值填充方法",
                "options": ["interpolation", "forward_fill", "backward_fill", "mean"]
            }
        },
        "required_features": {
            "temperature": {"description": "温度(°C)", "symbol": "T"},
            "oxygen": {"description": "溶解氧(mg/L)", "symbol": "O₂"},
            "TN": {"description": "总氮(mg/L)", "symbol": "Nₜ"},
            "TP": {"description": "总磷(mg/L)", "symbol": "Pₜ"},
            "NH": {"description": "氨氮(mg/L)", "symbol": "N_NH"},
            "pH": {"description": "pH值", "symbol": "pH"},
            "turbidity": {"description": "浊度(NTU)", "symbol": "τ"},
            "conductivity": {"description": "电导率(μS/cm)", "symbol": "σ"},
            "permanganate": {"description": "高锰酸盐指数(mg/L)", "symbol": "COD_Mn"},
            "rain_sum": {"description": "降雨量(mm)", "symbol": "R"},
            "wind_speed_10m_max": {"description": "风速(m/s)", "symbol": "u"},
            "shortwave_radiation_sum": {"description": "短波辐射(MJ/m²)", "symbol": "I_s"}
        },
        "advantages": [
            "无需手动构造60天×12特征=720个数值",
            "支持自动获取历史数据",
            "支持实时数据补充和覆盖",
            "内置数据质量验证",
            "灵活的缺失值处理"
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
        "main_v2:app",
        host="0.0.0.0",
        port=8001,  # 使用不同端口避免冲突
        reload=True,
        log_level="info"
    )
