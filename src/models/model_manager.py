# -*- coding: utf-8 -*-
"""
模型管理器 - 负责加载和管理所有预训练模型
"""

import pickle
import os
import asyncio
import logging
from typing import Dict, Optional, Any, List
import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor

from src.config.settings import Settings

logger = logging.getLogger(__name__)

# 定义模型架构类（必须与训练时的定义一致）
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class GRU_D(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRU_D, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class GRUDModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUDModel, self).__init__()
        self.grud = GRU_D(input_size, hidden_size, num_layers, output_size, dropout)
        
    def forward(self, x):
        return self.grud(x)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                               self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x[:, :, -1]
        return self.linear(x)

class ModelManager:
    """模型管理器类"""
    
    def __init__(self):
        self.settings = Settings()
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """初始化模型管理器，加载所有模型"""
        logger.info("开始初始化模型管理器...")
        
        # 预加载所有可用的模型
        await self._load_all_models()
        
        logger.info(f"模型管理器初始化完成，已加载 {len(self.models)} 个模型")
    
    async def _load_all_models(self):
        """异步加载所有模型"""
        tasks = []
        
        for station in self.settings.SUPPORTED_STATIONS:
            for model_type in self.settings.SUPPORTED_MODELS:
                task = asyncio.create_task(self._load_single_model(station, model_type))
                tasks.append(task)
        
        # 并行加载所有模型
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计加载结果
        success_count = sum(1 for result in results if result is True)
        logger.info(f"成功加载 {success_count}/{len(tasks)} 个模型")
    
    async def _load_single_model(self, station: str, model_type: str) -> bool:
        """加载单个模型"""
        try:
            model_key = f"{station}_{model_type}"
            
            # 构建模型文件路径
            file_pattern = self.settings.MODEL_FILE_PATTERNS.get(model_type)
            if not file_pattern:
                logger.warning(f"未知的模型类型: {model_type}")
                return False
            
            model_file = file_pattern.format(station=station)
            model_path = os.path.join(self.settings.MODEL_BASE_PATH, model_file)
            
            if not os.path.exists(model_path):
                logger.warning(f"模型文件不存在: {model_path}")
                return False
            
            # 在线程池中加载模型
            loop = asyncio.get_event_loop()
            model_data = await loop.run_in_executor(
                self.executor, self._load_model_file, model_path
            )
            
            if model_data:
                self.models[model_key] = model_data
                logger.debug(f"成功加载模型: {model_key}")
                return True
            else:
                logger.error(f"加载模型失败: {model_key}")
                return False
                
        except Exception as e:
            logger.error(f"加载模型 {station}_{model_type} 时发生错误: {e}")
            return False
    
    def _load_model_file(self, model_path: str) -> Optional[Dict[str, Any]]:
        """加载模型文件（在线程池中执行）"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 验证模型数据结构
            if not self._validate_model_data(model_data):
                logger.error(f"模型数据格式无效: {model_path}")
                return None
            
            return model_data
            
        except Exception as e:
            logger.error(f"加载模型文件失败 {model_path}: {e}")
            return None
    
    def _validate_model_data(self, model_data: Dict[str, Any]) -> bool:
        """验证模型数据格式"""
        required_keys = ['models', 'scalers']
        return all(key in model_data for key in required_keys)
    
    async def get_model(self, station: str, model_type: str, predict_days: int) -> Optional[Any]:
        """
        获取指定的模型
        
        Args:
            station: 监测站点
            model_type: 模型类型
            predict_days: 预测天数
            
        Returns:
            模型对象，如果不存在则返回None
        """
        try:
            model_key = f"{station}_{model_type}"
            
            if model_key not in self.models:
                # 尝试动态加载模型
                success = await self._load_single_model(station, model_type)
                if not success:
                    return None
            
            model_data = self.models[model_key]
            
            # 根据预测天数获取对应的模型
            if 'models' in model_data and predict_days in model_data['models']:
                return model_data['models'][predict_days]
            else:
                logger.warning(f"未找到预测天数 {predict_days} 的模型: {model_key}")
                return None
                
        except Exception as e:
            logger.error(f"获取模型失败 {station}_{model_type}_{predict_days}: {e}")
            return None
    
    async def get_scaler(self, station: str, model_type: str, predict_days: int) -> Optional[Any]:
        """
        获取指定的数据标准化器
        
        Args:
            station: 监测站点
            model_type: 模型类型  
            predict_days: 预测天数
            
        Returns:
            标准化器对象，如果不存在则返回None
        """
        try:
            model_key = f"{station}_{model_type}"
            
            if model_key not in self.models:
                return None
            
            model_data = self.models[model_key]
            
            # 获取对应的标准化器
            if 'scalers' in model_data and predict_days in model_data['scalers']:
                return model_data['scalers'][predict_days]
            else:
                logger.warning(f"未找到预测天数 {predict_days} 的标准化器: {model_key}")
                return None
                
        except Exception as e:
            logger.error(f"获取标准化器失败 {station}_{model_type}_{predict_days}: {e}")
            return None
    
    async def predict_with_model(
        self, 
        station: str, 
        model_type: str, 
        predict_days: int, 
        input_data: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        使用指定模型进行预测
        
        Args:
            station: 监测站点
            model_type: 模型类型
            predict_days: 预测天数
            input_data: 输入数据
            
        Returns:
            预测结果，如果失败则返回None
        """
        try:
            model = await self.get_model(station, model_type, predict_days)
            if model is None:
                return None
            
            # 在线程池中执行预测
            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(
                self.executor, 
                self._execute_prediction, 
                model, 
                model_type, 
                input_data
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"模型预测失败 {station}_{model_type}_{predict_days}: {e}")
            return None
    
    def _execute_prediction(self, model: Any, model_type: str, input_data: np.ndarray) -> Optional[np.ndarray]:
        """执行模型预测（在线程池中执行）"""
        try:
            if model_type.lower() == 'xgboost':
                # XGBoost模型预测
                prediction = model.predict(input_data)
            else:
                # PyTorch模型预测
                model.eval()
                with torch.no_grad():
                    # 转换为PyTorch张量
                    input_tensor = torch.FloatTensor(input_data)
                    prediction = model(input_tensor)
                    prediction = prediction.cpu().numpy()
            
            return prediction
            
        except Exception as e:
            logger.error(f"执行预测时发生错误: {e}")
            return None
    
    async def get_loaded_models_status(self) -> Dict[str, List[str]]:
        """获取已加载模型的状态"""
        status = {}
        
        for station in self.settings.SUPPORTED_STATIONS:
            station_models = []
            for model_type in self.settings.SUPPORTED_MODELS:
                model_key = f"{station}_{model_type}"
                if model_key in self.models:
                    station_models.append(model_type)
            
            if station_models:
                status[station] = station_models
        
        return status
    
    async def get_model_performance_info(self, station: str, model_type: str) -> Optional[Dict[str, Any]]:
        """获取模型性能信息"""
        try:
            model_key = f"{station}_{model_type}"
            
            if model_key not in self.models:
                return None
            
            model_data = self.models[model_key]
            
            # 如果模型数据中包含性能信息，直接返回
            if 'performance_info' in model_data:
                return model_data['performance_info']
            
            # 否则返回基础信息
            return {
                "model_type": model_type,
                "station": station,
                "available_days": list(model_data.get('models', {}).keys())
            }
            
        except Exception as e:
            logger.error(f"获取模型性能信息失败 {station}_{model_type}: {e}")
            return None
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
