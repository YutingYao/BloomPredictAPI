# -*- coding: utf-8 -*-
"""
数据预处理工具
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """数据预处理器"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_order = [
            'temperature', 'pH', 'oxygen', 'permanganate', 'TN', 'conductivity',
            'turbidity', 'rain_sum', 'wind_speed_10m_max', 'shortwave_radiation_sum',
            'TP', 'NH'
        ]
    
    def normalize_input_data(self, data: Dict[str, float], station: str) -> np.ndarray:
        """
        标准化输入数据
        
        Args:
            data: 输入数据字典
            station: 监测站点名称
            
        Returns:
            标准化后的数据数组
        """
        try:
            # 按照固定顺序排列特征
            feature_values = []
            for feature in self.feature_order:
                if feature in data:
                    feature_values.append(data[feature])
                else:
                    logger.warning(f"缺少特征 {feature}，使用默认值 0")
                    feature_values.append(0.0)
            
            # 转换为numpy数组
            features = np.array(feature_values).reshape(1, -1)
            
            # 如果有预训练的scaler，使用它进行标准化
            scaler_key = f"{station}_scaler"
            if scaler_key in self.scalers:
                features = self.scalers[scaler_key].transform(features)
            else:
                # 使用简单的归一化（基于经验值范围）
                features = self._simple_normalize(features)
            
            return features
            
        except Exception as e:
            logger.error(f"数据标准化失败: {e}")
            raise ValueError(f"数据预处理错误: {str(e)}")
    
    def _simple_normalize(self, features: np.ndarray) -> np.ndarray:
        """
        简单的归一化方法（基于经验值范围）
        """
        # 定义各特征的期望范围（基于历史数据）
        feature_ranges = {
            0: (0, 35),      # temperature
            1: (6, 9),       # pH
            2: (0, 15),      # oxygen
            3: (0, 10),      # permanganate
            4: (0, 5),       # TN
            5: (0, 1000),    # conductivity
            6: (0, 100),     # turbidity
            7: (0, 100),     # rain_sum
            8: (0, 20),      # wind_speed_10m_max
            9: (0, 30),      # shortwave_radiation_sum
            10: (0, 1),      # TP
            11: (0, 5),      # NH
        }
        
        normalized = features.copy()
        for i, (min_val, max_val) in feature_ranges.items():
            if i < features.shape[1]:
                # 防止除零错误
                if max_val > min_val:
                    normalized[0, i] = (features[0, i] - min_val) / (max_val - min_val)
                    # 确保值在0-1范围内
                    normalized[0, i] = np.clip(normalized[0, i], 0, 1)
        
        return normalized
    
    def create_sequence_data(self, data: np.ndarray, seq_length: int = 7) -> np.ndarray:
        """
        创建时间序列数据（用于LSTM/GRU-D/TCN模型）
        
        Args:
            data: 输入数据
            seq_length: 序列长度
            
        Returns:
            时间序列数据
        """
        try:
            # 如果输入数据只有一行，重复以创建序列
            if data.shape[0] == 1:
                # 重复当前数据作为历史序列
                sequence_data = np.repeat(data, seq_length, axis=0)
                # 添加少量噪声以模拟历史变化
                noise = np.random.normal(0, 0.01, sequence_data.shape)
                sequence_data = sequence_data + noise
            else:
                # 如果有足够的历史数据，使用最后seq_length个数据点
                sequence_data = data[-seq_length:]
            
            # 重新调整形状为 (1, seq_length, features)
            return sequence_data.reshape(1, seq_length, -1)
            
        except Exception as e:
            logger.error(f"创建序列数据失败: {e}")
            raise ValueError(f"序列数据处理错误: {str(e)}")
    
    def create_xgb_features(self, data: np.ndarray, seq_length: int = 7) -> np.ndarray:
        """
        为XGBoost模型创建特征（展平+统计特征）
        
        Args:
            data: 输入数据
            seq_length: 序列长度
            
        Returns:
            XGBoost特征数据
        """
        try:
            # 创建序列数据
            sequence_data = self.create_sequence_data(data, seq_length)
            sequence_2d = sequence_data.reshape(seq_length, -1)
            
            # 展平时间窗口特征
            flattened_features = sequence_2d.flatten()
            
            # 添加统计特征
            mean_features = np.mean(sequence_2d, axis=0)
            std_features = np.std(sequence_2d, axis=0)
            max_features = np.max(sequence_2d, axis=0)
            min_features = np.min(sequence_2d, axis=0)
            
            # 组合所有特征
            combined_features = np.concatenate([
                flattened_features,
                mean_features,
                std_features,
                max_features,
                min_features
            ])
            
            return combined_features.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"创建XGBoost特征失败: {e}")
            raise ValueError(f"XGBoost特征处理错误: {str(e)}")
    
    def post_process_prediction(self, prediction: np.ndarray, model_type: str) -> List[float]:
        """
        后处理预测结果
        
        Args:
            prediction: 原始预测结果
            model_type: 模型类型
            
        Returns:
            处理后的预测值列表
        """
        try:
            # 确保预测值为列表格式
            if isinstance(prediction, np.ndarray):
                if prediction.ndim > 1:
                    prediction = prediction.flatten()
                prediction = prediction.tolist()
            elif not isinstance(prediction, list):
                prediction = [float(prediction)]
            
            # 对预测值进行合理性检查和调整
            processed_prediction = []
            for value in prediction:
                # 确保值为浮点数
                try:
                    float_value = float(value)
                    # 限制在合理范围内（增长率通常在-2到5之间）
                    float_value = np.clip(float_value, -2.0, 5.0)
                    processed_prediction.append(round(float_value, 4))
                except (ValueError, TypeError):
                    logger.warning(f"无效的预测值: {value}，使用默认值0")
                    processed_prediction.append(0.0)
            
            return processed_prediction
            
        except Exception as e:
            logger.error(f"预测结果后处理失败: {e}")
            return [0.0]  # 返回默认值
    
    def validate_and_clean_data(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        验证和清理输入数据
        
        Args:
            data: 原始输入数据
            
        Returns:
            清理后的数据字典
        """
        cleaned_data = {}
        
        for key, value in data.items():
            try:
                # 转换为浮点数
                float_value = float(value)
                
                # 检查是否为有效数值
                if np.isnan(float_value) or np.isinf(float_value):
                    logger.warning(f"无效数值 {key}: {value}，使用默认值")
                    float_value = 0.0
                
                cleaned_data[key] = float_value
                
            except (ValueError, TypeError):
                logger.warning(f"无法转换数值 {key}: {value}，使用默认值0")
                cleaned_data[key] = 0.0
        
        return cleaned_data
