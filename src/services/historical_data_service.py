# -*- coding: utf-8 -*-
"""
历史数据服务 - 处理60天历史数据的获取和预处理
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import asyncio
from functools import lru_cache

from src.config.settings import Settings

logger = logging.getLogger(__name__)

class HistoricalDataService:
    """历史数据服务类"""
    
    def __init__(self):
        self.settings = Settings()
        self.data_cache = {}  # 数据缓存
        self.base_features = [
            'temperature', 'pH', 'oxygen', 'permanganate', 'TN', 'conductivity',
            'turbidity', 'rain_sum', 'wind_speed_10m_max', 'shortwave_radiation_sum',
            'TP', 'NH'
        ]
        
        # 站点文件映射
        self.station_file_map = {
            '胥湖心': 'data/002-气象-胥湖心-merged_with_weather_with_composite_features_processed.csv',
            '锡东水厂': 'data/002-气象-锡东水厂-merged_with_weather_with_composite_features_processed.csv',
            '平台山': 'data/002-气象-平台山-merged_with_weather_with_composite_features_processed.csv',
            'tuoshan': 'data/002-气象-tuoshan-merged_with_weather_with_composite_features_processed.csv',
            'lanshanzui': 'data/002-气象-lanshanzui-merged_with_weather_with_composite_features_processed.csv',
            '五里湖心': 'data/002-气象-五里湖心-merged_with_weather_with_composite_features_processed.csv'
        }
    
    async def load_station_data(self, station: str) -> pd.DataFrame:
        """加载站点历史数据"""
        try:
            # 检查缓存
            if station in self.data_cache:
                logger.info(f"从缓存获取 {station} 站点数据")
                return self.data_cache[station]
            
            # 获取文件路径
            if station not in self.station_file_map:
                raise ValueError(f"不支持的站点: {station}")
            
            file_path = Path(self.station_file_map[station])
            if not file_path.exists():
                raise FileNotFoundError(f"数据文件不存在: {file_path}")
            
            logger.info(f"正在加载 {station} 站点数据: {file_path}")
            
            # 异步读取CSV文件
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, pd.read_csv, file_path)
            
            # 数据预处理
            df = self._preprocess_data(df)
            
            # 缓存数据
            self.data_cache[station] = df
            
            logger.info(f"成功加载 {station} 站点数据，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"加载 {station} 站点数据失败: {e}")
            raise
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        try:
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            
            # 排序
            df = df.sort_values('date').reset_index(drop=True)
            
            # 处理缺失值（特殊值 -1.0 表示缺失）
            df = df.replace(-1.0, np.nan)
            
            # 检查必需的特征列
            missing_features = [f for f in self.base_features if f not in df.columns]
            if missing_features:
                logger.warning(f"缺失特征列: {missing_features}")
            
            return df
            
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            raise
    
    async def get_historical_sequence(
        self, 
        station: str, 
        end_date: str, 
        seq_length: int = 60,
        fill_missing_method: str = "interpolation"
    ) -> np.ndarray:
        """获取历史数据序列"""
        try:
            # 加载站点数据
            df = await self.load_station_data(station)
            
            # 转换结束日期
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=seq_length - 1)
            
            logger.info(f"获取 {station} 从 {start_dt.date()} 到 {end_dt.date()} 的历史数据")
            
            # 筛选时间范围
            mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
            sequence_df = df[mask].copy()
            
            if len(sequence_df) == 0:
                raise ValueError(f"指定时间范围内没有数据: {start_dt.date()} 到 {end_dt.date()}")
            
            # 创建完整的日期范围
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
            full_df = pd.DataFrame({'date': date_range})
            
            # 合并数据，保留所有日期
            sequence_df = full_df.merge(sequence_df, on='date', how='left')
            
            # 提取特征数据
            feature_data = sequence_df[self.base_features].copy()
            
            # 处理缺失值
            feature_data = self._handle_missing_values(feature_data, fill_missing_method)
            
            # 转换为numpy数组
            sequence_array = feature_data.values
            
            # 验证数据形状
            expected_shape = (seq_length, len(self.base_features))
            if sequence_array.shape != expected_shape:
                logger.warning(f"数据形状不匹配: 期望 {expected_shape}, 实际 {sequence_array.shape}")
                
                # 如果数据不足，使用插值或重复填充
                if sequence_array.shape[0] < seq_length:
                    sequence_array = self._pad_sequence(sequence_array, seq_length)
                elif sequence_array.shape[0] > seq_length:
                    sequence_array = sequence_array[-seq_length:]
            
            logger.info(f"成功获取历史序列，形状: {sequence_array.shape}")
            return sequence_array
            
        except Exception as e:
            logger.error(f"获取历史序列失败: {e}")
            raise
    
    def _handle_missing_values(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """处理缺失值"""
        try:
            if method == "interpolation":
                # 线性插值
                data = data.interpolate(method='linear', limit_direction='both')
            elif method == "forward_fill":
                # 前向填充
                data = data.fillna(method='ffill')
            elif method == "backward_fill":
                # 后向填充
                data = data.fillna(method='bfill')
            elif method == "mean":
                # 均值填充
                data = data.fillna(data.mean())
            else:
                logger.warning(f"未知的填充方法: {method}，使用默认插值")
                data = data.interpolate(method='linear', limit_direction='both')
            
            # 如果仍有缺失值，使用均值填充
            if data.isnull().any().any():
                logger.warning("插值后仍有缺失值，使用均值填充")
                data = data.fillna(data.mean())
            
            # 如果均值填充后仍有缺失值（可能是整列为空），使用0填充
            if data.isnull().any().any():
                logger.warning("均值填充后仍有缺失值，使用0填充")
                data = data.fillna(0)
            
            return data
            
        except Exception as e:
            logger.error(f"处理缺失值失败: {e}")
            raise
    
    def _pad_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """填充序列到目标长度"""
        try:
            current_length = sequence.shape[0]
            if current_length >= target_length:
                return sequence[-target_length:]
            
            # 计算需要填充的长度
            pad_length = target_length - current_length
            
            if current_length > 0:
                # 重复最后一行
                last_row = sequence[-1:].repeat(pad_length, axis=0)
                padded_sequence = np.vstack([sequence, last_row])
            else:
                # 如果没有数据，使用零填充
                padded_sequence = np.zeros((target_length, sequence.shape[1]))
            
            return padded_sequence
            
        except Exception as e:
            logger.error(f"填充序列失败: {e}")
            raise
    
    async def apply_supplementary_data(
        self, 
        base_sequence: np.ndarray,
        supplementary_data: List[Dict[str, Any]],
        end_date: str,
        seq_length: int
    ) -> np.ndarray:
        """应用补充数据到基础序列"""
        try:
            if not supplementary_data:
                return base_sequence
            
            # 创建修改后的序列副本
            modified_sequence = base_sequence.copy()
            
            # 计算日期范围
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=seq_length - 1)
            
            for data_point in supplementary_data:
                # 解析数据点日期
                point_date = datetime.strptime(data_point['date'], '%Y-%m-%d')
                
                # 检查日期是否在序列范围内
                if point_date < start_dt or point_date > end_dt:
                    logger.warning(f"补充数据日期超出范围: {point_date.date()}")
                    continue
                
                # 计算在序列中的索引
                day_offset = (point_date - start_dt).days
                if 0 <= day_offset < seq_length:
                    # 更新对应特征的值
                    for i, feature in enumerate(self.base_features):
                        if feature in data_point and data_point[feature] is not None:
                            modified_sequence[day_offset, i] = data_point[feature]
                            logger.debug(f"更新 {point_date.date()} 的 {feature}: {data_point[feature]}")
            
            logger.info(f"成功应用 {len(supplementary_data)} 个补充数据点")
            return modified_sequence
            
        except Exception as e:
            logger.error(f"应用补充数据失败: {e}")
            raise
    
    async def validate_data_quality(self, sequence: np.ndarray) -> Dict[str, Any]:
        """验证数据质量"""
        try:
            quality_report = {
                "shape": sequence.shape,
                "missing_values": np.isnan(sequence).sum(),
                "infinite_values": np.isinf(sequence).sum(),
                "feature_stats": {},
                "quality_score": 0.0,
                "warnings": []
            }
            
            # 检查每个特征的统计信息
            for i, feature in enumerate(self.base_features):
                feature_data = sequence[:, i]
                stats = {
                    "mean": float(np.mean(feature_data)),
                    "std": float(np.std(feature_data)),
                    "min": float(np.min(feature_data)),
                    "max": float(np.max(feature_data)),
                    "missing_count": int(np.isnan(feature_data).sum())
                }
                quality_report["feature_stats"][feature] = stats
                
                # 检查异常值
                if stats["min"] < 0 and feature not in ['temperature']:
                    quality_report["warnings"].append(f"{feature} 存在负值: {stats['min']}")
                
                # 检查变化过大的值
                if stats["std"] > stats["mean"] * 2:
                    quality_report["warnings"].append(f"{feature} 方差过大，可能存在异常值")
            
            # 计算质量分数
            missing_ratio = quality_report["missing_values"] / sequence.size
            quality_score = max(0.0, 1.0 - missing_ratio * 2)  # 缺失值影响分数
            
            if quality_report["infinite_values"] > 0:
                quality_score -= 0.2
            
            if len(quality_report["warnings"]) > 0:
                quality_score -= len(quality_report["warnings"]) * 0.1
            
            quality_report["quality_score"] = max(0.0, min(1.0, quality_score))
            
            logger.info(f"数据质量评估完成，质量分数: {quality_report['quality_score']:.3f}")
            return quality_report
            
        except Exception as e:
            logger.error(f"数据质量验证失败: {e}")
            raise
    
    async def get_available_date_range(self, station: str) -> Tuple[str, str]:
        """获取站点可用的日期范围"""
        try:
            df = await self.load_station_data(station)
            min_date = df['date'].min().strftime('%Y-%m-%d')
            max_date = df['date'].max().strftime('%Y-%m-%d')
            return min_date, max_date
            
        except Exception as e:
            logger.error(f"获取日期范围失败: {e}")
            raise
    
    def clear_cache(self):
        """清除数据缓存"""
        self.data_cache.clear()
        logger.info("数据缓存已清除")
    
    async def get_data_summary(self, station: str) -> Dict[str, Any]:
        """获取数据摘要信息"""
        try:
            df = await self.load_station_data(station)
            min_date, max_date = await self.get_available_date_range(station)
            
            summary = {
                "station": station,
                "total_records": len(df),
                "date_range": {
                    "start": min_date,
                    "end": max_date,
                    "days": (pd.to_datetime(max_date) - pd.to_datetime(min_date)).days + 1
                },
                "features": self.base_features,
                "missing_data_summary": {},
                "data_coverage": {}
            }
            
            # 统计缺失数据
            for feature in self.base_features:
                if feature in df.columns:
                    missing_count = df[feature].isnull().sum()
                    summary["missing_data_summary"][feature] = {
                        "missing_count": int(missing_count),
                        "missing_ratio": float(missing_count / len(df))
                    }
            
            # 计算数据覆盖率
            total_possible = len(df) * len(self.base_features)
            total_missing = sum(df[f].isnull().sum() for f in self.base_features if f in df.columns)
            summary["data_coverage"]["overall_coverage"] = 1.0 - (total_missing / total_possible)
            
            return summary
            
        except Exception as e:
            logger.error(f"获取数据摘要失败: {e}")
            raise
