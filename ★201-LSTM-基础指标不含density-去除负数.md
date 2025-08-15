请你添加一个for循环，分别训练7天的增长率density_growth模型，包括今天到明天（T+1）的增长率、今天到后天（T+2）的增长率、今天到大后天（T+3）的增长率、...、今天到（T+7）的增长率。

这样，在predict的时候，就能同时预测7天的增长率density_growth，包括今天到明天（T+1）的增长率、今天到后天（T+2）的增长率、今天到大后天（T+3）的增长率、...、今天到（T+7）的增长率。

然后就能基于今天的density，乘上增长率density_growth，得到未来的连续7天的density。具体而言，明天的density是基于今天，后天的density也是基于今天，大后天的density也是基于今天，以此类推。

对于最终得到的这些未来的连续7天的density，我们还可以分别计算RSME。一般来讲明天（T+1）的density的RSME会更准确，而（T+7）的density的RSME误差较大。

需要你注意：
- 训练的时候需要调用create_sequence_for_date，构建高维的train_sequences.append(seq)，seq包含了60天的数据
- 预测的时候，每一个step，也就是每进入新的一天：都需要加入今天的数据，弹出60天以前的数据。

请你按照下面的建议修改：
1. **数据要求说明**：
- 我们目前的数据集范围：2021年1月1日 - 2024年5月31日



2. **数据集划分建议**：
```python
# 使用固定日期划分训练集和测试集，而不是固定比例
```
  - 训练集：2021年1月1日 - 2024年1月31日。比如需要2021年1月~2月60天的数据来训练2021年3月1日的density_growth（T+1），并且还需要训练2021年3月1日的density_growth（T+2）....，并且还需要训练2021年3月1日的density_growth（T+30）.所以训练集需要在前段预留60天，后段预留30天。
  - 测试集：2023年11月1日 - 2024年5月31日。比如需要2023年11月~12月60天的数据来预测2024年1月1日的density_growth（T+1），并且还需要预测2024年1月1日的density_growth（T+2）....，并且还需要预测2024年1月1日的density_growth（T+30）.所以测试集需要在前段预留60天，后段预留30天。

3. **建议添加数据检查代码**：
```python
# 在训练开始前添加以下代码
print(f"数据集总长度: {len(df)} 天")
print(f"训练集起始日期: {df.index[0]}")
print(f"训练集结束日期: {df.index[train_size-1]}")
print(f"测试集起始日期: {df.index[train_size]}")
print(f"测试集结束日期: {df.index[-1]}")

# 检查是否有足够的历史数据
first_valid_date = df.index[seq_length]
print(f"第一个可用于训练的日期: {first_valid_date}")
print(f"可用于训练的天数: {train_size - seq_length} 天")
```



```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import logging
from torch.optim.lr_scheduler import OneCycleLR

# 设置字体
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimSun', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
chinese_font = fm.FontProperties(fname=r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\SIMSUN.TTC', size=14)
english_font = fm.FontProperties(fname=r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\TIMES.TTF', size=14)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 按照新分类重新排序的地点
location_order = ['胥湖心', '锡东水厂', '平台山', 'tuoshan', 'lanshanzui', '五里湖心']

# 重新分类的生态区域和对应的颜色 - 配色不要变
zone_info = {
    '胥湖心': {'zone_cn': '重污染与高风险区', 'zone_en': 'Heavily Polluted & High-Risk Area', 'color_primary': '#1B365D', 'color_secondary': '#B0C4DE'},  # 深海军蓝与浅蓝灰
    '锡东水厂': {'zone_cn': '重污染与高风险区', 'zone_en': 'Heavily Polluted & High-Risk Area', 'color_primary': '#1B365D', 'color_secondary': '#B0C4DE'},
    '平台山': {'zone_cn': '背景与参照区', 'zone_en': 'Background & Reference Area', 'color_primary': '#2D5016', 'color_secondary': '#9ACD32'},  # 深森林绿与浅橄榄绿
    'tuoshan': {'zone_cn': '背景与参照区', 'zone_en': 'Background & Reference Area', 'color_primary': '#2D5016', 'color_secondary': '#9ACD32'},
    'lanshanzui': {'zone_cn': '边界条件区', 'zone_en': 'Boundary Condition Area', 'color_primary': '#8B4513', 'color_secondary': '#F0E68C'},  # 深赭石色与浅卡其色
    '五里湖心': {'zone_cn': '边界条件区', 'zone_en': 'Boundary Condition Area', 'color_primary': '#8B4513', 'color_secondary': '#F0E68C'}
}

# 站点文件映射
station_files_map = {
    '胥湖心': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-胥湖心-merged_with_weather_with_composite_features_processed.csv',
    '锡东水厂': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-锡东水厂-merged_with_weather_with_composite_features_processed.csv',
    '平台山': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-平台山-merged_with_weather_with_composite_features_processed.csv',
    'tuoshan': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-tuoshan-merged_with_weather_with_composite_features_processed.csv',
    'lanshanzui': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-lanshanzui-merged_with_weather_with_composite_features_processed.csv',
    '五里湖心': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-五里湖心-merged_with_weather_with_composite_features_processed.csv'
}

# 英文站点名转中文字典
station_name_map = {
    'lanshanzui': '兰山嘴',
    'tuoshan': '拖山'
}

# 英文站点名称映射
station_name_en_map = {
    '胥湖心': 'Xuhu Center', 
    '锡东水厂': 'Xidong Water Plant', 
    '平台山': 'Pingtai Mountain',
    'tuoshan': 'Tuoshan Mountain', 
    'lanshanzui': 'Lanshan Cape', 
    '五里湖心': 'Wulihu Center'
}

# 定义中英文变量名称
variables = {
    'density': {'cn': '藻密度', 'en': 'Algae Density'}, 
    'chla': {'cn': '叶绿素a', 'en': 'Chlorophyll a'}
}

# 使用基础指标来进行预测和建模计算，建简化的LaTeX表示如下
basic_indicator_formulas = {
    'temperature': r'$T$',
    'oxygen': r'$O_2$',
    'TN': r'$N_t$',
    'TP': r'$P_t$',
    'NH': r'$N_{NH}$',
    'pH': r'$pH$',
    'turbidity': r'$\tau$',
    'conductivity': r'$\sigma$',
    'permanganate': r'$COD_{Mn}$',
    'rain_sum': r'$R$',
    'wind_speed_10m_max': r'$u$',
    'shortwave_radiation_sum': r'$I_s$'
}

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

def create_sequence_for_date(data, date_index, seq_length):
    """为指定日期创建输入序列"""
    if date_index < seq_length:
        return None
    return data[date_index - seq_length:date_index]

# 基础特征 - 使用基础指标
base_features = ['temperature', 'pH', 'oxygen', 'permanganate', 'TN', 'conductivity', 
                'turbidity', 'rain_sum', 'wind_speed_10m_max', 
                'shortwave_radiation_sum']

# 存储所有站点的数据和模型
all_station_data = {}
all_station_models = {}

# 第一个循环：数据处理和模型训练
for station_name in location_order:
    file_path = station_files_map[station_name]
    print(f"\n正在处理站点: {station_name}")
    
    # 读取数据
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 剔除6-7月的数据
    df = df[~df.index.month.isin([6, 7])]

    # 计算T+1到T+30的增长率
    for days in range(1, 31):
        df[f'density_growth_{days}d'] = (df['density'].shift(-days) - df['density']) / df['density']
        df[f'density_growth_{days}d'] = df[f'density_growth_{days}d'].interpolate(method='linear')
        window_size = 3
        df[f'density_growth_{days}d_smoothed'] = df[f'density_growth_{days}d'].rolling(window=window_size, center=True).mean()
        df[f'density_growth_{days}d_smoothed'] = df[f'density_growth_{days}d_smoothed'].interpolate(method='linear')

    # 存储站点数据
    all_station_data[station_name] = df.copy()

    # 数据集划分
    train_start_date = '2021-01-01'
    train_end_date = '2024-01-31'
    test_start_date = '2023-11-01'
    test_end_date = '2024-05-31'

    # 数据集检查
    print(f"数据集总长度: {len(df)} 天")
    print(f"数据集起始日期: {df.index[0]}")
    print(f"数据集结束日期: {df.index[-1]}")

    # 获取训练集和测试集的索引
    train_mask = (df.index >= train_start_date) & (df.index <= train_end_date)
    test_mask = (df.index >= test_start_date) & (df.index <= test_end_date)
    train_size = sum(train_mask)

    print(f"训练集起始日期: {df.index[train_mask].min()}")
    print(f"训练集结束日期: {df.index[train_mask].max()}")
    print(f"测试集起始日期: {df.index[test_mask].min()}")
    print(f"测试集结束日期: {df.index[test_mask].max()}")

    seq_length = 60
    # 检查是否有足够的历史数据
    first_valid_date = df.index[seq_length]
    print(f"第一个可用于训练的日期: {first_valid_date}")
    print(f"可用于训练的天数: {train_size - seq_length} 天")

    # 为每个预测天数创建一个模型
    station_models = {}
    
    for days in range(1, 31):
        print(f"\n训练 T+{days} 天的模型")
        
        # 添加对应天数的增长率特征
        features = base_features + [f'density_growth_{days}d', f'density_growth_{days}d_smoothed']
        
        data = df[features].values
        data = np.nan_to_num(data, nan=0)
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # 使用固定日期划分的训练集和测试集数据
        train_data = scaled_data[train_mask]
        
        # 构建训练序列
        train_sequences = []
        train_targets = []
        for i in range(seq_length, len(train_data)):
            seq = create_sequence_for_date(scaled_data, i, seq_length)
            if seq is not None:
                train_sequences.append(seq)
                train_targets.append(train_data[i, features.index(f'density_growth_{days}d_smoothed')])
        
        X_train = torch.FloatTensor(train_sequences)
        y_train = torch.FloatTensor(train_targets)
        
        train_dataset = TensorDataset(X_train, y_train.unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # 初始化和训练模型
        model = LSTMModel(input_size=len(features), hidden_size=64, num_layers=2, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        epochs = 150
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        station_models[days] = (model, scaler, features, train_mask, test_mask)
    
    all_station_models[station_name] = station_models

# 第二个循环：预测阶段
all_station_predictions = {}

for station_name in location_order:
    print(f"\n正在预测站点: {station_name}")
    
    df = all_station_data[station_name]
    station_models = all_station_models[station_name]
    
    predictions_all = {}
    actual_values_all = {}
    
    for days in range(1, 31):
        print(f"预测 T+{days} 天")
        
        model, scaler, features, train_mask, test_mask = station_models[days]
        
        data = df[features].values
        data = np.nan_to_num(data, nan=0)
        scaled_data = scaler.transform(data)
        
        test_data = scaled_data[test_mask]
        seq_length = 60
        
        # 预测阶段
        model.eval()
        predictions = []
        actual_values = []
        current_sequence = scaled_data[test_mask][0:seq_length]
        
        with torch.no_grad():
            for i in range(len(test_data)-days):
                current_sequence_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
                pred = model(current_sequence_tensor)
                
                predictions.append(pred.item())
                actual_values.append(test_data[i+days, features.index(f'density_growth_{days}d_smoothed')])
                
                current_sequence = np.vstack([
                    current_sequence[1:],
                    test_data[i]
                ])
        
        # 反归一化预测结果
        pred_full_features = np.zeros((len(predictions), len(features)))
        actual_full_features = np.zeros((len(actual_values), len(features)))
        
        growth_idx = features.index(f'density_growth_{days}d_smoothed')
        pred_full_features[:, growth_idx] = np.array(predictions)
        actual_full_features[:, growth_idx] = np.array(actual_values)
        
        predictions_all[days] = scaler.inverse_transform(pred_full_features)[:, growth_idx]
        actual_values_all[days] = scaler.inverse_transform(actual_full_features)[:, growth_idx]

    all_station_predictions[station_name] = {
        'predictions': predictions_all,
        'actuals': actual_values_all
    }


```

    
    正在处理站点: 胥湖心
    数据集总长度: 1064 天
    数据集起始日期: 2021-01-01 00:00:00
    数据集结束日期: 2024-05-31 00:00:00
    训练集起始日期: 2021-01-01 00:00:00
    训练集结束日期: 2024-01-31 00:00:00
    测试集起始日期: 2023-11-01 00:00:00
    测试集结束日期: 2024-05-31 00:00:00
    第一个可用于训练的日期: 2021-03-02 00:00:00
    可用于训练的天数: 883 天
    
    训练 T+1 天的模型
    

    C:\Users\hdec\AppData\Local\Temp\ipykernel_2304\2903565021.py:189: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_new.cpp:257.)
      X_train = torch.FloatTensor(train_sequences)
    

    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0003
    
    训练 T+2 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0010
    Epoch [150/150], Loss: 0.0005
    
    训练 T+3 天的模型
    Epoch [50/150], Loss: 0.0013
    Epoch [100/150], Loss: 0.0014
    Epoch [150/150], Loss: 0.0007
    
    训练 T+4 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0010
    Epoch [150/150], Loss: 0.0004
    
    训练 T+5 天的模型
    Epoch [50/150], Loss: 0.0038
    Epoch [100/150], Loss: 0.0143
    Epoch [150/150], Loss: 0.0006
    
    训练 T+6 天的模型
    Epoch [50/150], Loss: 0.0015
    Epoch [100/150], Loss: 0.0020
    Epoch [150/150], Loss: 0.0011
    
    训练 T+7 天的模型
    Epoch [50/150], Loss: 0.0016
    Epoch [100/150], Loss: 0.0037
    Epoch [150/150], Loss: 0.0012
    
    训练 T+8 天的模型
    Epoch [50/150], Loss: 0.0012
    Epoch [100/150], Loss: 0.0008
    Epoch [150/150], Loss: 0.0042
    
    训练 T+9 天的模型
    Epoch [50/150], Loss: 0.0026
    Epoch [100/150], Loss: 0.0006
    Epoch [150/150], Loss: 0.0003
    
    训练 T+10 天的模型
    Epoch [50/150], Loss: 0.0039
    Epoch [100/150], Loss: 0.0007
    Epoch [150/150], Loss: 0.0007
    
    训练 T+11 天的模型
    Epoch [50/150], Loss: 0.0024
    Epoch [100/150], Loss: 0.0011
    Epoch [150/150], Loss: 0.0003
    
    训练 T+12 天的模型
    Epoch [50/150], Loss: 0.0006
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0005
    
    训练 T+13 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0014
    Epoch [150/150], Loss: 0.0046
    
    训练 T+14 天的模型
    Epoch [50/150], Loss: 0.0007
    Epoch [100/150], Loss: 0.0005
    Epoch [150/150], Loss: 0.0010
    
    训练 T+15 天的模型
    Epoch [50/150], Loss: 0.0016
    Epoch [100/150], Loss: 0.0004
    Epoch [150/150], Loss: 0.0014
    
    训练 T+16 天的模型
    Epoch [50/150], Loss: 0.0009
    Epoch [100/150], Loss: 0.0012
    Epoch [150/150], Loss: 0.0003
    
    训练 T+17 天的模型
    Epoch [50/150], Loss: 0.0008
    Epoch [100/150], Loss: 0.0009
    Epoch [150/150], Loss: 0.0003
    
    训练 T+18 天的模型
    Epoch [50/150], Loss: 0.0032
    Epoch [100/150], Loss: 0.0026
    Epoch [150/150], Loss: 0.0007
    
    训练 T+19 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0021
    Epoch [150/150], Loss: 0.0007
    
    训练 T+20 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0004
    Epoch [150/150], Loss: 0.0003
    
    训练 T+21 天的模型
    Epoch [50/150], Loss: 0.0009
    Epoch [100/150], Loss: 0.0019
    Epoch [150/150], Loss: 0.0004
    
    训练 T+22 天的模型
    Epoch [50/150], Loss: 0.0033
    Epoch [100/150], Loss: 0.0006
    Epoch [150/150], Loss: 0.0017
    
    训练 T+23 天的模型
    Epoch [50/150], Loss: 0.0030
    Epoch [100/150], Loss: 0.0054
    Epoch [150/150], Loss: 0.0008
    
    训练 T+24 天的模型
    Epoch [50/150], Loss: 0.0009
    Epoch [100/150], Loss: 0.0006
    Epoch [150/150], Loss: 0.0010
    
    训练 T+25 天的模型
    Epoch [50/150], Loss: 0.0023
    Epoch [100/150], Loss: 0.0008
    Epoch [150/150], Loss: 0.0011
    
    训练 T+26 天的模型
    Epoch [50/150], Loss: 0.0006
    Epoch [100/150], Loss: 0.0016
    Epoch [150/150], Loss: 0.0009
    
    训练 T+27 天的模型
    Epoch [50/150], Loss: 0.0008
    Epoch [100/150], Loss: 0.0017
    Epoch [150/150], Loss: 0.0005
    
    训练 T+28 天的模型
    Epoch [50/150], Loss: 0.0006
    Epoch [100/150], Loss: 0.0005
    Epoch [150/150], Loss: 0.0006
    
    训练 T+29 天的模型
    Epoch [50/150], Loss: 0.0062
    Epoch [100/150], Loss: 0.0004
    Epoch [150/150], Loss: 0.0009
    
    训练 T+30 天的模型
    Epoch [50/150], Loss: 0.0031
    Epoch [100/150], Loss: 0.0009
    Epoch [150/150], Loss: 0.0006
    
    正在处理站点: 锡东水厂
    数据集总长度: 1064 天
    数据集起始日期: 2021-01-01 00:00:00
    数据集结束日期: 2024-05-31 00:00:00
    训练集起始日期: 2021-01-01 00:00:00
    训练集结束日期: 2024-01-31 00:00:00
    测试集起始日期: 2023-11-01 00:00:00
    测试集结束日期: 2024-05-31 00:00:00
    第一个可用于训练的日期: 2021-03-02 00:00:00
    可用于训练的天数: 883 天
    
    训练 T+1 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+2 天的模型
    Epoch [50/150], Loss: 0.0001
    Epoch [100/150], Loss: 0.0001
    Epoch [150/150], Loss: 0.0001
    
    训练 T+3 天的模型
    Epoch [50/150], Loss: 0.0050
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0001
    
    训练 T+4 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0005
    Epoch [150/150], Loss: 0.0050
    
    训练 T+5 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0004
    Epoch [150/150], Loss: 0.0022
    
    训练 T+6 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0008
    Epoch [150/150], Loss: 0.0003
    
    训练 T+7 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0003
    Epoch [150/150], Loss: 0.0002
    
    训练 T+8 天的模型
    Epoch [50/150], Loss: 0.0001
    Epoch [100/150], Loss: 0.0001
    Epoch [150/150], Loss: 0.0001
    
    训练 T+9 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0001
    Epoch [150/150], Loss: 0.0001
    
    训练 T+10 天的模型
    Epoch [50/150], Loss: 0.0007
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0003
    
    训练 T+11 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0001
    Epoch [150/150], Loss: 0.0018
    
    训练 T+12 天的模型
    Epoch [50/150], Loss: 0.0001
    Epoch [100/150], Loss: 0.0008
    Epoch [150/150], Loss: 0.0011
    
    训练 T+13 天的模型
    Epoch [50/150], Loss: 0.0003
    Epoch [100/150], Loss: 0.0001
    Epoch [150/150], Loss: 0.0051
    
    训练 T+14 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0005
    Epoch [150/150], Loss: 0.0001
    
    训练 T+15 天的模型
    Epoch [50/150], Loss: 0.0003
    Epoch [100/150], Loss: 0.0017
    Epoch [150/150], Loss: 0.0010
    
    训练 T+16 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0001
    
    训练 T+17 天的模型
    Epoch [50/150], Loss: 0.0008
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0026
    
    训练 T+18 天的模型
    Epoch [50/150], Loss: 0.0001
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0008
    
    训练 T+19 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+20 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0005
    Epoch [150/150], Loss: 0.0002
    
    训练 T+21 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0023
    Epoch [150/150], Loss: 0.0004
    
    训练 T+22 天的模型
    Epoch [50/150], Loss: 0.0001
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0000
    
    训练 T+23 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+24 天的模型
    Epoch [50/150], Loss: 0.0001
    Epoch [100/150], Loss: 0.0001
    Epoch [150/150], Loss: 0.0001
    
    训练 T+25 天的模型
    Epoch [50/150], Loss: 0.0009
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0001
    
    训练 T+26 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0005
    Epoch [150/150], Loss: 0.0021
    
    训练 T+27 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0019
    
    训练 T+28 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0003
    Epoch [150/150], Loss: 0.0001
    
    训练 T+29 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0025
    Epoch [150/150], Loss: 0.0001
    
    训练 T+30 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0003
    Epoch [150/150], Loss: 0.0006
    
    正在处理站点: 平台山
    数据集总长度: 1064 天
    数据集起始日期: 2021-01-01 00:00:00
    数据集结束日期: 2024-05-31 00:00:00
    训练集起始日期: 2021-01-01 00:00:00
    训练集结束日期: 2024-01-31 00:00:00
    测试集起始日期: 2023-11-01 00:00:00
    测试集结束日期: 2024-05-31 00:00:00
    第一个可用于训练的日期: 2021-03-02 00:00:00
    可用于训练的天数: 883 天
    
    训练 T+1 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+2 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+3 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+4 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+5 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+6 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+7 天的模型
    Epoch [50/150], Loss: 0.0001
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+8 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+9 天的模型
    Epoch [50/150], Loss: 0.0001
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+10 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+11 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+12 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+13 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+14 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+15 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+16 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+17 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+18 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+19 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+20 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+21 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+22 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+23 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+24 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+25 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+26 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+27 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+28 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+29 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+30 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    正在处理站点: tuoshan
    数据集总长度: 1064 天
    数据集起始日期: 2021-01-01 00:00:00
    数据集结束日期: 2024-05-31 00:00:00
    训练集起始日期: 2021-01-01 00:00:00
    训练集结束日期: 2024-01-31 00:00:00
    测试集起始日期: 2023-11-01 00:00:00
    测试集结束日期: 2024-05-31 00:00:00
    第一个可用于训练的日期: 2021-03-02 00:00:00
    可用于训练的天数: 883 天
    
    训练 T+1 天的模型
    Epoch [50/150], Loss: 0.0010
    Epoch [100/150], Loss: 0.0013
    Epoch [150/150], Loss: 0.0006
    
    训练 T+2 天的模型
    Epoch [50/150], Loss: 0.0007
    Epoch [100/150], Loss: 0.0003
    Epoch [150/150], Loss: 0.0015
    
    训练 T+3 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0001
    Epoch [150/150], Loss: 0.0002
    
    训练 T+4 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0005
    Epoch [150/150], Loss: 0.0002
    
    训练 T+5 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0004
    Epoch [150/150], Loss: 0.0005
    
    训练 T+6 天的模型
    Epoch [50/150], Loss: 0.0006
    Epoch [100/150], Loss: 0.0008
    Epoch [150/150], Loss: 0.0004
    
    训练 T+7 天的模型
    Epoch [50/150], Loss: 0.0006
    Epoch [100/150], Loss: 0.0008
    Epoch [150/150], Loss: 0.0004
    
    训练 T+8 天的模型
    Epoch [50/150], Loss: 0.0003
    Epoch [100/150], Loss: 0.0006
    Epoch [150/150], Loss: 0.0010
    
    训练 T+9 天的模型
    Epoch [50/150], Loss: 0.0008
    Epoch [100/150], Loss: 0.0003
    Epoch [150/150], Loss: 0.0013
    
    训练 T+10 天的模型
    Epoch [50/150], Loss: 0.0010
    Epoch [100/150], Loss: 0.0004
    Epoch [150/150], Loss: 0.0005
    
    训练 T+11 天的模型
    Epoch [50/150], Loss: 0.0010
    Epoch [100/150], Loss: 0.0006
    Epoch [150/150], Loss: 0.0006
    
    训练 T+12 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0003
    Epoch [150/150], Loss: 0.0010
    
    训练 T+13 天的模型
    Epoch [50/150], Loss: 0.0015
    Epoch [100/150], Loss: 0.0006
    Epoch [150/150], Loss: 0.0006
    
    训练 T+14 天的模型
    Epoch [50/150], Loss: 0.0009
    Epoch [100/150], Loss: 0.0007
    Epoch [150/150], Loss: 0.0009
    
    训练 T+15 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0007
    Epoch [150/150], Loss: 0.0012
    
    训练 T+16 天的模型
    Epoch [50/150], Loss: 0.0015
    Epoch [100/150], Loss: 0.0017
    Epoch [150/150], Loss: 0.0054
    
    训练 T+17 天的模型
    Epoch [50/150], Loss: 0.0018
    Epoch [100/150], Loss: 0.0008
    Epoch [150/150], Loss: 0.0004
    
    训练 T+18 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0102
    Epoch [150/150], Loss: 0.0014
    
    训练 T+19 天的模型
    Epoch [50/150], Loss: 0.0020
    Epoch [100/150], Loss: 0.0008
    Epoch [150/150], Loss: 0.0022
    
    训练 T+20 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0224
    Epoch [150/150], Loss: 0.0007
    
    训练 T+21 天的模型
    Epoch [50/150], Loss: 0.0046
    Epoch [100/150], Loss: 0.0004
    Epoch [150/150], Loss: 0.0024
    
    训练 T+22 天的模型
    Epoch [50/150], Loss: 0.0037
    Epoch [100/150], Loss: 0.0018
    Epoch [150/150], Loss: 0.0006
    
    训练 T+23 天的模型
    Epoch [50/150], Loss: 0.0012
    Epoch [100/150], Loss: 0.0008
    Epoch [150/150], Loss: 0.0003
    
    训练 T+24 天的模型
    Epoch [50/150], Loss: 0.0021
    Epoch [100/150], Loss: 0.0010
    Epoch [150/150], Loss: 0.0023
    
    训练 T+25 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0004
    Epoch [150/150], Loss: 0.0003
    
    训练 T+26 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0008
    Epoch [150/150], Loss: 0.0004
    
    训练 T+27 天的模型
    Epoch [50/150], Loss: 0.0007
    Epoch [100/150], Loss: 0.0098
    Epoch [150/150], Loss: 0.0038
    
    训练 T+28 天的模型
    Epoch [50/150], Loss: 0.0015
    Epoch [100/150], Loss: 0.0012
    Epoch [150/150], Loss: 0.0011
    
    训练 T+29 天的模型
    Epoch [50/150], Loss: 0.0008
    Epoch [100/150], Loss: 0.0006
    Epoch [150/150], Loss: 0.0015
    
    训练 T+30 天的模型
    Epoch [50/150], Loss: 0.0045
    Epoch [100/150], Loss: 0.0007
    Epoch [150/150], Loss: 0.0027
    
    正在处理站点: lanshanzui
    数据集总长度: 1064 天
    数据集起始日期: 2021-01-01 00:00:00
    数据集结束日期: 2024-05-31 00:00:00
    训练集起始日期: 2021-01-01 00:00:00
    训练集结束日期: 2024-01-31 00:00:00
    测试集起始日期: 2023-11-01 00:00:00
    测试集结束日期: 2024-05-31 00:00:00
    第一个可用于训练的日期: 2021-03-02 00:00:00
    可用于训练的天数: 883 天
    
    训练 T+1 天的模型
    Epoch [50/150], Loss: 0.0007
    Epoch [100/150], Loss: 0.0003
    Epoch [150/150], Loss: 0.0003
    
    训练 T+2 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0005
    Epoch [150/150], Loss: 0.0005
    
    训练 T+3 天的模型
    Epoch [50/150], Loss: 0.0006
    Epoch [100/150], Loss: 0.0017
    Epoch [150/150], Loss: 0.0009
    
    训练 T+4 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0014
    Epoch [150/150], Loss: 0.0020
    
    训练 T+5 天的模型
    Epoch [50/150], Loss: 0.0021
    Epoch [100/150], Loss: 0.0050
    Epoch [150/150], Loss: 0.0015
    
    训练 T+6 天的模型
    Epoch [50/150], Loss: 0.0010
    Epoch [100/150], Loss: 0.0005
    Epoch [150/150], Loss: 0.0011
    
    训练 T+7 天的模型
    Epoch [50/150], Loss: 0.0021
    Epoch [100/150], Loss: 0.0010
    Epoch [150/150], Loss: 0.0012
    
    训练 T+8 天的模型
    Epoch [50/150], Loss: 0.0007
    Epoch [100/150], Loss: 0.0014
    Epoch [150/150], Loss: 0.0004
    
    训练 T+9 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0020
    Epoch [150/150], Loss: 0.0016
    
    训练 T+10 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0014
    Epoch [150/150], Loss: 0.0005
    
    训练 T+11 天的模型
    Epoch [50/150], Loss: 0.0006
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0002
    
    训练 T+12 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0003
    Epoch [150/150], Loss: 0.0003
    
    训练 T+13 天的模型
    Epoch [50/150], Loss: 0.0017
    Epoch [100/150], Loss: 0.0010
    Epoch [150/150], Loss: 0.0011
    
    训练 T+14 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0007
    Epoch [150/150], Loss: 0.0010
    
    训练 T+15 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0025
    Epoch [150/150], Loss: 0.0004
    
    训练 T+16 天的模型
    Epoch [50/150], Loss: 0.0006
    Epoch [100/150], Loss: 0.0004
    Epoch [150/150], Loss: 0.0005
    
    训练 T+17 天的模型
    Epoch [50/150], Loss: 0.0008
    Epoch [100/150], Loss: 0.0003
    Epoch [150/150], Loss: 0.0002
    
    训练 T+18 天的模型
    Epoch [50/150], Loss: 0.0025
    Epoch [100/150], Loss: 0.0004
    Epoch [150/150], Loss: 0.0003
    
    训练 T+19 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0003
    Epoch [150/150], Loss: 0.0001
    
    训练 T+20 天的模型
    Epoch [50/150], Loss: 0.0009
    Epoch [100/150], Loss: 0.0010
    Epoch [150/150], Loss: 0.0001
    
    训练 T+21 天的模型
    Epoch [50/150], Loss: 0.0008
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0005
    
    训练 T+22 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0003
    
    训练 T+23 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0012
    Epoch [150/150], Loss: 0.0002
    
    训练 T+24 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0004
    Epoch [150/150], Loss: 0.0001
    
    训练 T+25 天的模型
    Epoch [50/150], Loss: 0.0003
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0004
    
    训练 T+26 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0001
    Epoch [150/150], Loss: 0.0001
    
    训练 T+27 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0004
    Epoch [150/150], Loss: 0.0002
    
    训练 T+28 天的模型
    Epoch [50/150], Loss: 0.0023
    Epoch [100/150], Loss: 0.0006
    Epoch [150/150], Loss: 0.0003
    
    训练 T+29 天的模型
    Epoch [50/150], Loss: 0.0003
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0004
    
    训练 T+30 天的模型
    Epoch [50/150], Loss: 0.0003
    Epoch [100/150], Loss: 0.0005
    Epoch [150/150], Loss: 0.0002
    
    正在处理站点: 五里湖心
    数据集总长度: 1064 天
    数据集起始日期: 2021-01-01 00:00:00
    数据集结束日期: 2024-05-31 00:00:00
    训练集起始日期: 2021-01-01 00:00:00
    训练集结束日期: 2024-01-31 00:00:00
    测试集起始日期: 2023-11-01 00:00:00
    测试集结束日期: 2024-05-31 00:00:00
    第一个可用于训练的日期: 2021-03-02 00:00:00
    可用于训练的天数: 883 天
    
    训练 T+1 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0027
    Epoch [150/150], Loss: 0.0001
    
    训练 T+2 天的模型
    Epoch [50/150], Loss: 0.0003
    Epoch [100/150], Loss: 0.0003
    Epoch [150/150], Loss: 0.0007
    
    训练 T+3 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0003
    Epoch [150/150], Loss: 0.0002
    
    训练 T+4 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0055
    Epoch [150/150], Loss: 0.0004
    
    训练 T+5 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0003
    Epoch [150/150], Loss: 0.0002
    
    训练 T+6 天的模型
    Epoch [50/150], Loss: 0.0007
    Epoch [100/150], Loss: 0.0010
    Epoch [150/150], Loss: 0.0002
    
    训练 T+7 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0020
    Epoch [150/150], Loss: 0.0002
    
    训练 T+8 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0003
    
    训练 T+9 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0000
    
    训练 T+10 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0109
    Epoch [150/150], Loss: 0.0002
    
    训练 T+11 天的模型
    Epoch [50/150], Loss: 0.0003
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0003
    
    训练 T+12 天的模型
    Epoch [50/150], Loss: 0.0006
    Epoch [100/150], Loss: 0.0092
    Epoch [150/150], Loss: 0.0002
    
    训练 T+13 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0011
    Epoch [150/150], Loss: 0.0001
    
    训练 T+14 天的模型
    Epoch [50/150], Loss: 0.0009
    Epoch [100/150], Loss: 0.0001
    Epoch [150/150], Loss: 0.0004
    
    训练 T+15 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0003
    
    训练 T+16 天的模型
    Epoch [50/150], Loss: 0.0005
    Epoch [100/150], Loss: 0.0001
    Epoch [150/150], Loss: 0.0005
    
    训练 T+17 天的模型
    Epoch [50/150], Loss: 0.0006
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0001
    
    训练 T+18 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0002
    Epoch [150/150], Loss: 0.0001
    
    训练 T+19 天的模型
    Epoch [50/150], Loss: 0.0003
    Epoch [100/150], Loss: 0.0078
    Epoch [150/150], Loss: 0.0001
    
    训练 T+20 天的模型
    Epoch [50/150], Loss: 0.0244
    Epoch [100/150], Loss: 0.0010
    Epoch [150/150], Loss: 0.0004
    
    训练 T+21 天的模型
    Epoch [50/150], Loss: 0.0001
    Epoch [100/150], Loss: 0.0000
    Epoch [150/150], Loss: 0.0000
    
    训练 T+22 天的模型
    Epoch [50/150], Loss: 0.0021
    Epoch [100/150], Loss: 0.0007
    Epoch [150/150], Loss: 0.0001
    
    训练 T+23 天的模型
    Epoch [50/150], Loss: 0.0003
    Epoch [100/150], Loss: 0.0006
    Epoch [150/150], Loss: 0.0078
    
    训练 T+24 天的模型
    Epoch [50/150], Loss: 0.0004
    Epoch [100/150], Loss: 0.0009
    Epoch [150/150], Loss: 0.0002
    
    训练 T+25 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0001
    Epoch [150/150], Loss: 0.0001
    
    训练 T+26 天的模型
    Epoch [50/150], Loss: 0.0013
    Epoch [100/150], Loss: 0.0001
    Epoch [150/150], Loss: 0.0003
    
    训练 T+27 天的模型
    Epoch [50/150], Loss: 0.0000
    Epoch [100/150], Loss: 0.0001
    Epoch [150/150], Loss: 0.0001
    
    训练 T+28 天的模型
    Epoch [50/150], Loss: 0.0002
    Epoch [100/150], Loss: 0.0001
    Epoch [150/150], Loss: 0.0000
    
    训练 T+29 天的模型
    Epoch [50/150], Loss: 0.0001
    Epoch [100/150], Loss: 0.0003
    Epoch [150/150], Loss: 0.0000
    
    训练 T+30 天的模型
    Epoch [50/150], Loss: 0.0008
    Epoch [100/150], Loss: 0.0003
    Epoch [150/150], Loss: 0.0001
    
    正在预测站点: 胥湖心
    预测 T+1 天
    预测 T+2 天
    预测 T+3 天
    预测 T+4 天
    预测 T+5 天
    预测 T+6 天
    预测 T+7 天
    预测 T+8 天
    预测 T+9 天
    预测 T+10 天
    预测 T+11 天
    预测 T+12 天
    预测 T+13 天
    预测 T+14 天
    预测 T+15 天
    预测 T+16 天
    预测 T+17 天
    预测 T+18 天
    预测 T+19 天
    预测 T+20 天
    预测 T+21 天
    预测 T+22 天
    预测 T+23 天
    预测 T+24 天
    预测 T+25 天
    预测 T+26 天
    预测 T+27 天
    预测 T+28 天
    预测 T+29 天
    预测 T+30 天
    
    正在预测站点: 锡东水厂
    预测 T+1 天
    预测 T+2 天
    预测 T+3 天
    预测 T+4 天
    预测 T+5 天
    预测 T+6 天
    预测 T+7 天
    预测 T+8 天
    预测 T+9 天
    预测 T+10 天
    预测 T+11 天
    预测 T+12 天
    预测 T+13 天
    预测 T+14 天
    预测 T+15 天
    预测 T+16 天
    预测 T+17 天
    预测 T+18 天
    预测 T+19 天
    预测 T+20 天
    预测 T+21 天
    预测 T+22 天
    预测 T+23 天
    预测 T+24 天
    预测 T+25 天
    预测 T+26 天
    预测 T+27 天
    预测 T+28 天
    预测 T+29 天
    预测 T+30 天
    
    正在预测站点: 平台山
    预测 T+1 天
    预测 T+2 天
    预测 T+3 天
    预测 T+4 天
    预测 T+5 天
    预测 T+6 天
    预测 T+7 天
    预测 T+8 天
    预测 T+9 天
    预测 T+10 天
    预测 T+11 天
    预测 T+12 天
    预测 T+13 天
    预测 T+14 天
    预测 T+15 天
    预测 T+16 天
    预测 T+17 天
    预测 T+18 天
    预测 T+19 天
    预测 T+20 天
    预测 T+21 天
    预测 T+22 天
    预测 T+23 天
    预测 T+24 天
    预测 T+25 天
    预测 T+26 天
    预测 T+27 天
    预测 T+28 天
    预测 T+29 天
    预测 T+30 天
    
    正在预测站点: tuoshan
    预测 T+1 天
    预测 T+2 天
    预测 T+3 天
    预测 T+4 天
    预测 T+5 天
    预测 T+6 天
    预测 T+7 天
    预测 T+8 天
    预测 T+9 天
    预测 T+10 天
    预测 T+11 天
    预测 T+12 天
    预测 T+13 天
    预测 T+14 天
    预测 T+15 天
    预测 T+16 天
    预测 T+17 天
    预测 T+18 天
    预测 T+19 天
    预测 T+20 天
    预测 T+21 天
    预测 T+22 天
    预测 T+23 天
    预测 T+24 天
    预测 T+25 天
    预测 T+26 天
    预测 T+27 天
    预测 T+28 天
    预测 T+29 天
    预测 T+30 天
    
    正在预测站点: lanshanzui
    预测 T+1 天
    预测 T+2 天
    预测 T+3 天
    预测 T+4 天
    预测 T+5 天
    预测 T+6 天
    预测 T+7 天
    预测 T+8 天
    预测 T+9 天
    预测 T+10 天
    预测 T+11 天
    预测 T+12 天
    预测 T+13 天
    预测 T+14 天
    预测 T+15 天
    预测 T+16 天
    预测 T+17 天
    预测 T+18 天
    预测 T+19 天
    预测 T+20 天
    预测 T+21 天
    预测 T+22 天
    预测 T+23 天
    预测 T+24 天
    预测 T+25 天
    预测 T+26 天
    预测 T+27 天
    预测 T+28 天
    预测 T+29 天
    预测 T+30 天
    
    正在预测站点: 五里湖心
    预测 T+1 天
    预测 T+2 天
    预测 T+3 天
    预测 T+4 天
    预测 T+5 天
    预测 T+6 天
    预测 T+7 天
    预测 T+8 天
    预测 T+9 天
    预测 T+10 天
    预测 T+11 天
    预测 T+12 天
    预测 T+13 天
    预测 T+14 天
    预测 T+15 天
    预测 T+16 天
    预测 T+17 天
    预测 T+18 天
    预测 T+19 天
    预测 T+20 天
    预测 T+21 天
    预测 T+22 天
    预测 T+23 天
    预测 T+24 天
    预测 T+25 天
    预测 T+26 天
    预测 T+27 天
    预测 T+28 天
    预测 T+29 天
    预测 T+30 天
    


```python

# 第三个循环：绘图和评估
for station_name in location_order:
    print(f"\n正在绘图站点: {station_name}")
    
    df = all_station_data[station_name]
    predictions_all = all_station_predictions[station_name]['predictions']
    actual_values_all = all_station_predictions[station_name]['actuals']
    
    # 计算每个预测天数的RMSE
    for days in range(1, 31):
        rmse = np.sqrt(mean_squared_error(actual_values_all[days], predictions_all[days]))
        print(f"T+{days}天预测的RMSE: {rmse:.4f}")

    # 绘制2024年1-5月的预测结果，使用站点对应的颜色
    station_zone = zone_info[station_name]
    primary_color = station_zone['color_primary']
    secondary_color = station_zone['color_secondary']
    
    plt.figure(figsize=(15, 10))
    start_date = '2024-01-01'
    end_date = '2024-05-31'

    # 获取测试集日期
    test_mask = (df.index >= '2023-11-01') & (df.index <= '2024-05-31')
    test_dates = df.index[test_mask]

    for days in range(1, 31):
        plt.subplot(6, 5, days)
        
        # 获取当前天数的预测值和实际值
        pred_values = predictions_all[days]
        actual_values = actual_values_all[days]
        
        # 确保预测日期长度与预测值长度匹配
        predictions_length = len(pred_values)
        if len(test_dates) > predictions_length:
            prediction_dates = test_dates[:predictions_length]
        else:
            prediction_dates = test_dates
        
        # 创建时间掩码来筛选2024年1-5月的数据
        mask = (prediction_dates >= start_date) & (prediction_dates <= end_date)
        
        # 确保掩码长度与数据长度匹配
        if len(mask) > len(pred_values):
            mask = mask[:len(pred_values)]
        elif len(mask) < len(pred_values):
            # 如果掩码长度不足，填充False
            mask = np.concatenate([mask, np.full(len(pred_values) - len(mask), False)])
        
        # 应用掩码筛选数据
        filtered_dates = prediction_dates[mask]
        filtered_pred = pred_values[mask]
        filtered_actual = actual_values[mask]
        
        plt.plot(filtered_dates, filtered_pred, 
                 label='预测值', alpha=0.7, color=primary_color)
        plt.plot(filtered_dates, filtered_actual, 
                 label='实际值', alpha=0.7, color=secondary_color)
        
        plt.title(f'T+{days}天藻密度增长率预测', fontproperties=chinese_font, fontsize=12)
        plt.xlabel('日期', fontproperties=chinese_font, fontsize=10)
        plt.ylabel('增长率', fontproperties=chinese_font, fontsize=10)
        plt.legend(prop=chinese_font)
        plt.grid(True)
        plt.xticks(rotation=45)
        
    plt.suptitle(f'{station_name}站点 ({station_zone["zone_cn"]}) - 藻密度增长率预测结果', 
                 fontproperties=chinese_font, fontsize=16)
    plt.tight_layout()
    plt.show()
```

    
    正在绘图站点: 胥湖心
    T+1天预测的RMSE: 0.4108
    T+2天预测的RMSE: 0.8438
    T+3天预测的RMSE: 0.9789
    T+4天预测的RMSE: 0.9583
    T+5天预测的RMSE: 1.2963
    T+6天预测的RMSE: 1.5798
    T+7天预测的RMSE: 1.7728
    T+8天预测的RMSE: 1.9075
    T+9天预测的RMSE: 2.0480
    T+10天预测的RMSE: 2.0637
    T+11天预测的RMSE: 2.2059
    T+12天预测的RMSE: 2.1643
    T+13天预测的RMSE: 2.1728
    T+14天预测的RMSE: 2.0373
    T+15天预测的RMSE: 1.9214
    T+16天预测的RMSE: 1.8065
    T+17天预测的RMSE: 1.7288
    T+18天预测的RMSE: 1.6393
    T+19天预测的RMSE: 1.5107
    T+20天预测的RMSE: 1.5528
    T+21天预测的RMSE: 1.8000
    T+22天预测的RMSE: 1.9052
    T+23天预测的RMSE: 2.0127
    T+24天预测的RMSE: 1.9548
    T+25天预测的RMSE: 2.2728
    T+26天预测的RMSE: 2.1927
    T+27天预测的RMSE: 2.5165
    T+28天预测的RMSE: 2.6571
    T+29天预测的RMSE: 3.1019
    T+30天预测的RMSE: 2.9441
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_3_1.png)
    


    
    正在绘图站点: 锡东水厂
    T+1天预测的RMSE: 0.4604
    T+2天预测的RMSE: 0.9987
    T+3天预测的RMSE: 1.0404
    T+4天预测的RMSE: 1.0112
    T+5天预测的RMSE: 1.2053
    T+6天预测的RMSE: 1.2302
    T+7天预测的RMSE: 1.5013
    T+8天预测的RMSE: 1.4735
    T+9天预测的RMSE: 1.9027
    T+10天预测的RMSE: 2.0937
    T+11天预测的RMSE: 2.5183
    T+12天预测的RMSE: 1.8628
    T+13天预测的RMSE: 1.5273
    T+14天预测的RMSE: 1.6661
    T+15天预测的RMSE: 1.4802
    T+16天预测的RMSE: 1.5089
    T+17天预测的RMSE: 1.7270
    T+18天预测的RMSE: 1.8073
    T+19天预测的RMSE: 1.9090
    T+20天预测的RMSE: 2.0768
    T+21天预测的RMSE: 1.6871
    T+22天预测的RMSE: 1.6866
    T+23天预测的RMSE: 1.6559
    T+24天预测的RMSE: 1.6325
    T+25天预测的RMSE: 1.9190
    T+26天预测的RMSE: 1.9158
    T+27天预测的RMSE: 2.0488
    T+28天预测的RMSE: 1.8730
    T+29天预测的RMSE: 2.4712
    T+30天预测的RMSE: 2.6040
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_3_3.png)
    


    
    正在绘图站点: 平台山
    T+1天预测的RMSE: 1269.4114
    T+2天预测的RMSE: 2497.8370
    T+3天预测的RMSE: 6641.9705
    T+4天预测的RMSE: 3828.8475
    T+5天预测的RMSE: 4542.9031
    T+6天预测的RMSE: 4496.4427
    T+7天预测的RMSE: 5735.9004
    T+8天预测的RMSE: 3621.1481
    T+9天预测的RMSE: 1764.3642
    T+10天预测的RMSE: 1865.6952
    T+11天预测的RMSE: 306.8187
    T+12天预测的RMSE: 5389.5882
    T+13天预测的RMSE: 2801.3056
    T+14天预测的RMSE: 1738.4652
    T+15天预测的RMSE: 1192.0137
    T+16天预测的RMSE: 902.9303
    T+17天预测的RMSE: 6123.8231
    T+18天预测的RMSE: 5489.8347
    T+19天预测的RMSE: 2510.0105
    T+20天预测的RMSE: 3593.2752
    T+21天预测的RMSE: 2761.0179
    T+22天预测的RMSE: 2343.2284
    T+23天预测的RMSE: 3102.8223
    T+24天预测的RMSE: 5522.4608
    T+25天预测的RMSE: 854.4195
    T+26天预测的RMSE: 2920.2316
    T+27天预测的RMSE: 4010.3195
    T+28天预测的RMSE: 4707.5240
    T+29天预测的RMSE: 3727.7658
    T+30天预测的RMSE: 1728.2376
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_3_5.png)
    


    
    正在绘图站点: tuoshan
    T+1天预测的RMSE: 0.4886
    T+2天预测的RMSE: 0.6834
    T+3天预测的RMSE: 1.2497
    T+4天预测的RMSE: 1.1233
    T+5天预测的RMSE: 1.0306
    T+6天预测的RMSE: 1.0837
    T+7天预测的RMSE: 1.0929
    T+8天预测的RMSE: 1.1726
    T+9天预测的RMSE: 1.0349
    T+10天预测的RMSE: 1.1879
    T+11天预测的RMSE: 1.2094
    T+12天预测的RMSE: 1.2339
    T+13天预测的RMSE: 1.1173
    T+14天预测的RMSE: 1.0928
    T+15天预测的RMSE: 1.0531
    T+16天预测的RMSE: 1.1201
    T+17天预测的RMSE: 1.1276
    T+18天预测的RMSE: 1.0471
    T+19天预测的RMSE: 1.0660
    T+20天预测的RMSE: 1.1032
    T+21天预测的RMSE: 1.0852
    T+22天预测的RMSE: 1.0511
    T+23天预测的RMSE: 1.2619
    T+24天预测的RMSE: 1.5168
    T+25天预测的RMSE: 1.5782
    T+26天预测的RMSE: 1.2438
    T+27天预测的RMSE: 1.1353
    T+28天预测的RMSE: 1.0052
    T+29天预测的RMSE: 1.0529
    T+30天预测的RMSE: 1.0819
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_3_7.png)
    


    
    正在绘图站点: lanshanzui
    T+1天预测的RMSE: 0.1087
    T+2天预测的RMSE: 0.2970
    T+3天预测的RMSE: 0.3019
    T+4天预测的RMSE: 0.2622
    T+5天预测的RMSE: 0.2772
    T+6天预测的RMSE: 0.3305
    T+7天预测的RMSE: 0.2740
    T+8天预测的RMSE: 0.2942
    T+9天预测的RMSE: 0.3250
    T+10天预测的RMSE: 0.3013
    T+11天预测的RMSE: 0.2990
    T+12天预测的RMSE: 0.3309
    T+13天预测的RMSE: 0.3957
    T+14天预测的RMSE: 0.4059
    T+15天预测的RMSE: 0.3875
    T+16天预测的RMSE: 0.3253
    T+17天预测的RMSE: 0.3751
    T+18天预测的RMSE: 0.3413
    T+19天预测的RMSE: 0.3240
    T+20天预测的RMSE: 0.3891
    T+21天预测的RMSE: 0.4910
    T+22天预测的RMSE: 0.3469
    T+23天预测的RMSE: 0.3548
    T+24天预测的RMSE: 0.4381
    T+25天预测的RMSE: 0.2972
    T+26天预测的RMSE: 0.3244
    T+27天预测的RMSE: 0.3285
    T+28天预测的RMSE: 0.4531
    T+29天预测的RMSE: 0.3250
    T+30天预测的RMSE: 0.3429
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_3_9.png)
    


    
    正在绘图站点: 五里湖心
    T+1天预测的RMSE: 0.2380
    T+2天预测的RMSE: 0.5142
    T+3天预测的RMSE: 0.6081
    T+4天预测的RMSE: 0.6714
    T+5天预测的RMSE: 0.7917
    T+6天预测的RMSE: 0.8584
    T+7天预测的RMSE: 0.8037
    T+8天预测的RMSE: 0.9179
    T+9天预测的RMSE: 0.8630
    T+10天预测的RMSE: 0.7535
    T+11天预测的RMSE: 0.8209
    T+12天预测的RMSE: 1.0545
    T+13天预测的RMSE: 1.1594
    T+14天预测的RMSE: 1.2087
    T+15天预测的RMSE: 1.3353
    T+16天预测的RMSE: 1.3653
    T+17天预测的RMSE: 1.2973
    T+18天预测的RMSE: 1.2583
    T+19天预测的RMSE: 1.2252
    T+20天预测的RMSE: 1.1499
    T+21天预测的RMSE: 0.9975
    T+22天预测的RMSE: 1.1380
    T+23天预测的RMSE: 1.2938
    T+24天预测的RMSE: 1.2690
    T+25天预测的RMSE: 1.1907
    T+26天预测的RMSE: 1.0012
    T+27天预测的RMSE: 0.9735
    T+28天预测的RMSE: 0.9017
    T+29天预测的RMSE: 0.8877
    T+30天预测的RMSE: 1.1414
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_3_11.png)
    



```python
# 保存模型和相关参数
import pickle

# 为每个站点分别保存模型和相关数据
for station_name in location_order:
    print(f"正在保存站点 {station_name} 的模型数据...")
    
    # 创建一个字典来存储当前站点的所有需要保存的内容
    station_model_data = {
        'station_name': station_name,  # 站点名称
        'models': all_station_models[station_name],  # 包含了所有天数的模型、scaler和特征
        'predictions_all': all_station_predictions[station_name]['predictions'],  # 所有预测结果
        'actual_values_all': all_station_predictions[station_name]['actuals'],  # 所有实际值
        'base_features': base_features,  # 基础特征列表
        'zone_info': zone_info[station_name]  # 站点区域信息
    }
    
    # 根据站点名称生成文件名
    filename = f'00-lstm_model_data_{station_name}-去除负数.pkl'
    
    # 保存模型数据
    with open(filename, 'wb') as f:
        pickle.dump(station_model_data, f)
    
    print(f"站点 {station_name} 的模型和相关数据已保存到 {filename}")

print("\n所有站点的模型数据保存完成！")

```

    正在保存站点 胥湖心 的模型数据...
    站点 胥湖心 的模型和相关数据已保存到 00-lstm_model_data_胥湖心-去除负数.pkl
    正在保存站点 锡东水厂 的模型数据...
    站点 锡东水厂 的模型和相关数据已保存到 00-lstm_model_data_锡东水厂-去除负数.pkl
    正在保存站点 平台山 的模型数据...
    站点 平台山 的模型和相关数据已保存到 00-lstm_model_data_平台山-去除负数.pkl
    正在保存站点 tuoshan 的模型数据...
    站点 tuoshan 的模型和相关数据已保存到 00-lstm_model_data_tuoshan-去除负数.pkl
    正在保存站点 lanshanzui 的模型数据...
    站点 lanshanzui 的模型和相关数据已保存到 00-lstm_model_data_lanshanzui-去除负数.pkl
    正在保存站点 五里湖心 的模型数据...
    站点 五里湖心 的模型和相关数据已保存到 00-lstm_model_data_五里湖心-去除负数.pkl
    
    所有站点的模型数据保存完成！
    


```python

# # 基于增长率计算2024年每一天未来30天的藻密度
# density_predictions = {}  # 存储每一天的未来30天预测密度
# density_actuals = {}     # 存储每一天的未来30天实际密度

# # 获取密度特征的索引
# density_idx = base_features.index('density')

# # 获取2024年的日期范围
# dates_2024 = pd.date_range(start='2024-01-01', end='2024-12-31')

# # 对2024年的每一天进行预测
# for current_date in dates_2024:
#     # 找到当前日期在测试集中的索引
#     date_idx = np.where(pd.to_datetime(test_dates) == current_date)[0]
    
#     if len(date_idx) > 0:
#         idx = date_idx[0]
#         # 确保索引不超出范围
#         if idx < len(test_data) and idx < len(predictions_all[1]) - 30:  # 减去30确保有足够的预测数据
#             current_density = test_data[idx][density_idx]  # 获取当前日期的藻密度
            
#             # 初始化当前日期的预测和实际值字典
#             if current_date not in density_predictions:
#                 density_predictions[current_date] = {}
#                 density_actuals[current_date] = {}
            
#             # 计算未来30天的藻密度，都是基于当前日期的密度
#             for days in range(1, 31):  # 改为1-30天
#                 # 直接用当天的密度乘以对应天数的增长率
#                 if idx + days <= len(predictions_all[days]):  # 确保索引在有效范围内
#                     density_predictions[current_date][days] = current_density * (1 + predictions_all[days][idx])
#                     density_actuals[current_date][days] = current_density * (1 + actual_values_all[days][idx])

```


```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import logging
from torch.optim.lr_scheduler import OneCycleLR

# 设置字体
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimSun', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
chinese_font = fm.FontProperties(fname=r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\SIMSUN.TTC', size=14)
english_font = fm.FontProperties(fname=r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\TIMES.TTF', size=14)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 按照新分类重新排序的地点
location_order = ['胥湖心', '锡东水厂', '平台山', 'tuoshan', 'lanshanzui', '五里湖心']

# 重新分类的生态区域和对应的颜色 - 配色不要变
zone_info = {
    '胥湖心': {'zone_cn': '重污染与高风险区', 'zone_en': 'Heavily Polluted & High-Risk Area', 'color_primary': '#1B365D', 'color_secondary': '#B0C4DE'},  # 深海军蓝与浅蓝灰
    '锡东水厂': {'zone_cn': '重污染与高风险区', 'zone_en': 'Heavily Polluted & High-Risk Area', 'color_primary': '#1B365D', 'color_secondary': '#B0C4DE'},
    '平台山': {'zone_cn': '背景与参照区', 'zone_en': 'Background & Reference Area', 'color_primary': '#2D5016', 'color_secondary': '#9ACD32'},  # 深森林绿与浅橄榄绿
    'tuoshan': {'zone_cn': '背景与参照区', 'zone_en': 'Background & Reference Area', 'color_primary': '#2D5016', 'color_secondary': '#9ACD32'},
    'lanshanzui': {'zone_cn': '边界条件区', 'zone_en': 'Boundary Condition Area', 'color_primary': '#8B4513', 'color_secondary': '#F0E68C'},  # 深赭石色与浅卡其色
    '五里湖心': {'zone_cn': '边界条件区', 'zone_en': 'Boundary Condition Area', 'color_primary': '#8B4513', 'color_secondary': '#F0E68C'}
}

# 站点文件映射
station_files_map = {
    '胥湖心': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-胥湖心-merged_with_weather_with_composite_features_processed.csv',
    '锡东水厂': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-锡东水厂-merged_with_weather_with_composite_features_processed.csv',
    '平台山': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-平台山-merged_with_weather_with_composite_features_processed.csv',
    'tuoshan': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-tuoshan-merged_with_weather_with_composite_features_processed.csv',
    'lanshanzui': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-lanshanzui-merged_with_weather_with_composite_features_processed.csv',
    '五里湖心': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-五里湖心-merged_with_weather_with_composite_features_processed.csv'
}

# 英文站点名转中文字典
station_name_map = {
    'lanshanzui': '兰山嘴',
    'tuoshan': '拖山'
}

# 英文站点名称映射
station_name_en_map = {
    '胥湖心': 'Xuhu Center', 
    '锡东水厂': 'Xidong Water Plant', 
    '平台山': 'Pingtai Mountain',
    'tuoshan': 'Tuoshan Mountain', 
    'lanshanzui': 'Lanshan Cape', 
    '五里湖心': 'Wulihu Center'
}

# 定义中英文变量名称
variables = {
    'density': {'cn': '藻密度', 'en': 'Algae Density'}, 
    'chla': {'cn': '叶绿素a', 'en': 'Chlorophyll a'}
}

# 使用基础指标来进行预测和建模计算，建简化的LaTeX表示如下
basic_indicator_formulas = {
    'temperature': r'$T$',
    'oxygen': r'$O_2$',
    'TN': r'$N_t$',
    'TP': r'$P_t$',
    'NH': r'$N_{NH}$',
    'pH': r'$pH$',
    'turbidity': r'$\tau$',
    'conductivity': r'$\sigma$',
    'permanganate': r'$COD_{Mn}$',
    'rain_sum': r'$R$',
    'wind_speed_10m_max': r'$u$',
    'shortwave_radiation_sum': r'$I_s$'
}

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

def create_sequence_for_date(data, date_index, seq_length):
    """为指定日期创建输入序列"""
    if date_index < seq_length:
        return None
    return data[date_index - seq_length:date_index]

# 基础特征 - 使用基础指标
base_features = ['temperature', 'pH', 'oxygen', 'permanganate', 'TN', 'conductivity', 
                'turbidity', 'density', 'rain_sum', 'wind_speed_10m_max', 
                'shortwave_radiation_sum']


for station_name in location_order:
    print(f"\n正在绘制站点: {station_name}")
    
    # 从保存的文件中加载该站点的模型和预测数据
    filename = f'00-lstm_model_data_{station_name}-去除负数.pkl'
    try:
        with open(filename, 'rb') as f:
            station_model_data = pickle.load(f)
        
        station_predictions = station_model_data['predictions_all']  # 从文件中获取所有预测结果
        station_actuals = station_model_data['actual_values_all']    # 从文件中获取所有实际值
        current_zone_info = station_model_data['zone_info']          # 从文件中获取站点区域信息
        
        # 获取站点颜色
        primary_color = current_zone_info['color_primary']
        secondary_color = current_zone_info['color_secondary']

    except FileNotFoundError:
        print(f"错误: 未找到站点 {station_name} 的模型数据文件 {filename}，跳过此站点。")
        continue # 跳过当前站点，继续下一个
    except KeyError as e:
        print(f"错误: 文件 {filename} 中缺少必要的键 '{e}'，跳过此站点。")
        continue # 跳过当前站点，继续下一个
    
    # 计算每个预测天数的RMSE
    print(f"\n{station_name}站点藻密度增长率预测的RMSE:")
    for days in range(1, 31):
        if days in station_predictions and days in station_actuals:
            rmse = np.sqrt(mean_squared_error(station_actuals[days], station_predictions[days]))
            print(f"T+{days}天预测的RMSE: {rmse:.4f}")
    
    # 每6天绘制一张图，共5张图
    for group in range(5):  # 0-4共5组
        plt.figure(figsize=(15, 10))
        
        for i in range(6):  # 每组6天
            days = group * 6 + i + 1
            if days <= 30:  # 确保不超过30天
                plt.subplot(3, 2, i+1)
                
                # 获取该预测天数的数据
                if days in station_predictions and days in station_actuals:
                    pred_values = station_predictions[days]
                    actual_values = station_actuals[days]
                    
                    # 创建日期索引（基于测试集长度）
                    test_start_date = '2023-11-01'
                    test_end_date = '2024-05-31'
                    prediction_dates = pd.date_range(start=test_start_date, end=test_end_date)
                    
                    # 确保数据长度匹配
                    min_length = min(len(pred_values), len(actual_values), len(prediction_dates))
                    pred_values = pred_values[:min_length]
                    actual_values = actual_values[:min_length]
                    prediction_dates = prediction_dates[:min_length]
                    
                    # 只绘制2024年1-5月的数据
                    mask = (prediction_dates >= '2024-01-01') & (prediction_dates <= '2024-05-31')
                    
                    # 检查掩码和数据长度是否匹配
                    if len(mask) == len(pred_values) and len(mask) == len(actual_values):
                        # 应用掩码筛选数据
                        filtered_dates = prediction_dates[mask]
                        filtered_pred = pred_values[mask]
                        filtered_actual = actual_values[mask]
                        
                        plt.plot(filtered_dates, filtered_pred, 
                                label='预测值', alpha=0.7, color=primary_color)
                        plt.plot(filtered_dates, filtered_actual, 
                                label='实际值', alpha=0.7, color=secondary_color)
                        
                        plt.title(f'{station_name} - T+{days}天增长率预测', 
                                 fontproperties=chinese_font, fontsize=12)
                        plt.xlabel('日期', fontproperties=chinese_font, fontsize=10)
                        plt.ylabel('增长率', fontproperties=chinese_font, fontsize=10)
                        plt.legend(prop=chinese_font)
                        plt.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        
                        # 设置刻度字体为英文字体
                        ax = plt.gca()
                        for label in ax.get_xticklabels() + ax.get_yticklabels():
                            label.set_fontproperties(english_font)
                    else:
                        plt.text(0.5, 0.5, f'数据长度不匹配\n掩码长度: {len(mask)}\n预测值长度: {len(pred_values)}', 
                               ha='center', va='center', transform=plt.gca().transAxes,
                               fontproperties=chinese_font)
        
        plt.suptitle(f'{station_name}站点 - 第{group+1}组: T+{group*6+1}天至T+{min((group+1)*6, 30)}天预测结果', 
                     fontproperties=chinese_font, fontsize=14)
        plt.tight_layout()
        plt.show()

# 计算每个站点每个预测天数的各种误差指标
print("各站点增长率预测的误差指标:")

for station_name in location_order:
    print(f"\n{station_name}站点:")
    station_predictions = all_station_predictions[station_name]['predictions']
    station_actuals = all_station_predictions[station_name]['actuals']
    
    # 计算每个预测天数的误差指标
    for days in range(1, 31):
        if days in station_predictions and days in station_actuals:
            pred = station_predictions[days]
            actual = station_actuals[days]
            
            # 计算各种误差指标
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            mse = mean_squared_error(actual, pred)
            
            # 计算相对误差指标
            mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100  # 避免除零
            
            # 计算相关系数
            correlation = np.corrcoef(actual, pred)[0, 1]
            
            # 计算决定系数 R²
            ss_res = np.sum((actual - pred) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            print(f"T+{days}天预测 - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.2f}%, 相关系数: {correlation:.4f}, R²: {r2:.4f}")


```

    
    正在绘制站点: 胥湖心
    
    胥湖心站点藻密度增长率预测的RMSE:
    T+1天预测的RMSE: 0.4108
    T+2天预测的RMSE: 0.8438
    T+3天预测的RMSE: 0.9789
    T+4天预测的RMSE: 0.9583
    T+5天预测的RMSE: 1.2963
    T+6天预测的RMSE: 1.5798
    T+7天预测的RMSE: 1.7728
    T+8天预测的RMSE: 1.9075
    T+9天预测的RMSE: 2.0480
    T+10天预测的RMSE: 2.0637
    T+11天预测的RMSE: 2.2059
    T+12天预测的RMSE: 2.1643
    T+13天预测的RMSE: 2.1728
    T+14天预测的RMSE: 2.0373
    T+15天预测的RMSE: 1.9214
    T+16天预测的RMSE: 1.8065
    T+17天预测的RMSE: 1.7288
    T+18天预测的RMSE: 1.6393
    T+19天预测的RMSE: 1.5107
    T+20天预测的RMSE: 1.5528
    T+21天预测的RMSE: 1.8000
    T+22天预测的RMSE: 1.9052
    T+23天预测的RMSE: 2.0127
    T+24天预测的RMSE: 1.9548
    T+25天预测的RMSE: 2.2728
    T+26天预测的RMSE: 2.1927
    T+27天预测的RMSE: 2.5165
    T+28天预测的RMSE: 2.6571
    T+29天预测的RMSE: 3.1019
    T+30天预测的RMSE: 2.9441
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_1.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_2.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_3.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_4.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_5.png)
    


    
    正在绘制站点: 锡东水厂
    
    锡东水厂站点藻密度增长率预测的RMSE:
    T+1天预测的RMSE: 0.4604
    T+2天预测的RMSE: 0.9987
    T+3天预测的RMSE: 1.0404
    T+4天预测的RMSE: 1.0112
    T+5天预测的RMSE: 1.2053
    T+6天预测的RMSE: 1.2302
    T+7天预测的RMSE: 1.5013
    T+8天预测的RMSE: 1.4735
    T+9天预测的RMSE: 1.9027
    T+10天预测的RMSE: 2.0937
    T+11天预测的RMSE: 2.5183
    T+12天预测的RMSE: 1.8628
    T+13天预测的RMSE: 1.5273
    T+14天预测的RMSE: 1.6661
    T+15天预测的RMSE: 1.4802
    T+16天预测的RMSE: 1.5089
    T+17天预测的RMSE: 1.7270
    T+18天预测的RMSE: 1.8073
    T+19天预测的RMSE: 1.9090
    T+20天预测的RMSE: 2.0768
    T+21天预测的RMSE: 1.6871
    T+22天预测的RMSE: 1.6866
    T+23天预测的RMSE: 1.6559
    T+24天预测的RMSE: 1.6325
    T+25天预测的RMSE: 1.9190
    T+26天预测的RMSE: 1.9158
    T+27天预测的RMSE: 2.0488
    T+28天预测的RMSE: 1.8730
    T+29天预测的RMSE: 2.4712
    T+30天预测的RMSE: 2.6040
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_7.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_8.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_9.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_10.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_11.png)
    


    
    正在绘制站点: 平台山
    
    平台山站点藻密度增长率预测的RMSE:
    T+1天预测的RMSE: 1269.4114
    T+2天预测的RMSE: 2497.8370
    T+3天预测的RMSE: 6641.9705
    T+4天预测的RMSE: 3828.8475
    T+5天预测的RMSE: 4542.9031
    T+6天预测的RMSE: 4496.4427
    T+7天预测的RMSE: 5735.9004
    T+8天预测的RMSE: 3621.1481
    T+9天预测的RMSE: 1764.3642
    T+10天预测的RMSE: 1865.6952
    T+11天预测的RMSE: 306.8187
    T+12天预测的RMSE: 5389.5882
    T+13天预测的RMSE: 2801.3056
    T+14天预测的RMSE: 1738.4652
    T+15天预测的RMSE: 1192.0137
    T+16天预测的RMSE: 902.9303
    T+17天预测的RMSE: 6123.8231
    T+18天预测的RMSE: 5489.8347
    T+19天预测的RMSE: 2510.0105
    T+20天预测的RMSE: 3593.2752
    T+21天预测的RMSE: 2761.0179
    T+22天预测的RMSE: 2343.2284
    T+23天预测的RMSE: 3102.8223
    T+24天预测的RMSE: 5522.4608
    T+25天预测的RMSE: 854.4195
    T+26天预测的RMSE: 2920.2316
    T+27天预测的RMSE: 4010.3195
    T+28天预测的RMSE: 4707.5240
    T+29天预测的RMSE: 3727.7658
    T+30天预测的RMSE: 1728.2376
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_13.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_14.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_15.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_16.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_17.png)
    


    
    正在绘制站点: tuoshan
    
    tuoshan站点藻密度增长率预测的RMSE:
    T+1天预测的RMSE: 0.4886
    T+2天预测的RMSE: 0.6834
    T+3天预测的RMSE: 1.2497
    T+4天预测的RMSE: 1.1233
    T+5天预测的RMSE: 1.0306
    T+6天预测的RMSE: 1.0837
    T+7天预测的RMSE: 1.0929
    T+8天预测的RMSE: 1.1726
    T+9天预测的RMSE: 1.0349
    T+10天预测的RMSE: 1.1879
    T+11天预测的RMSE: 1.2094
    T+12天预测的RMSE: 1.2339
    T+13天预测的RMSE: 1.1173
    T+14天预测的RMSE: 1.0928
    T+15天预测的RMSE: 1.0531
    T+16天预测的RMSE: 1.1201
    T+17天预测的RMSE: 1.1276
    T+18天预测的RMSE: 1.0471
    T+19天预测的RMSE: 1.0660
    T+20天预测的RMSE: 1.1032
    T+21天预测的RMSE: 1.0852
    T+22天预测的RMSE: 1.0511
    T+23天预测的RMSE: 1.2619
    T+24天预测的RMSE: 1.5168
    T+25天预测的RMSE: 1.5782
    T+26天预测的RMSE: 1.2438
    T+27天预测的RMSE: 1.1353
    T+28天预测的RMSE: 1.0052
    T+29天预测的RMSE: 1.0529
    T+30天预测的RMSE: 1.0819
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_19.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_20.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_21.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_22.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_23.png)
    


    
    正在绘制站点: lanshanzui
    
    lanshanzui站点藻密度增长率预测的RMSE:
    T+1天预测的RMSE: 0.1087
    T+2天预测的RMSE: 0.2970
    T+3天预测的RMSE: 0.3019
    T+4天预测的RMSE: 0.2622
    T+5天预测的RMSE: 0.2772
    T+6天预测的RMSE: 0.3305
    T+7天预测的RMSE: 0.2740
    T+8天预测的RMSE: 0.2942
    T+9天预测的RMSE: 0.3250
    T+10天预测的RMSE: 0.3013
    T+11天预测的RMSE: 0.2990
    T+12天预测的RMSE: 0.3309
    T+13天预测的RMSE: 0.3957
    T+14天预测的RMSE: 0.4059
    T+15天预测的RMSE: 0.3875
    T+16天预测的RMSE: 0.3253
    T+17天预测的RMSE: 0.3751
    T+18天预测的RMSE: 0.3413
    T+19天预测的RMSE: 0.3240
    T+20天预测的RMSE: 0.3891
    T+21天预测的RMSE: 0.4910
    T+22天预测的RMSE: 0.3469
    T+23天预测的RMSE: 0.3548
    T+24天预测的RMSE: 0.4381
    T+25天预测的RMSE: 0.2972
    T+26天预测的RMSE: 0.3244
    T+27天预测的RMSE: 0.3285
    T+28天预测的RMSE: 0.4531
    T+29天预测的RMSE: 0.3250
    T+30天预测的RMSE: 0.3429
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_25.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_26.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_27.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_28.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_29.png)
    


    
    正在绘制站点: 五里湖心
    
    五里湖心站点藻密度增长率预测的RMSE:
    T+1天预测的RMSE: 0.2380
    T+2天预测的RMSE: 0.5142
    T+3天预测的RMSE: 0.6081
    T+4天预测的RMSE: 0.6714
    T+5天预测的RMSE: 0.7917
    T+6天预测的RMSE: 0.8584
    T+7天预测的RMSE: 0.8037
    T+8天预测的RMSE: 0.9179
    T+9天预测的RMSE: 0.8630
    T+10天预测的RMSE: 0.7535
    T+11天预测的RMSE: 0.8209
    T+12天预测的RMSE: 1.0545
    T+13天预测的RMSE: 1.1594
    T+14天预测的RMSE: 1.2087
    T+15天预测的RMSE: 1.3353
    T+16天预测的RMSE: 1.3653
    T+17天预测的RMSE: 1.2973
    T+18天预测的RMSE: 1.2583
    T+19天预测的RMSE: 1.2252
    T+20天预测的RMSE: 1.1499
    T+21天预测的RMSE: 0.9975
    T+22天预测的RMSE: 1.1380
    T+23天预测的RMSE: 1.2938
    T+24天预测的RMSE: 1.2690
    T+25天预测的RMSE: 1.1907
    T+26天预测的RMSE: 1.0012
    T+27天预测的RMSE: 0.9735
    T+28天预测的RMSE: 0.9017
    T+29天预测的RMSE: 0.8877
    T+30天预测的RMSE: 1.1414
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_31.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_32.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_33.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_34.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_6_35.png)
    


    各站点增长率预测的误差指标:
    
    胥湖心站点:
    T+1天预测 - RMSE: 0.4108, MAE: 0.2343, MSE: 0.1687, MAPE: 81474857.35%, 相关系数: 0.2062, R²: -0.2732
    T+2天预测 - RMSE: 0.8438, MAE: 0.4748, MSE: 0.7120, MAPE: 294886085.31%, 相关系数: 0.0576, R²: -0.6443
    T+3天预测 - RMSE: 0.9789, MAE: 0.6260, MSE: 0.9583, MAPE: 281325756.73%, 相关系数: 0.0208, R²: -0.7805
    T+4天预测 - RMSE: 0.9583, MAE: 0.6507, MSE: 0.9183, MAPE: 212176446.87%, 相关系数: 0.0100, R²: -0.7435
    T+5天预测 - RMSE: 1.2963, MAE: 0.8678, MSE: 1.6803, MAPE: 600491108.91%, 相关系数: -0.0685, R²: -1.2555
    T+6天预测 - RMSE: 1.5798, MAE: 1.0517, MSE: 2.4956, MAPE: 559683179.39%, 相关系数: -0.1440, R²: -1.4045
    T+7天预测 - RMSE: 1.7728, MAE: 1.1565, MSE: 3.1427, MAPE: 180128208.30%, 相关系数: -0.2105, R²: -1.3373
    T+8天预测 - RMSE: 1.9075, MAE: 1.2668, MSE: 3.6386, MAPE: 101622621.11%, 相关系数: -0.2681, R²: -1.3486
    T+9天预测 - RMSE: 2.0480, MAE: 1.3913, MSE: 4.1943, MAPE: 50296034.32%, 相关系数: -0.2948, R²: -1.5030
    T+10天预测 - RMSE: 2.0637, MAE: 1.4125, MSE: 4.2589, MAPE: 39312251.47%, 相关系数: -0.3152, R²: -1.5481
    T+11天预测 - RMSE: 2.2059, MAE: 1.4853, MSE: 4.8661, MAPE: 9935066.95%, 相关系数: -0.3282, R²: -1.6104
    T+12天预测 - RMSE: 2.1643, MAE: 1.4749, MSE: 4.6844, MAPE: 476.75%, 相关系数: -0.2648, R²: -1.5115
    T+13天预测 - RMSE: 2.1728, MAE: 1.4592, MSE: 4.7211, MAPE: 353.06%, 相关系数: -0.2160, R²: -1.3940
    T+14天预测 - RMSE: 2.0373, MAE: 1.3306, MSE: 4.1506, MAPE: 399.02%, 相关系数: -0.1666, R²: -1.2249
    T+15天预测 - RMSE: 1.9214, MAE: 1.2682, MSE: 3.6918, MAPE: 676.90%, 相关系数: -0.1334, R²: -1.2139
    T+16天预测 - RMSE: 1.8065, MAE: 1.2458, MSE: 3.2633, MAPE: 1137.77%, 相关系数: -0.0655, R²: -0.9884
    T+17天预测 - RMSE: 1.7288, MAE: 1.1932, MSE: 2.9887, MAPE: 478.05%, 相关系数: -0.0173, R²: -1.1444
    T+18天预测 - RMSE: 1.6393, MAE: 1.0896, MSE: 2.6873, MAPE: 308.21%, 相关系数: 0.0305, R²: -0.9773
    T+19天预测 - RMSE: 1.5107, MAE: 0.9855, MSE: 2.2823, MAPE: 1117.95%, 相关系数: 0.0224, R²: -0.8177
    T+20天预测 - RMSE: 1.5528, MAE: 1.1030, MSE: 2.4113, MAPE: 353.52%, 相关系数: -0.0340, R²: -1.1196
    T+21天预测 - RMSE: 1.8000, MAE: 1.2605, MSE: 3.2399, MAPE: 296.82%, 相关系数: -0.0437, R²: -1.0801
    T+22天预测 - RMSE: 1.9052, MAE: 1.3239, MSE: 3.6298, MAPE: 1121.64%, 相关系数: -0.0911, R²: -1.0771
    T+23天预测 - RMSE: 2.0127, MAE: 1.2972, MSE: 4.0509, MAPE: 474.79%, 相关系数: -0.1161, R²: -1.3118
    T+24天预测 - RMSE: 1.9548, MAE: 1.2749, MSE: 3.8212, MAPE: 1281.64%, 相关系数: -0.1609, R²: -2.2700
    T+25天预测 - RMSE: 2.2728, MAE: 1.4247, MSE: 5.1658, MAPE: 533.69%, 相关系数: -0.1681, R²: -2.5050
    T+26天预测 - RMSE: 2.1927, MAE: 1.3902, MSE: 4.8079, MAPE: 704.87%, 相关系数: -0.2180, R²: -2.7614
    T+27天预测 - RMSE: 2.5165, MAE: 1.4821, MSE: 6.3328, MAPE: 771.44%, 相关系数: -0.1886, R²: -2.6235
    T+28天预测 - RMSE: 2.6571, MAE: 1.5086, MSE: 7.0600, MAPE: 994.42%, 相关系数: -0.1888, R²: -2.5711
    T+29天预测 - RMSE: 3.1019, MAE: 1.6731, MSE: 9.6216, MAPE: 652.18%, 相关系数: -0.2037, R²: -3.5732
    T+30天预测 - RMSE: 2.9441, MAE: 1.6173, MSE: 8.6679, MAPE: 1313.92%, 相关系数: -0.2121, R²: -3.0135
    
    锡东水厂站点:
    T+1天预测 - RMSE: 0.4604, MAE: 0.2985, MSE: 0.2119, MAPE: 143997436.64%, 相关系数: 0.2558, R²: -0.0867
    T+2天预测 - RMSE: 0.9987, MAE: 0.5717, MSE: 0.9975, MAPE: 208020465.89%, 相关系数: -0.0496, R²: -0.7773
    T+3天预测 - RMSE: 1.0404, MAE: 0.6630, MSE: 1.0823, MAPE: 106684331.78%, 相关系数: -0.0708, R²: -0.9916
    T+4天预测 - RMSE: 1.0112, MAE: 0.7209, MSE: 1.0226, MAPE: 119438428.78%, 相关系数: -0.0384, R²: -1.0451
    T+5天预测 - RMSE: 1.2053, MAE: 0.8190, MSE: 1.4528, MAPE: 98193382.46%, 相关系数: 0.0586, R²: -1.1265
    T+6天预测 - RMSE: 1.2302, MAE: 0.8302, MSE: 1.5133, MAPE: 128273598.59%, 相关系数: -0.0863, R²: -0.8355
    T+7天预测 - RMSE: 1.5013, MAE: 0.9585, MSE: 2.2538, MAPE: 114983780.98%, 相关系数: -0.1427, R²: -1.2669
    T+8天预测 - RMSE: 1.4735, MAE: 0.9441, MSE: 2.1713, MAPE: 91318743.31%, 相关系数: -0.1636, R²: -1.0646
    T+9天预测 - RMSE: 1.9027, MAE: 1.2210, MSE: 3.6202, MAPE: 103491130.69%, 相关系数: -0.1667, R²: -1.2359
    T+10天预测 - RMSE: 2.0937, MAE: 1.2921, MSE: 4.3835, MAPE: 665.94%, 相关系数: -0.2265, R²: -1.0906
    T+11天预测 - RMSE: 2.5183, MAE: 1.5105, MSE: 6.3417, MAPE: 573.38%, 相关系数: -0.1711, R²: -1.1878
    T+12天预测 - RMSE: 1.8628, MAE: 1.2137, MSE: 3.4699, MAPE: 879.82%, 相关系数: -0.2023, R²: -1.1392
    T+13天预测 - RMSE: 1.5273, MAE: 1.1061, MSE: 2.3328, MAPE: 340.73%, 相关系数: -0.1875, R²: -1.2600
    T+14天预测 - RMSE: 1.6661, MAE: 1.2892, MSE: 2.7758, MAPE: 728.24%, 相关系数: -0.3251, R²: -1.8982
    T+15天预测 - RMSE: 1.4802, MAE: 1.1758, MSE: 2.1911, MAPE: 573.88%, 相关系数: -0.2133, R²: -1.4771
    T+16天预测 - RMSE: 1.5089, MAE: 1.1915, MSE: 2.2768, MAPE: 580.99%, 相关系数: -0.2163, R²: -1.0116
    T+17天预测 - RMSE: 1.7270, MAE: 1.2861, MSE: 2.9824, MAPE: 2110.47%, 相关系数: -0.1694, R²: -1.3036
    T+18天预测 - RMSE: 1.8073, MAE: 1.2806, MSE: 3.2664, MAPE: 343.20%, 相关系数: -0.1086, R²: -1.0424
    T+19天预测 - RMSE: 1.9090, MAE: 1.3709, MSE: 3.6445, MAPE: 2027.14%, 相关系数: -0.1043, R²: -1.4567
    T+20天预测 - RMSE: 2.0768, MAE: 1.4359, MSE: 4.3131, MAPE: 341.91%, 相关系数: -0.0074, R²: -1.8758
    T+21天预测 - RMSE: 1.6871, MAE: 1.2795, MSE: 2.8463, MAPE: 388.84%, 相关系数: -0.0004, R²: -1.0337
    T+22天预测 - RMSE: 1.6866, MAE: 1.2670, MSE: 2.8447, MAPE: 370.69%, 相关系数: 0.0130, R²: -1.0776
    T+23天预测 - RMSE: 1.6559, MAE: 1.3017, MSE: 2.7419, MAPE: 383.38%, 相关系数: 0.1442, R²: -0.5700
    T+24天预测 - RMSE: 1.6325, MAE: 1.2491, MSE: 2.6649, MAPE: 357.25%, 相关系数: 0.0927, R²: -0.6716
    T+25天预测 - RMSE: 1.9190, MAE: 1.3892, MSE: 3.6826, MAPE: 698.52%, 相关系数: -0.0696, R²: -1.0148
    T+26天预测 - RMSE: 1.9158, MAE: 1.3902, MSE: 3.6704, MAPE: 751.57%, 相关系数: -0.0382, R²: -1.0290
    T+27天预测 - RMSE: 2.0488, MAE: 1.5112, MSE: 4.1976, MAPE: 679.84%, 相关系数: -0.1364, R²: -1.1981
    T+28天预测 - RMSE: 1.8730, MAE: 1.4078, MSE: 3.5079, MAPE: 1100.17%, 相关系数: -0.0338, R²: -0.7273
    T+29天预测 - RMSE: 2.4712, MAE: 1.7554, MSE: 6.1066, MAPE: 1375.05%, 相关系数: -0.2162, R²: -1.4010
    T+30天预测 - RMSE: 2.6040, MAE: 1.8047, MSE: 6.7807, MAPE: 772.17%, 相关系数: -0.1465, R²: -1.1584
    
    平台山站点:
    T+1天预测 - RMSE: 1269.4114, MAE: 1268.6285, MSE: 1611405.2299, MAPE: 12686285299323.51%, 相关系数: nan, R²: -34161790874744064.0000
    T+2天预测 - RMSE: 2497.8370, MAE: 2492.7535, MSE: 6239189.5859, MAPE: 24927535425902.57%, 相关系数: nan, R²: -131646900262109152.0000
    T+3天预测 - RMSE: 6641.9705, MAE: 6639.3141, MSE: 44115772.1482, MAPE: 66393141290609.88%, 相关系数: nan, R²: -926431215112957696.0000
    T+4天预测 - RMSE: 3828.8475, MAE: 3827.1498, MSE: 14660073.1349, MAPE: 38271497768561.27%, 相关系数: nan, R²: -306395528518884480.0000
    T+5天预测 - RMSE: 4542.9031, MAE: 4540.9578, MSE: 20637968.7056, MAPE: 45409578317671.94%, 相关系数: nan, R²: -429269749075523200.0000
    T+6天预测 - RMSE: 4496.4427, MAE: 4494.7157, MSE: 20217996.9087, MAPE: 44947156846646.66%, 相关系数: nan, R²: -418512536009291072.0000
    T+7天预测 - RMSE: 5735.9004, MAE: 5734.4093, MSE: 32900553.6865, MAPE: 57344092607632.40%, 相关系数: nan, R²: -677751405942908800.0000
    T+8天预测 - RMSE: 3621.1481, MAE: 3618.6697, MSE: 13112713.8267, MAPE: 36186696510574.18%, 相关系数: nan, R²: -268810633447573760.0000
    T+9天预测 - RMSE: 1764.3642, MAE: 1752.3098, MSE: 3112980.9099, MAPE: 17523098357926.37%, 相关系数: nan, R²: -63504810562466216.0000
    T+10天预测 - RMSE: 1865.6952, MAE: 1863.3920, MSE: 3480818.7153, MAPE: 18633919826376.64%, 相关系数: nan, R²: -70660619920252944.0000
    T+11天预测 - RMSE: 306.8187, MAE: 270.2065, MSE: 94137.7232, MAPE: 2702064501800.19%, 相关系数: nan, R²: -1901582008732098.7500
    T+12天预测 - RMSE: 5389.5882, MAE: 5389.0763, MSE: 29047660.6809, MAPE: 53890762513149.74%, 相关系数: nan, R²: -583857979686576256.0000
    T+13天预测 - RMSE: 2801.3056, MAE: 2800.0296, MSE: 7847313.0194, MAPE: 28000296263118.71%, 相关系数: nan, R²: -156946260387300832.0000
    T+14天预测 - RMSE: 1738.4652, MAE: 1734.2631, MSE: 3022261.0837, MAPE: 17342631344834.25%, 相关系数: nan, R²: -60142995566069176.0000
    T+15天预测 - RMSE: 1192.0137, MAE: 1179.9513, MSE: 1420896.7139, MAPE: 11799513015804.45%, 相关系数: nan, R²: -28133754934655248.0000
    T+16天预测 - RMSE: 902.9303, MAE: 886.3998, MSE: 815283.1971, MAPE: 8863998298573.66%, 相关系数: nan, R²: -16061078982760808.0000
    T+17天预测 - RMSE: 6123.8231, MAE: 6122.0712, MSE: 37501209.6560, MAPE: 61220712347754.17%, 相关系数: nan, R²: -735023709256828160.0000
    T+18天预测 - RMSE: 5489.8347, MAE: 5488.1633, MSE: 30138284.7937, MAPE: 54881633306445.91%, 相关系数: nan, R²: -587696553476391936.0000
    T+19天预测 - RMSE: 2510.0105, MAE: 2504.4010, MSE: 6300152.4846, MAPE: 25044010345399.48%, 相关系数: nan, R²: -122222958201930800.0000
    T+20天预测 - RMSE: 3593.2752, MAE: 3588.4103, MSE: 12911626.8582, MAPE: 35884103486132.60%, 相关系数: nan, R²: -249194398362813312.0000
    T+21天预测 - RMSE: 2761.0179, MAE: 2758.0520, MSE: 7623219.9680, MAPE: 27580519544274.87%, 相关系数: nan, R²: -146365823384755520.0000
    T+22天预测 - RMSE: 2343.2284, MAE: 2341.5270, MSE: 5490719.2244, MAPE: 23415270293370.91%, 相关系数: nan, R²: -104872737185640432.0000
    T+23天预测 - RMSE: 3102.8223, MAE: 3097.2664, MSE: 9627506.1765, MAPE: 30972664385779.92%, 相关系数: nan, R²: -182922617353603776.0000
    T+24天预测 - RMSE: 5522.4608, MAE: 5518.4330, MSE: 30497573.8207, MAPE: 55184329753331.82%, 相关系数: nan, R²: -576404145210631424.0000
    T+25天预测 - RMSE: 854.4195, MAE: 842.5224, MSE: 730032.7228, MAPE: 8425223665233.03%, 相关系数: nan, R²: -13724615187710880.0000
    T+26天预测 - RMSE: 2920.2316, MAE: 2917.9881, MSE: 8527752.4225, MAPE: 29179880586985.49%, 相关系数: nan, R²: -159468970300363264.0000
    T+27天预测 - RMSE: 4010.3195, MAE: 4002.3195, MSE: 16082662.4165, MAPE: 40023194628721.98%, 相关系数: nan, R²: -299137520946170112.0000
    T+28天预测 - RMSE: 4707.5240, MAE: 4705.9945, MSE: 22160782.4870, MAPE: 47059944872827.01%, 相关系数: nan, R²: -409974476010402048.0000
    T+29天预测 - RMSE: 3727.7658, MAE: 3726.1805, MSE: 13896237.5978, MAPE: 37261804853320.58%, 相关系数: nan, R²: -255690771799521248.0000
    T+30天预测 - RMSE: 1728.2376, MAE: 1720.8837, MSE: 2986805.0962, MAPE: 17208837429136.93%, 相关系数: nan, R²: -54658533261310432.0000
    
    tuoshan站点:
    T+1天预测 - RMSE: 0.4886, MAE: 0.2220, MSE: 0.2387, MAPE: 547.83%, 相关系数: 0.3680, R²: -0.0568
    T+2天预测 - RMSE: 0.6834, MAE: 0.3417, MSE: 0.4670, MAPE: 364.88%, 相关系数: 0.2124, R²: -0.5353
    T+3天预测 - RMSE: 1.2497, MAE: 0.5142, MSE: 1.5617, MAPE: 1760.68%, 相关系数: -0.0614, R²: -1.0023
    T+4天预测 - RMSE: 1.1233, MAE: 0.5393, MSE: 1.2619, MAPE: 355.16%, 相关系数: -0.1226, R²: -1.2196
    T+5天预测 - RMSE: 1.0306, MAE: 0.5609, MSE: 1.0622, MAPE: 414.08%, 相关系数: -0.1099, R²: -1.2336
    T+6天预测 - RMSE: 1.0837, MAE: 0.6086, MSE: 1.1745, MAPE: 1285.02%, 相关系数: -0.1720, R²: -1.3288
    T+7天预测 - RMSE: 1.0929, MAE: 0.6725, MSE: 1.1945, MAPE: 1260.12%, 相关系数: -0.2106, R²: -1.3447
    T+8天预测 - RMSE: 1.1726, MAE: 0.7276, MSE: 1.3749, MAPE: 959.14%, 相关系数: -0.1907, R²: -1.6282
    T+9天预测 - RMSE: 1.0349, MAE: 0.6430, MSE: 1.0710, MAPE: 941.65%, 相关系数: -0.1895, R²: -1.0346
    T+10天预测 - RMSE: 1.1879, MAE: 0.7215, MSE: 1.4111, MAPE: 651.81%, 相关系数: -0.1778, R²: -1.4073
    T+11天预测 - RMSE: 1.2094, MAE: 0.7263, MSE: 1.4625, MAPE: 689.14%, 相关系数: -0.1677, R²: -1.4210
    T+12天预测 - RMSE: 1.2339, MAE: 0.7472, MSE: 1.5226, MAPE: 964.87%, 相关系数: -0.1040, R²: -1.4938
    T+13天预测 - RMSE: 1.1173, MAE: 0.7027, MSE: 1.2484, MAPE: 873.14%, 相关系数: -0.1093, R²: -1.1494
    T+14天预测 - RMSE: 1.0928, MAE: 0.7219, MSE: 1.1941, MAPE: 569.16%, 相关系数: -0.1381, R²: -1.0226
    T+15天预测 - RMSE: 1.0531, MAE: 0.7149, MSE: 1.1091, MAPE: 427.59%, 相关系数: -0.1849, R²: -1.0897
    T+16天预测 - RMSE: 1.1201, MAE: 0.7633, MSE: 1.2547, MAPE: 460.04%, 相关系数: -0.2122, R²: -1.3062
    T+17天预测 - RMSE: 1.1276, MAE: 0.7812, MSE: 1.2715, MAPE: 343.85%, 相关系数: -0.1409, R²: -1.2315
    T+18天预测 - RMSE: 1.0471, MAE: 0.7310, MSE: 1.0963, MAPE: 871.35%, 相关系数: -0.0822, R²: -1.0727
    T+19天预测 - RMSE: 1.0660, MAE: 0.7322, MSE: 1.1364, MAPE: 520.82%, 相关系数: -0.0343, R²: -0.7921
    T+20天预测 - RMSE: 1.1032, MAE: 0.7452, MSE: 1.2171, MAPE: 1331.16%, 相关系数: 0.0355, R²: -0.7142
    T+21天预测 - RMSE: 1.0852, MAE: 0.7624, MSE: 1.1776, MAPE: 393.35%, 相关系数: 0.0536, R²: -0.9280
    T+22天预测 - RMSE: 1.0511, MAE: 0.7347, MSE: 1.1049, MAPE: 258.71%, 相关系数: -0.0717, R²: -1.0226
    T+23天预测 - RMSE: 1.2619, MAE: 0.8135, MSE: 1.5923, MAPE: 590.88%, 相关系数: -0.0915, R²: -0.9980
    T+24天预测 - RMSE: 1.5168, MAE: 0.9698, MSE: 2.3005, MAPE: 365.86%, 相关系数: -0.0806, R²: -1.2446
    T+25天预测 - RMSE: 1.5782, MAE: 0.9572, MSE: 2.4908, MAPE: 367.65%, 相关系数: -0.0657, R²: -1.0015
    T+26天预测 - RMSE: 1.2438, MAE: 0.8756, MSE: 1.5471, MAPE: 311.83%, 相关系数: -0.0939, R²: -1.2360
    T+27天预测 - RMSE: 1.1353, MAE: 0.8720, MSE: 1.2889, MAPE: 3289.75%, 相关系数: 0.0069, R²: -1.3124
    T+28天预测 - RMSE: 1.0052, MAE: 0.7937, MSE: 1.0104, MAPE: 331.89%, 相关系数: 0.0573, R²: -0.9732
    T+29天预测 - RMSE: 1.0529, MAE: 0.8385, MSE: 1.1087, MAPE: 652.40%, 相关系数: 0.0173, R²: -1.0578
    T+30天预测 - RMSE: 1.0819, MAE: 0.8656, MSE: 1.1705, MAPE: 561.76%, 相关系数: -0.0296, R²: -1.1999
    
    lanshanzui站点:
    T+1天预测 - RMSE: 0.1087, MAE: 0.0725, MSE: 0.0118, MAPE: 1006.75%, 相关系数: 0.0827, R²: -2.2353
    T+2天预测 - RMSE: 0.2970, MAE: 0.1483, MSE: 0.0882, MAPE: 870.27%, 相关系数: 0.0086, R²: -9.6845
    T+3天预测 - RMSE: 0.3019, MAE: 0.2152, MSE: 0.0912, MAPE: 724.99%, 相关系数: -0.0908, R²: -4.0215
    T+4天预测 - RMSE: 0.2622, MAE: 0.1820, MSE: 0.0688, MAPE: 538.51%, 相关系数: -0.1688, R²: -2.1050
    T+5天预测 - RMSE: 0.2772, MAE: 0.2039, MSE: 0.0768, MAPE: 575.90%, 相关系数: -0.2239, R²: -2.1424
    T+6天预测 - RMSE: 0.3305, MAE: 0.2345, MSE: 0.1092, MAPE: 1390.73%, 相关系数: -0.2811, R²: -3.2175
    T+7天预测 - RMSE: 0.2740, MAE: 0.2109, MSE: 0.0751, MAPE: 623970.29%, 相关系数: -0.2161, R²: -1.9957
    T+8天预测 - RMSE: 0.2942, MAE: 0.2248, MSE: 0.0865, MAPE: 597.28%, 相关系数: -0.2249, R²: -2.3957
    T+9天预测 - RMSE: 0.3250, MAE: 0.2540, MSE: 0.1057, MAPE: 644.78%, 相关系数: -0.1834, R²: -3.0772
    T+10天预测 - RMSE: 0.3013, MAE: 0.2377, MSE: 0.0908, MAPE: 585.70%, 相关系数: -0.2119, R²: -2.3917
    T+11天预测 - RMSE: 0.2990, MAE: 0.2315, MSE: 0.0894, MAPE: 705.90%, 相关系数: -0.3588, R²: -2.1726
    T+12天预测 - RMSE: 0.3309, MAE: 0.2545, MSE: 0.1095, MAPE: 880.61%, 相关系数: -0.4404, R²: -2.3356
    T+13天预测 - RMSE: 0.3957, MAE: 0.2956, MSE: 0.1566, MAPE: 764.14%, 相关系数: -0.4707, R²: -3.4983
    T+14天预测 - RMSE: 0.4059, MAE: 0.3045, MSE: 0.1648, MAPE: 881.08%, 相关系数: -0.3493, R²: -3.9918
    T+15天预测 - RMSE: 0.3875, MAE: 0.3029, MSE: 0.1502, MAPE: 458.36%, 相关系数: -0.4710, R²: -3.3863
    T+16天预测 - RMSE: 0.3253, MAE: 0.2531, MSE: 0.1059, MAPE: 332.42%, 相关系数: -0.3935, R²: -2.0686
    T+17天预测 - RMSE: 0.3751, MAE: 0.2887, MSE: 0.1407, MAPE: 479.15%, 相关系数: -0.3436, R²: -3.1037
    T+18天预测 - RMSE: 0.3413, MAE: 0.2737, MSE: 0.1165, MAPE: 563.69%, 相关系数: -0.3987, R²: -2.3999
    T+19天预测 - RMSE: 0.3240, MAE: 0.2617, MSE: 0.1050, MAPE: 597.65%, 相关系数: -0.3259, R²: -2.0430
    T+20天预测 - RMSE: 0.3891, MAE: 0.3118, MSE: 0.1514, MAPE: 441.47%, 相关系数: -0.3043, R²: -3.4133
    T+21天预测 - RMSE: 0.4910, MAE: 0.3872, MSE: 0.2411, MAPE: 555.72%, 相关系数: -0.2874, R²: -5.8287
    T+22天预测 - RMSE: 0.3469, MAE: 0.2876, MSE: 0.1203, MAPE: 343.70%, 相关系数: -0.2179, R²: -2.6269
    T+23天预测 - RMSE: 0.3548, MAE: 0.2878, MSE: 0.1259, MAPE: 1679.67%, 相关系数: -0.1893, R²: -3.0178
    T+24天预测 - RMSE: 0.4381, MAE: 0.3345, MSE: 0.1919, MAPE: 593.96%, 相关系数: -0.2614, R²: -4.4658
    T+25天预测 - RMSE: 0.2972, MAE: 0.2310, MSE: 0.0883, MAPE: 441.79%, 相关系数: -0.0238, R²: -1.9665
    T+26天预测 - RMSE: 0.3244, MAE: 0.2533, MSE: 0.1053, MAPE: 365.19%, 相关系数: -0.1332, R²: -2.5111
    T+27天预测 - RMSE: 0.3285, MAE: 0.2574, MSE: 0.1079, MAPE: 476.56%, 相关系数: -0.1877, R²: -2.4945
    T+28天预测 - RMSE: 0.4531, MAE: 0.3459, MSE: 0.2053, MAPE: 4298.68%, 相关系数: -0.1229, R²: -6.2814
    T+29天预测 - RMSE: 0.3250, MAE: 0.2526, MSE: 0.1057, MAPE: 341.05%, 相关系数: -0.0975, R²: -2.8676
    T+30天预测 - RMSE: 0.3429, MAE: 0.2757, MSE: 0.1176, MAPE: 3462.89%, 相关系数: -0.2896, R²: -3.1234
    
    五里湖心站点:
    T+1天预测 - RMSE: 0.2380, MAE: 0.1651, MSE: 0.0567, MAPE: 1135.68%, 相关系数: -0.0379, R²: -0.6842
    T+2天预测 - RMSE: 0.5142, MAE: 0.3166, MSE: 0.2644, MAPE: 391.65%, 相关系数: -0.1638, R²: -0.7758
    T+3天预测 - RMSE: 0.6081, MAE: 0.4155, MSE: 0.3698, MAPE: 1031.93%, 相关系数: -0.1728, R²: -0.9010
    T+4天预测 - RMSE: 0.6714, MAE: 0.4620, MSE: 0.4507, MAPE: 775.93%, 相关系数: -0.2635, R²: -1.0658
    T+5天预测 - RMSE: 0.7917, MAE: 0.5276, MSE: 0.6268, MAPE: 383.49%, 相关系数: -0.2803, R²: -1.1911
    T+6天预测 - RMSE: 0.8584, MAE: 0.5650, MSE: 0.7368, MAPE: 524.40%, 相关系数: -0.1059, R²: -1.1088
    T+7天预测 - RMSE: 0.8037, MAE: 0.5262, MSE: 0.6459, MAPE: 358.80%, 相关系数: 0.0482, R²: -0.7483
    T+8天预测 - RMSE: 0.9179, MAE: 0.6320, MSE: 0.8426, MAPE: 306.50%, 相关系数: 0.0366, R²: -1.2547
    T+9天预测 - RMSE: 0.8630, MAE: 0.5792, MSE: 0.7448, MAPE: 411.82%, 相关系数: -0.0255, R²: -1.3337
    T+10天预测 - RMSE: 0.7535, MAE: 0.5527, MSE: 0.5678, MAPE: 326.06%, 相关系数: -0.0329, R²: -0.8484
    T+11天预测 - RMSE: 0.8209, MAE: 0.5404, MSE: 0.6738, MAPE: 226.79%, 相关系数: 0.0374, R²: -0.6858
    T+12天预测 - RMSE: 1.0545, MAE: 0.6739, MSE: 1.1119, MAPE: 456.44%, 相关系数: -0.1496, R²: -1.0374
    T+13天预测 - RMSE: 1.1594, MAE: 0.7014, MSE: 1.3441, MAPE: 231.95%, 相关系数: -0.1977, R²: -1.0186
    T+14天预测 - RMSE: 1.2087, MAE: 0.7508, MSE: 1.4609, MAPE: 862.22%, 相关系数: -0.1801, R²: -0.9255
    T+15天预测 - RMSE: 1.3353, MAE: 0.8233, MSE: 1.7831, MAPE: 273.77%, 相关系数: -0.2056, R²: -1.0288
    T+16天预测 - RMSE: 1.3653, MAE: 0.8142, MSE: 1.8639, MAPE: 632.48%, 相关系数: -0.1507, R²: -0.9175
    T+17天预测 - RMSE: 1.2973, MAE: 0.8134, MSE: 1.6829, MAPE: 500.59%, 相关系数: -0.0092, R²: -0.8885
    T+18天预测 - RMSE: 1.2583, MAE: 0.8376, MSE: 1.5832, MAPE: 1026.73%, 相关系数: -0.0116, R²: -0.7278
    T+19天预测 - RMSE: 1.2252, MAE: 0.6798, MSE: 1.5012, MAPE: 544.01%, 相关系数: 0.0706, R²: -0.6340
    T+20天预测 - RMSE: 1.1499, MAE: 0.6968, MSE: 1.3223, MAPE: 526.85%, 相关系数: 0.1419, R²: -0.5348
    T+21天预测 - RMSE: 0.9975, MAE: 0.6600, MSE: 0.9949, MAPE: 766.63%, 相关系数: 0.2443, R²: -0.3588
    T+22天预测 - RMSE: 1.1380, MAE: 0.7323, MSE: 1.2950, MAPE: 590.89%, 相关系数: -0.0757, R²: -1.0775
    T+23天预测 - RMSE: 1.2938, MAE: 0.7994, MSE: 1.6740, MAPE: 1084.62%, 相关系数: -0.0652, R²: -0.9327
    T+24天预测 - RMSE: 1.2690, MAE: 0.7671, MSE: 1.6104, MAPE: 1016.53%, 相关系数: 0.1215, R²: -0.5663
    T+25天预测 - RMSE: 1.1907, MAE: 0.8182, MSE: 1.4178, MAPE: 325.58%, 相关系数: 0.1160, R²: -0.4373
    T+26天预测 - RMSE: 1.0012, MAE: 0.7201, MSE: 1.0025, MAPE: 434.11%, 相关系数: 0.3432, R²: -0.1637
    T+27天预测 - RMSE: 0.9735, MAE: 0.7349, MSE: 0.9477, MAPE: 563.97%, 相关系数: 0.3225, R²: -0.3316
    T+28天预测 - RMSE: 0.9017, MAE: 0.7052, MSE: 0.8130, MAPE: 336.74%, 相关系数: 0.3252, R²: -0.4468
    T+29天预测 - RMSE: 0.8877, MAE: 0.7108, MSE: 0.7880, MAPE: 319.52%, 相关系数: 0.4790, R²: 0.0701
    T+30天预测 - RMSE: 1.1414, MAE: 0.8300, MSE: 1.3027, MAPE: 364.19%, 相关系数: 0.2194, R²: -0.7707
    

    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    


```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import logging
from torch.optim.lr_scheduler import OneCycleLR

# 设置字体
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimSun', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
chinese_font = fm.FontProperties(fname=r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\SIMSUN.TTC', size=14)
english_font = fm.FontProperties(fname=r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\TIMES.TTF', size=14)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 按照新分类重新排序的地点
location_order = ['胥湖心', '锡东水厂', '平台山', 'tuoshan', 'lanshanzui', '五里湖心']

# 重新分类的生态区域和对应的颜色 - 配色不要变
zone_info = {
    '胥湖心': {'zone_cn': '重污染与高风险区', 'zone_en': 'Heavily Polluted & High-Risk Area', 'color_primary': '#1B365D', 'color_secondary': '#B0C4DE'},  # 深海军蓝与浅蓝灰
    '锡东水厂': {'zone_cn': '重污染与高风险区', 'zone_en': 'Heavily Polluted & High-Risk Area', 'color_primary': '#1B365D', 'color_secondary': '#B0C4DE'},
    '平台山': {'zone_cn': '背景与参照区', 'zone_en': 'Background & Reference Area', 'color_primary': '#2D5016', 'color_secondary': '#9ACD32'},  # 深森林绿与浅橄榄绿
    'tuoshan': {'zone_cn': '背景与参照区', 'zone_en': 'Background & Reference Area', 'color_primary': '#2D5016', 'color_secondary': '#9ACD32'},
    'lanshanzui': {'zone_cn': '边界条件区', 'zone_en': 'Boundary Condition Area', 'color_primary': '#8B4513', 'color_secondary': '#F0E68C'},  # 深赭石色与浅卡其色
    '五里湖心': {'zone_cn': '边界条件区', 'zone_en': 'Boundary Condition Area', 'color_primary': '#8B4513', 'color_secondary': '#F0E68C'}
}

# 站点文件映射
station_files_map = {
    '胥湖心': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-胥湖心-merged_with_weather_with_composite_features_processed.csv',
    '锡东水厂': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-锡东水厂-merged_with_weather_with_composite_features_processed.csv',
    '平台山': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-平台山-merged_with_weather_with_composite_features_processed.csv',
    'tuoshan': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-tuoshan-merged_with_weather_with_composite_features_processed.csv',
    'lanshanzui': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-lanshanzui-merged_with_weather_with_composite_features_processed.csv',
    '五里湖心': r'C:\Users\hdec\OneDrive\★蓝藻\AlgaeBloomForecast-20241122\data\002-气象-五里湖心-merged_with_weather_with_composite_features_processed.csv'
}

# 英文站点名转中文字典
station_name_map = {
    'lanshanzui': '兰山嘴',
    'tuoshan': '拖山'
}

# 英文站点名称映射
station_name_en_map = {
    '胥湖心': 'Xuhu Center', 
    '锡东水厂': 'Xidong Water Plant', 
    '平台山': 'Pingtai Mountain',
    'tuoshan': 'Tuoshan Mountain', 
    'lanshanzui': 'Lanshan Cape', 
    '五里湖心': 'Wulihu Center'
}

# 定义中英文变量名称
variables = {
    'density': {'cn': '藻密度', 'en': 'Algae Density'}, 
    'chla': {'cn': '叶绿素a', 'en': 'Chlorophyll a'}
}

# 使用基础指标来进行预测和建模计算，建简化的LaTeX表示如下
basic_indicator_formulas = {
    'temperature': r'$T$',
    'oxygen': r'$O_2$',
    'TN': r'$N_t$',
    'TP': r'$P_t$',
    'NH': r'$N_{NH}$',
    'pH': r'$pH$',
    'turbidity': r'$\tau$',
    'conductivity': r'$\sigma$',
    'permanganate': r'$COD_{Mn}$',
    'rain_sum': r'$R$',
    'wind_speed_10m_max': r'$u$',
    'shortwave_radiation_sum': r'$I_s$'
}

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

def create_sequence_for_date(data, date_index, seq_length):
    """为指定日期创建输入序列"""
    if date_index < seq_length:
        return None
    return data[date_index - seq_length:date_index]

# 基础特征 - 使用基础指标
base_features = ['temperature', 'pH', 'oxygen', 'permanganate', 'TN', 'conductivity', 
                'turbidity', 'density', 'rain_sum', 'wind_speed_10m_max', 
                'shortwave_radiation_sum']


for station_name in location_order:
    print(f"\n正在绘制站点: {station_name}")
    
    # 从保存的文件中加载该站点的模型和预测数据
    filename = f'00-lstm_model_data_{station_name}-去除负数.pkl'
    try:
        with open(filename, 'rb') as f:
            station_model_data = pickle.load(f)
        
        station_predictions = station_model_data['predictions_all']  # 从文件中获取所有预测结果
        station_actuals = station_model_data['actual_values_all']    # 从文件中获取所有实际值
        current_zone_info = station_model_data['zone_info']          # 从文件中获取站点区域信息
        
        # 获取站点颜色
        primary_color = current_zone_info['color_primary']
        secondary_color = current_zone_info['color_secondary']

    except FileNotFoundError:
        print(f"错误: 未找到站点 {station_name} 的模型数据文件 {filename}，跳过此站点。")
        continue # 跳过当前站点，继续下一个
    except KeyError as e:
        print(f"错误: 文件 {filename} 中缺少必要的键 '{e}'，跳过此站点。")
        continue # 跳过当前站点，继续下一个
    
    # 计算每个预测天数的RMSE
    print(f"\n{station_name}站点藻密度增长率预测的RMSE:")
    for days in range(1, 31):
        if days in station_predictions and days in station_actuals:
            rmse = np.sqrt(mean_squared_error(station_actuals[days], station_predictions[days]))
            print(f"T+{days}天预测的RMSE: {rmse:.4f}")
    
    # 每6天绘制一张图，共5张图
    for group in range(5):  # 0-4共5组
        plt.figure(figsize=(15, 10))
        
        for i in range(6):  # 每组6天
            days = group * 6 + i + 1
            if days <= 30:  # 确保不超过30天
                plt.subplot(3, 2, i+1)
                
                # 获取该预测天数的数据
                if days in station_predictions and days in station_actuals:
                    pred_values = station_predictions[days]
                    actual_values = station_actuals[days]
                    
                    # 创建日期索引（基于测试集长度）
                    test_start_date = '2023-11-01'
                    test_end_date = '2024-05-31'
                    prediction_dates = pd.date_range(start=test_start_date, end=test_end_date)
                    
                    # 确保数据长度匹配
                    min_length = min(len(pred_values), len(actual_values), len(prediction_dates))
                    pred_values = pred_values[:min_length]
                    actual_values = actual_values[:min_length]
                    prediction_dates = prediction_dates[:min_length]
                    
                    # 只绘制2024年1-5月的数据
                    mask = (prediction_dates >= '2024-01-01') & (prediction_dates <= '2024-05-31')
                    
                    # 检查掩码和数据长度是否匹配
                    if len(mask) == len(pred_values) and len(mask) == len(actual_values):
                        # 应用掩码筛选数据
                        filtered_dates = prediction_dates[mask]
                        filtered_pred = pred_values[mask]
                        filtered_actual = actual_values[mask]
                        
                        plt.plot(filtered_dates, filtered_pred, 
                                label='预测值', alpha=0.7, color=primary_color)
                        plt.plot(filtered_dates, filtered_actual, 
                                label='实际值', alpha=0.7, color=secondary_color)
                        
                        plt.title(f'{station_name} - T+{days}天增长率预测', 
                                 fontproperties=chinese_font, fontsize=12)
                        plt.xlabel('日期', fontproperties=chinese_font, fontsize=10)
                        plt.ylabel('增长率', fontproperties=chinese_font, fontsize=10)
                        plt.legend(prop=chinese_font)
                        plt.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        
                        # 设置刻度字体为英文字体
                        ax = plt.gca()
                        for label in ax.get_xticklabels() + ax.get_yticklabels():
                            label.set_fontproperties(english_font)
                    else:
                        plt.text(0.5, 0.5, f'数据长度不匹配\n掩码长度: {len(mask)}\n预测值长度: {len(pred_values)}', 
                               ha='center', va='center', transform=plt.gca().transAxes,
                               fontproperties=chinese_font)
        
        plt.suptitle(f'{station_name}站点 - 第{group+1}组: T+{group*6+1}天至T+{min((group+1)*6, 30)}天预测结果', 
                     fontproperties=chinese_font, fontsize=14)
        plt.tight_layout()
        plt.show()

# 计算每个站点每个预测天数的各种误差指标
print("各站点增长率预测的误差指标:")

for station_name in location_order:
    print(f"\n{station_name}站点:")
    station_predictions = all_station_predictions[station_name]['predictions']
    station_actuals = all_station_predictions[station_name]['actuals']
    
    # 计算每个预测天数的误差指标
    for days in range(1, 31):
        if days in station_predictions and days in station_actuals:
            pred = station_predictions[days]
            actual = station_actuals[days]
            
            # 计算各种误差指标
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            mse = mean_squared_error(actual, pred)
            
            # 计算相对误差指标
            mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100  # 避免除零
            
            # 计算相关系数
            correlation = np.corrcoef(actual, pred)[0, 1]
            
            # 计算决定系数 R²
            ss_res = np.sum((actual - pred) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            print(f"T+{days}天预测 - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.2f}%, 相关系数: {correlation:.4f}, R²: {r2:.4f}")


```

    
    正在绘制站点: 胥湖心
    
    胥湖心站点藻密度增长率预测的RMSE:
    T+1天预测的RMSE: 0.4108
    T+2天预测的RMSE: 0.8438
    T+3天预测的RMSE: 0.9789
    T+4天预测的RMSE: 0.9583
    T+5天预测的RMSE: 1.2963
    T+6天预测的RMSE: 1.5798
    T+7天预测的RMSE: 1.7728
    T+8天预测的RMSE: 1.9075
    T+9天预测的RMSE: 2.0480
    T+10天预测的RMSE: 2.0637
    T+11天预测的RMSE: 2.2059
    T+12天预测的RMSE: 2.1643
    T+13天预测的RMSE: 2.1728
    T+14天预测的RMSE: 2.0373
    T+15天预测的RMSE: 1.9214
    T+16天预测的RMSE: 1.8065
    T+17天预测的RMSE: 1.7288
    T+18天预测的RMSE: 1.6393
    T+19天预测的RMSE: 1.5107
    T+20天预测的RMSE: 1.5528
    T+21天预测的RMSE: 1.8000
    T+22天预测的RMSE: 1.9052
    T+23天预测的RMSE: 2.0127
    T+24天预测的RMSE: 1.9548
    T+25天预测的RMSE: 2.2728
    T+26天预测的RMSE: 2.1927
    T+27天预测的RMSE: 2.5165
    T+28天预测的RMSE: 2.6571
    T+29天预测的RMSE: 3.1019
    T+30天预测的RMSE: 2.9441
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_1.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_2.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_3.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_4.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_5.png)
    


    
    正在绘制站点: 锡东水厂
    
    锡东水厂站点藻密度增长率预测的RMSE:
    T+1天预测的RMSE: 0.4604
    T+2天预测的RMSE: 0.9987
    T+3天预测的RMSE: 1.0404
    T+4天预测的RMSE: 1.0112
    T+5天预测的RMSE: 1.2053
    T+6天预测的RMSE: 1.2302
    T+7天预测的RMSE: 1.5013
    T+8天预测的RMSE: 1.4735
    T+9天预测的RMSE: 1.9027
    T+10天预测的RMSE: 2.0937
    T+11天预测的RMSE: 2.5183
    T+12天预测的RMSE: 1.8628
    T+13天预测的RMSE: 1.5273
    T+14天预测的RMSE: 1.6661
    T+15天预测的RMSE: 1.4802
    T+16天预测的RMSE: 1.5089
    T+17天预测的RMSE: 1.7270
    T+18天预测的RMSE: 1.8073
    T+19天预测的RMSE: 1.9090
    T+20天预测的RMSE: 2.0768
    T+21天预测的RMSE: 1.6871
    T+22天预测的RMSE: 1.6866
    T+23天预测的RMSE: 1.6559
    T+24天预测的RMSE: 1.6325
    T+25天预测的RMSE: 1.9190
    T+26天预测的RMSE: 1.9158
    T+27天预测的RMSE: 2.0488
    T+28天预测的RMSE: 1.8730
    T+29天预测的RMSE: 2.4712
    T+30天预测的RMSE: 2.6040
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_7.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_8.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_9.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_10.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_11.png)
    


    
    正在绘制站点: 平台山
    
    平台山站点藻密度增长率预测的RMSE:
    T+1天预测的RMSE: 1269.4114
    T+2天预测的RMSE: 2497.8370
    T+3天预测的RMSE: 6641.9705
    T+4天预测的RMSE: 3828.8475
    T+5天预测的RMSE: 4542.9031
    T+6天预测的RMSE: 4496.4427
    T+7天预测的RMSE: 5735.9004
    T+8天预测的RMSE: 3621.1481
    T+9天预测的RMSE: 1764.3642
    T+10天预测的RMSE: 1865.6952
    T+11天预测的RMSE: 306.8187
    T+12天预测的RMSE: 5389.5882
    T+13天预测的RMSE: 2801.3056
    T+14天预测的RMSE: 1738.4652
    T+15天预测的RMSE: 1192.0137
    T+16天预测的RMSE: 902.9303
    T+17天预测的RMSE: 6123.8231
    T+18天预测的RMSE: 5489.8347
    T+19天预测的RMSE: 2510.0105
    T+20天预测的RMSE: 3593.2752
    T+21天预测的RMSE: 2761.0179
    T+22天预测的RMSE: 2343.2284
    T+23天预测的RMSE: 3102.8223
    T+24天预测的RMSE: 5522.4608
    T+25天预测的RMSE: 854.4195
    T+26天预测的RMSE: 2920.2316
    T+27天预测的RMSE: 4010.3195
    T+28天预测的RMSE: 4707.5240
    T+29天预测的RMSE: 3727.7658
    T+30天预测的RMSE: 1728.2376
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_13.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_14.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_15.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_16.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_17.png)
    


    
    正在绘制站点: tuoshan
    
    tuoshan站点藻密度增长率预测的RMSE:
    T+1天预测的RMSE: 0.4886
    T+2天预测的RMSE: 0.6834
    T+3天预测的RMSE: 1.2497
    T+4天预测的RMSE: 1.1233
    T+5天预测的RMSE: 1.0306
    T+6天预测的RMSE: 1.0837
    T+7天预测的RMSE: 1.0929
    T+8天预测的RMSE: 1.1726
    T+9天预测的RMSE: 1.0349
    T+10天预测的RMSE: 1.1879
    T+11天预测的RMSE: 1.2094
    T+12天预测的RMSE: 1.2339
    T+13天预测的RMSE: 1.1173
    T+14天预测的RMSE: 1.0928
    T+15天预测的RMSE: 1.0531
    T+16天预测的RMSE: 1.1201
    T+17天预测的RMSE: 1.1276
    T+18天预测的RMSE: 1.0471
    T+19天预测的RMSE: 1.0660
    T+20天预测的RMSE: 1.1032
    T+21天预测的RMSE: 1.0852
    T+22天预测的RMSE: 1.0511
    T+23天预测的RMSE: 1.2619
    T+24天预测的RMSE: 1.5168
    T+25天预测的RMSE: 1.5782
    T+26天预测的RMSE: 1.2438
    T+27天预测的RMSE: 1.1353
    T+28天预测的RMSE: 1.0052
    T+29天预测的RMSE: 1.0529
    T+30天预测的RMSE: 1.0819
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_19.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_20.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_21.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_22.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_23.png)
    


    
    正在绘制站点: lanshanzui
    
    lanshanzui站点藻密度增长率预测的RMSE:
    T+1天预测的RMSE: 0.1087
    T+2天预测的RMSE: 0.2970
    T+3天预测的RMSE: 0.3019
    T+4天预测的RMSE: 0.2622
    T+5天预测的RMSE: 0.2772
    T+6天预测的RMSE: 0.3305
    T+7天预测的RMSE: 0.2740
    T+8天预测的RMSE: 0.2942
    T+9天预测的RMSE: 0.3250
    T+10天预测的RMSE: 0.3013
    T+11天预测的RMSE: 0.2990
    T+12天预测的RMSE: 0.3309
    T+13天预测的RMSE: 0.3957
    T+14天预测的RMSE: 0.4059
    T+15天预测的RMSE: 0.3875
    T+16天预测的RMSE: 0.3253
    T+17天预测的RMSE: 0.3751
    T+18天预测的RMSE: 0.3413
    T+19天预测的RMSE: 0.3240
    T+20天预测的RMSE: 0.3891
    T+21天预测的RMSE: 0.4910
    T+22天预测的RMSE: 0.3469
    T+23天预测的RMSE: 0.3548
    T+24天预测的RMSE: 0.4381
    T+25天预测的RMSE: 0.2972
    T+26天预测的RMSE: 0.3244
    T+27天预测的RMSE: 0.3285
    T+28天预测的RMSE: 0.4531
    T+29天预测的RMSE: 0.3250
    T+30天预测的RMSE: 0.3429
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_25.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_26.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_27.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_28.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_29.png)
    


    
    正在绘制站点: 五里湖心
    
    五里湖心站点藻密度增长率预测的RMSE:
    T+1天预测的RMSE: 0.2380
    T+2天预测的RMSE: 0.5142
    T+3天预测的RMSE: 0.6081
    T+4天预测的RMSE: 0.6714
    T+5天预测的RMSE: 0.7917
    T+6天预测的RMSE: 0.8584
    T+7天预测的RMSE: 0.8037
    T+8天预测的RMSE: 0.9179
    T+9天预测的RMSE: 0.8630
    T+10天预测的RMSE: 0.7535
    T+11天预测的RMSE: 0.8209
    T+12天预测的RMSE: 1.0545
    T+13天预测的RMSE: 1.1594
    T+14天预测的RMSE: 1.2087
    T+15天预测的RMSE: 1.3353
    T+16天预测的RMSE: 1.3653
    T+17天预测的RMSE: 1.2973
    T+18天预测的RMSE: 1.2583
    T+19天预测的RMSE: 1.2252
    T+20天预测的RMSE: 1.1499
    T+21天预测的RMSE: 0.9975
    T+22天预测的RMSE: 1.1380
    T+23天预测的RMSE: 1.2938
    T+24天预测的RMSE: 1.2690
    T+25天预测的RMSE: 1.1907
    T+26天预测的RMSE: 1.0012
    T+27天预测的RMSE: 0.9735
    T+28天预测的RMSE: 0.9017
    T+29天预测的RMSE: 0.8877
    T+30天预测的RMSE: 1.1414
    


    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_31.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_32.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_33.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_34.png)
    



    
![png](%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_files/%E2%98%85201-LSTM-%E5%9F%BA%E7%A1%80%E6%8C%87%E6%A0%87%E4%B8%8D%E5%90%ABdensity-%E5%8E%BB%E9%99%A4%E8%B4%9F%E6%95%B0_7_35.png)
    


    各站点增长率预测的误差指标:
    
    胥湖心站点:
    T+1天预测 - RMSE: 0.4108, MAE: 0.2343, MSE: 0.1687, MAPE: 81474857.35%, 相关系数: 0.2062, R²: -0.2732
    T+2天预测 - RMSE: 0.8438, MAE: 0.4748, MSE: 0.7120, MAPE: 294886085.31%, 相关系数: 0.0576, R²: -0.6443
    T+3天预测 - RMSE: 0.9789, MAE: 0.6260, MSE: 0.9583, MAPE: 281325756.73%, 相关系数: 0.0208, R²: -0.7805
    T+4天预测 - RMSE: 0.9583, MAE: 0.6507, MSE: 0.9183, MAPE: 212176446.87%, 相关系数: 0.0100, R²: -0.7435
    T+5天预测 - RMSE: 1.2963, MAE: 0.8678, MSE: 1.6803, MAPE: 600491108.91%, 相关系数: -0.0685, R²: -1.2555
    T+6天预测 - RMSE: 1.5798, MAE: 1.0517, MSE: 2.4956, MAPE: 559683179.39%, 相关系数: -0.1440, R²: -1.4045
    T+7天预测 - RMSE: 1.7728, MAE: 1.1565, MSE: 3.1427, MAPE: 180128208.30%, 相关系数: -0.2105, R²: -1.3373
    T+8天预测 - RMSE: 1.9075, MAE: 1.2668, MSE: 3.6386, MAPE: 101622621.11%, 相关系数: -0.2681, R²: -1.3486
    T+9天预测 - RMSE: 2.0480, MAE: 1.3913, MSE: 4.1943, MAPE: 50296034.32%, 相关系数: -0.2948, R²: -1.5030
    T+10天预测 - RMSE: 2.0637, MAE: 1.4125, MSE: 4.2589, MAPE: 39312251.47%, 相关系数: -0.3152, R²: -1.5481
    T+11天预测 - RMSE: 2.2059, MAE: 1.4853, MSE: 4.8661, MAPE: 9935066.95%, 相关系数: -0.3282, R²: -1.6104
    T+12天预测 - RMSE: 2.1643, MAE: 1.4749, MSE: 4.6844, MAPE: 476.75%, 相关系数: -0.2648, R²: -1.5115
    T+13天预测 - RMSE: 2.1728, MAE: 1.4592, MSE: 4.7211, MAPE: 353.06%, 相关系数: -0.2160, R²: -1.3940
    T+14天预测 - RMSE: 2.0373, MAE: 1.3306, MSE: 4.1506, MAPE: 399.02%, 相关系数: -0.1666, R²: -1.2249
    T+15天预测 - RMSE: 1.9214, MAE: 1.2682, MSE: 3.6918, MAPE: 676.90%, 相关系数: -0.1334, R²: -1.2139
    T+16天预测 - RMSE: 1.8065, MAE: 1.2458, MSE: 3.2633, MAPE: 1137.77%, 相关系数: -0.0655, R²: -0.9884
    T+17天预测 - RMSE: 1.7288, MAE: 1.1932, MSE: 2.9887, MAPE: 478.05%, 相关系数: -0.0173, R²: -1.1444
    T+18天预测 - RMSE: 1.6393, MAE: 1.0896, MSE: 2.6873, MAPE: 308.21%, 相关系数: 0.0305, R²: -0.9773
    T+19天预测 - RMSE: 1.5107, MAE: 0.9855, MSE: 2.2823, MAPE: 1117.95%, 相关系数: 0.0224, R²: -0.8177
    T+20天预测 - RMSE: 1.5528, MAE: 1.1030, MSE: 2.4113, MAPE: 353.52%, 相关系数: -0.0340, R²: -1.1196
    T+21天预测 - RMSE: 1.8000, MAE: 1.2605, MSE: 3.2399, MAPE: 296.82%, 相关系数: -0.0437, R²: -1.0801
    T+22天预测 - RMSE: 1.9052, MAE: 1.3239, MSE: 3.6298, MAPE: 1121.64%, 相关系数: -0.0911, R²: -1.0771
    T+23天预测 - RMSE: 2.0127, MAE: 1.2972, MSE: 4.0509, MAPE: 474.79%, 相关系数: -0.1161, R²: -1.3118
    T+24天预测 - RMSE: 1.9548, MAE: 1.2749, MSE: 3.8212, MAPE: 1281.64%, 相关系数: -0.1609, R²: -2.2700
    T+25天预测 - RMSE: 2.2728, MAE: 1.4247, MSE: 5.1658, MAPE: 533.69%, 相关系数: -0.1681, R²: -2.5050
    T+26天预测 - RMSE: 2.1927, MAE: 1.3902, MSE: 4.8079, MAPE: 704.87%, 相关系数: -0.2180, R²: -2.7614
    T+27天预测 - RMSE: 2.5165, MAE: 1.4821, MSE: 6.3328, MAPE: 771.44%, 相关系数: -0.1886, R²: -2.6235
    T+28天预测 - RMSE: 2.6571, MAE: 1.5086, MSE: 7.0600, MAPE: 994.42%, 相关系数: -0.1888, R²: -2.5711
    T+29天预测 - RMSE: 3.1019, MAE: 1.6731, MSE: 9.6216, MAPE: 652.18%, 相关系数: -0.2037, R²: -3.5732
    T+30天预测 - RMSE: 2.9441, MAE: 1.6173, MSE: 8.6679, MAPE: 1313.92%, 相关系数: -0.2121, R²: -3.0135
    
    锡东水厂站点:
    T+1天预测 - RMSE: 0.4604, MAE: 0.2985, MSE: 0.2119, MAPE: 143997436.64%, 相关系数: 0.2558, R²: -0.0867
    T+2天预测 - RMSE: 0.9987, MAE: 0.5717, MSE: 0.9975, MAPE: 208020465.89%, 相关系数: -0.0496, R²: -0.7773
    T+3天预测 - RMSE: 1.0404, MAE: 0.6630, MSE: 1.0823, MAPE: 106684331.78%, 相关系数: -0.0708, R²: -0.9916
    T+4天预测 - RMSE: 1.0112, MAE: 0.7209, MSE: 1.0226, MAPE: 119438428.78%, 相关系数: -0.0384, R²: -1.0451
    T+5天预测 - RMSE: 1.2053, MAE: 0.8190, MSE: 1.4528, MAPE: 98193382.46%, 相关系数: 0.0586, R²: -1.1265
    T+6天预测 - RMSE: 1.2302, MAE: 0.8302, MSE: 1.5133, MAPE: 128273598.59%, 相关系数: -0.0863, R²: -0.8355
    T+7天预测 - RMSE: 1.5013, MAE: 0.9585, MSE: 2.2538, MAPE: 114983780.98%, 相关系数: -0.1427, R²: -1.2669
    T+8天预测 - RMSE: 1.4735, MAE: 0.9441, MSE: 2.1713, MAPE: 91318743.31%, 相关系数: -0.1636, R²: -1.0646
    T+9天预测 - RMSE: 1.9027, MAE: 1.2210, MSE: 3.6202, MAPE: 103491130.69%, 相关系数: -0.1667, R²: -1.2359
    T+10天预测 - RMSE: 2.0937, MAE: 1.2921, MSE: 4.3835, MAPE: 665.94%, 相关系数: -0.2265, R²: -1.0906
    T+11天预测 - RMSE: 2.5183, MAE: 1.5105, MSE: 6.3417, MAPE: 573.38%, 相关系数: -0.1711, R²: -1.1878
    T+12天预测 - RMSE: 1.8628, MAE: 1.2137, MSE: 3.4699, MAPE: 879.82%, 相关系数: -0.2023, R²: -1.1392
    T+13天预测 - RMSE: 1.5273, MAE: 1.1061, MSE: 2.3328, MAPE: 340.73%, 相关系数: -0.1875, R²: -1.2600
    T+14天预测 - RMSE: 1.6661, MAE: 1.2892, MSE: 2.7758, MAPE: 728.24%, 相关系数: -0.3251, R²: -1.8982
    T+15天预测 - RMSE: 1.4802, MAE: 1.1758, MSE: 2.1911, MAPE: 573.88%, 相关系数: -0.2133, R²: -1.4771
    T+16天预测 - RMSE: 1.5089, MAE: 1.1915, MSE: 2.2768, MAPE: 580.99%, 相关系数: -0.2163, R²: -1.0116
    T+17天预测 - RMSE: 1.7270, MAE: 1.2861, MSE: 2.9824, MAPE: 2110.47%, 相关系数: -0.1694, R²: -1.3036
    T+18天预测 - RMSE: 1.8073, MAE: 1.2806, MSE: 3.2664, MAPE: 343.20%, 相关系数: -0.1086, R²: -1.0424
    T+19天预测 - RMSE: 1.9090, MAE: 1.3709, MSE: 3.6445, MAPE: 2027.14%, 相关系数: -0.1043, R²: -1.4567
    T+20天预测 - RMSE: 2.0768, MAE: 1.4359, MSE: 4.3131, MAPE: 341.91%, 相关系数: -0.0074, R²: -1.8758
    T+21天预测 - RMSE: 1.6871, MAE: 1.2795, MSE: 2.8463, MAPE: 388.84%, 相关系数: -0.0004, R²: -1.0337
    T+22天预测 - RMSE: 1.6866, MAE: 1.2670, MSE: 2.8447, MAPE: 370.69%, 相关系数: 0.0130, R²: -1.0776
    T+23天预测 - RMSE: 1.6559, MAE: 1.3017, MSE: 2.7419, MAPE: 383.38%, 相关系数: 0.1442, R²: -0.5700
    T+24天预测 - RMSE: 1.6325, MAE: 1.2491, MSE: 2.6649, MAPE: 357.25%, 相关系数: 0.0927, R²: -0.6716
    T+25天预测 - RMSE: 1.9190, MAE: 1.3892, MSE: 3.6826, MAPE: 698.52%, 相关系数: -0.0696, R²: -1.0148
    T+26天预测 - RMSE: 1.9158, MAE: 1.3902, MSE: 3.6704, MAPE: 751.57%, 相关系数: -0.0382, R²: -1.0290
    T+27天预测 - RMSE: 2.0488, MAE: 1.5112, MSE: 4.1976, MAPE: 679.84%, 相关系数: -0.1364, R²: -1.1981
    T+28天预测 - RMSE: 1.8730, MAE: 1.4078, MSE: 3.5079, MAPE: 1100.17%, 相关系数: -0.0338, R²: -0.7273
    T+29天预测 - RMSE: 2.4712, MAE: 1.7554, MSE: 6.1066, MAPE: 1375.05%, 相关系数: -0.2162, R²: -1.4010
    T+30天预测 - RMSE: 2.6040, MAE: 1.8047, MSE: 6.7807, MAPE: 772.17%, 相关系数: -0.1465, R²: -1.1584
    
    平台山站点:
    T+1天预测 - RMSE: 1269.4114, MAE: 1268.6285, MSE: 1611405.2299, MAPE: 12686285299323.51%, 相关系数: nan, R²: -34161790874744064.0000
    T+2天预测 - RMSE: 2497.8370, MAE: 2492.7535, MSE: 6239189.5859, MAPE: 24927535425902.57%, 相关系数: nan, R²: -131646900262109152.0000
    T+3天预测 - RMSE: 6641.9705, MAE: 6639.3141, MSE: 44115772.1482, MAPE: 66393141290609.88%, 相关系数: nan, R²: -926431215112957696.0000
    T+4天预测 - RMSE: 3828.8475, MAE: 3827.1498, MSE: 14660073.1349, MAPE: 38271497768561.27%, 相关系数: nan, R²: -306395528518884480.0000
    T+5天预测 - RMSE: 4542.9031, MAE: 4540.9578, MSE: 20637968.7056, MAPE: 45409578317671.94%, 相关系数: nan, R²: -429269749075523200.0000
    T+6天预测 - RMSE: 4496.4427, MAE: 4494.7157, MSE: 20217996.9087, MAPE: 44947156846646.66%, 相关系数: nan, R²: -418512536009291072.0000
    T+7天预测 - RMSE: 5735.9004, MAE: 5734.4093, MSE: 32900553.6865, MAPE: 57344092607632.40%, 相关系数: nan, R²: -677751405942908800.0000
    T+8天预测 - RMSE: 3621.1481, MAE: 3618.6697, MSE: 13112713.8267, MAPE: 36186696510574.18%, 相关系数: nan, R²: -268810633447573760.0000
    T+9天预测 - RMSE: 1764.3642, MAE: 1752.3098, MSE: 3112980.9099, MAPE: 17523098357926.37%, 相关系数: nan, R²: -63504810562466216.0000
    T+10天预测 - RMSE: 1865.6952, MAE: 1863.3920, MSE: 3480818.7153, MAPE: 18633919826376.64%, 相关系数: nan, R²: -70660619920252944.0000
    T+11天预测 - RMSE: 306.8187, MAE: 270.2065, MSE: 94137.7232, MAPE: 2702064501800.19%, 相关系数: nan, R²: -1901582008732098.7500
    T+12天预测 - RMSE: 5389.5882, MAE: 5389.0763, MSE: 29047660.6809, MAPE: 53890762513149.74%, 相关系数: nan, R²: -583857979686576256.0000
    T+13天预测 - RMSE: 2801.3056, MAE: 2800.0296, MSE: 7847313.0194, MAPE: 28000296263118.71%, 相关系数: nan, R²: -156946260387300832.0000
    T+14天预测 - RMSE: 1738.4652, MAE: 1734.2631, MSE: 3022261.0837, MAPE: 17342631344834.25%, 相关系数: nan, R²: -60142995566069176.0000
    T+15天预测 - RMSE: 1192.0137, MAE: 1179.9513, MSE: 1420896.7139, MAPE: 11799513015804.45%, 相关系数: nan, R²: -28133754934655248.0000
    T+16天预测 - RMSE: 902.9303, MAE: 886.3998, MSE: 815283.1971, MAPE: 8863998298573.66%, 相关系数: nan, R²: -16061078982760808.0000
    T+17天预测 - RMSE: 6123.8231, MAE: 6122.0712, MSE: 37501209.6560, MAPE: 61220712347754.17%, 相关系数: nan, R²: -735023709256828160.0000
    T+18天预测 - RMSE: 5489.8347, MAE: 5488.1633, MSE: 30138284.7937, MAPE: 54881633306445.91%, 相关系数: nan, R²: -587696553476391936.0000
    T+19天预测 - RMSE: 2510.0105, MAE: 2504.4010, MSE: 6300152.4846, MAPE: 25044010345399.48%, 相关系数: nan, R²: -122222958201930800.0000
    T+20天预测 - RMSE: 3593.2752, MAE: 3588.4103, MSE: 12911626.8582, MAPE: 35884103486132.60%, 相关系数: nan, R²: -249194398362813312.0000
    T+21天预测 - RMSE: 2761.0179, MAE: 2758.0520, MSE: 7623219.9680, MAPE: 27580519544274.87%, 相关系数: nan, R²: -146365823384755520.0000
    T+22天预测 - RMSE: 2343.2284, MAE: 2341.5270, MSE: 5490719.2244, MAPE: 23415270293370.91%, 相关系数: nan, R²: -104872737185640432.0000
    T+23天预测 - RMSE: 3102.8223, MAE: 3097.2664, MSE: 9627506.1765, MAPE: 30972664385779.92%, 相关系数: nan, R²: -182922617353603776.0000
    T+24天预测 - RMSE: 5522.4608, MAE: 5518.4330, MSE: 30497573.8207, MAPE: 55184329753331.82%, 相关系数: nan, R²: -576404145210631424.0000
    T+25天预测 - RMSE: 854.4195, MAE: 842.5224, MSE: 730032.7228, MAPE: 8425223665233.03%, 相关系数: nan, R²: -13724615187710880.0000
    T+26天预测 - RMSE: 2920.2316, MAE: 2917.9881, MSE: 8527752.4225, MAPE: 29179880586985.49%, 相关系数: nan, R²: -159468970300363264.0000
    T+27天预测 - RMSE: 4010.3195, MAE: 4002.3195, MSE: 16082662.4165, MAPE: 40023194628721.98%, 相关系数: nan, R²: -299137520946170112.0000
    T+28天预测 - RMSE: 4707.5240, MAE: 4705.9945, MSE: 22160782.4870, MAPE: 47059944872827.01%, 相关系数: nan, R²: -409974476010402048.0000
    T+29天预测 - RMSE: 3727.7658, MAE: 3726.1805, MSE: 13896237.5978, MAPE: 37261804853320.58%, 相关系数: nan, R²: -255690771799521248.0000
    T+30天预测 - RMSE: 1728.2376, MAE: 1720.8837, MSE: 2986805.0962, MAPE: 17208837429136.93%, 相关系数: nan, R²: -54658533261310432.0000
    
    tuoshan站点:
    T+1天预测 - RMSE: 0.4886, MAE: 0.2220, MSE: 0.2387, MAPE: 547.83%, 相关系数: 0.3680, R²: -0.0568
    T+2天预测 - RMSE: 0.6834, MAE: 0.3417, MSE: 0.4670, MAPE: 364.88%, 相关系数: 0.2124, R²: -0.5353
    T+3天预测 - RMSE: 1.2497, MAE: 0.5142, MSE: 1.5617, MAPE: 1760.68%, 相关系数: -0.0614, R²: -1.0023
    T+4天预测 - RMSE: 1.1233, MAE: 0.5393, MSE: 1.2619, MAPE: 355.16%, 相关系数: -0.1226, R²: -1.2196
    T+5天预测 - RMSE: 1.0306, MAE: 0.5609, MSE: 1.0622, MAPE: 414.08%, 相关系数: -0.1099, R²: -1.2336
    T+6天预测 - RMSE: 1.0837, MAE: 0.6086, MSE: 1.1745, MAPE: 1285.02%, 相关系数: -0.1720, R²: -1.3288
    T+7天预测 - RMSE: 1.0929, MAE: 0.6725, MSE: 1.1945, MAPE: 1260.12%, 相关系数: -0.2106, R²: -1.3447
    T+8天预测 - RMSE: 1.1726, MAE: 0.7276, MSE: 1.3749, MAPE: 959.14%, 相关系数: -0.1907, R²: -1.6282
    T+9天预测 - RMSE: 1.0349, MAE: 0.6430, MSE: 1.0710, MAPE: 941.65%, 相关系数: -0.1895, R²: -1.0346
    T+10天预测 - RMSE: 1.1879, MAE: 0.7215, MSE: 1.4111, MAPE: 651.81%, 相关系数: -0.1778, R²: -1.4073
    T+11天预测 - RMSE: 1.2094, MAE: 0.7263, MSE: 1.4625, MAPE: 689.14%, 相关系数: -0.1677, R²: -1.4210
    T+12天预测 - RMSE: 1.2339, MAE: 0.7472, MSE: 1.5226, MAPE: 964.87%, 相关系数: -0.1040, R²: -1.4938
    T+13天预测 - RMSE: 1.1173, MAE: 0.7027, MSE: 1.2484, MAPE: 873.14%, 相关系数: -0.1093, R²: -1.1494
    T+14天预测 - RMSE: 1.0928, MAE: 0.7219, MSE: 1.1941, MAPE: 569.16%, 相关系数: -0.1381, R²: -1.0226
    T+15天预测 - RMSE: 1.0531, MAE: 0.7149, MSE: 1.1091, MAPE: 427.59%, 相关系数: -0.1849, R²: -1.0897
    T+16天预测 - RMSE: 1.1201, MAE: 0.7633, MSE: 1.2547, MAPE: 460.04%, 相关系数: -0.2122, R²: -1.3062
    T+17天预测 - RMSE: 1.1276, MAE: 0.7812, MSE: 1.2715, MAPE: 343.85%, 相关系数: -0.1409, R²: -1.2315
    T+18天预测 - RMSE: 1.0471, MAE: 0.7310, MSE: 1.0963, MAPE: 871.35%, 相关系数: -0.0822, R²: -1.0727
    T+19天预测 - RMSE: 1.0660, MAE: 0.7322, MSE: 1.1364, MAPE: 520.82%, 相关系数: -0.0343, R²: -0.7921
    T+20天预测 - RMSE: 1.1032, MAE: 0.7452, MSE: 1.2171, MAPE: 1331.16%, 相关系数: 0.0355, R²: -0.7142
    T+21天预测 - RMSE: 1.0852, MAE: 0.7624, MSE: 1.1776, MAPE: 393.35%, 相关系数: 0.0536, R²: -0.9280
    T+22天预测 - RMSE: 1.0511, MAE: 0.7347, MSE: 1.1049, MAPE: 258.71%, 相关系数: -0.0717, R²: -1.0226
    T+23天预测 - RMSE: 1.2619, MAE: 0.8135, MSE: 1.5923, MAPE: 590.88%, 相关系数: -0.0915, R²: -0.9980
    T+24天预测 - RMSE: 1.5168, MAE: 0.9698, MSE: 2.3005, MAPE: 365.86%, 相关系数: -0.0806, R²: -1.2446
    T+25天预测 - RMSE: 1.5782, MAE: 0.9572, MSE: 2.4908, MAPE: 367.65%, 相关系数: -0.0657, R²: -1.0015
    T+26天预测 - RMSE: 1.2438, MAE: 0.8756, MSE: 1.5471, MAPE: 311.83%, 相关系数: -0.0939, R²: -1.2360
    T+27天预测 - RMSE: 1.1353, MAE: 0.8720, MSE: 1.2889, MAPE: 3289.75%, 相关系数: 0.0069, R²: -1.3124
    T+28天预测 - RMSE: 1.0052, MAE: 0.7937, MSE: 1.0104, MAPE: 331.89%, 相关系数: 0.0573, R²: -0.9732
    T+29天预测 - RMSE: 1.0529, MAE: 0.8385, MSE: 1.1087, MAPE: 652.40%, 相关系数: 0.0173, R²: -1.0578
    T+30天预测 - RMSE: 1.0819, MAE: 0.8656, MSE: 1.1705, MAPE: 561.76%, 相关系数: -0.0296, R²: -1.1999
    
    lanshanzui站点:
    T+1天预测 - RMSE: 0.1087, MAE: 0.0725, MSE: 0.0118, MAPE: 1006.75%, 相关系数: 0.0827, R²: -2.2353
    T+2天预测 - RMSE: 0.2970, MAE: 0.1483, MSE: 0.0882, MAPE: 870.27%, 相关系数: 0.0086, R²: -9.6845
    T+3天预测 - RMSE: 0.3019, MAE: 0.2152, MSE: 0.0912, MAPE: 724.99%, 相关系数: -0.0908, R²: -4.0215
    T+4天预测 - RMSE: 0.2622, MAE: 0.1820, MSE: 0.0688, MAPE: 538.51%, 相关系数: -0.1688, R²: -2.1050
    T+5天预测 - RMSE: 0.2772, MAE: 0.2039, MSE: 0.0768, MAPE: 575.90%, 相关系数: -0.2239, R²: -2.1424
    T+6天预测 - RMSE: 0.3305, MAE: 0.2345, MSE: 0.1092, MAPE: 1390.73%, 相关系数: -0.2811, R²: -3.2175
    T+7天预测 - RMSE: 0.2740, MAE: 0.2109, MSE: 0.0751, MAPE: 623970.29%, 相关系数: -0.2161, R²: -1.9957
    T+8天预测 - RMSE: 0.2942, MAE: 0.2248, MSE: 0.0865, MAPE: 597.28%, 相关系数: -0.2249, R²: -2.3957
    T+9天预测 - RMSE: 0.3250, MAE: 0.2540, MSE: 0.1057, MAPE: 644.78%, 相关系数: -0.1834, R²: -3.0772
    T+10天预测 - RMSE: 0.3013, MAE: 0.2377, MSE: 0.0908, MAPE: 585.70%, 相关系数: -0.2119, R²: -2.3917
    T+11天预测 - RMSE: 0.2990, MAE: 0.2315, MSE: 0.0894, MAPE: 705.90%, 相关系数: -0.3588, R²: -2.1726
    T+12天预测 - RMSE: 0.3309, MAE: 0.2545, MSE: 0.1095, MAPE: 880.61%, 相关系数: -0.4404, R²: -2.3356
    T+13天预测 - RMSE: 0.3957, MAE: 0.2956, MSE: 0.1566, MAPE: 764.14%, 相关系数: -0.4707, R²: -3.4983
    T+14天预测 - RMSE: 0.4059, MAE: 0.3045, MSE: 0.1648, MAPE: 881.08%, 相关系数: -0.3493, R²: -3.9918
    T+15天预测 - RMSE: 0.3875, MAE: 0.3029, MSE: 0.1502, MAPE: 458.36%, 相关系数: -0.4710, R²: -3.3863
    T+16天预测 - RMSE: 0.3253, MAE: 0.2531, MSE: 0.1059, MAPE: 332.42%, 相关系数: -0.3935, R²: -2.0686
    T+17天预测 - RMSE: 0.3751, MAE: 0.2887, MSE: 0.1407, MAPE: 479.15%, 相关系数: -0.3436, R²: -3.1037
    T+18天预测 - RMSE: 0.3413, MAE: 0.2737, MSE: 0.1165, MAPE: 563.69%, 相关系数: -0.3987, R²: -2.3999
    T+19天预测 - RMSE: 0.3240, MAE: 0.2617, MSE: 0.1050, MAPE: 597.65%, 相关系数: -0.3259, R²: -2.0430
    T+20天预测 - RMSE: 0.3891, MAE: 0.3118, MSE: 0.1514, MAPE: 441.47%, 相关系数: -0.3043, R²: -3.4133
    T+21天预测 - RMSE: 0.4910, MAE: 0.3872, MSE: 0.2411, MAPE: 555.72%, 相关系数: -0.2874, R²: -5.8287
    T+22天预测 - RMSE: 0.3469, MAE: 0.2876, MSE: 0.1203, MAPE: 343.70%, 相关系数: -0.2179, R²: -2.6269
    T+23天预测 - RMSE: 0.3548, MAE: 0.2878, MSE: 0.1259, MAPE: 1679.67%, 相关系数: -0.1893, R²: -3.0178
    T+24天预测 - RMSE: 0.4381, MAE: 0.3345, MSE: 0.1919, MAPE: 593.96%, 相关系数: -0.2614, R²: -4.4658
    T+25天预测 - RMSE: 0.2972, MAE: 0.2310, MSE: 0.0883, MAPE: 441.79%, 相关系数: -0.0238, R²: -1.9665
    T+26天预测 - RMSE: 0.3244, MAE: 0.2533, MSE: 0.1053, MAPE: 365.19%, 相关系数: -0.1332, R²: -2.5111
    T+27天预测 - RMSE: 0.3285, MAE: 0.2574, MSE: 0.1079, MAPE: 476.56%, 相关系数: -0.1877, R²: -2.4945
    T+28天预测 - RMSE: 0.4531, MAE: 0.3459, MSE: 0.2053, MAPE: 4298.68%, 相关系数: -0.1229, R²: -6.2814
    T+29天预测 - RMSE: 0.3250, MAE: 0.2526, MSE: 0.1057, MAPE: 341.05%, 相关系数: -0.0975, R²: -2.8676
    T+30天预测 - RMSE: 0.3429, MAE: 0.2757, MSE: 0.1176, MAPE: 3462.89%, 相关系数: -0.2896, R²: -3.1234
    
    五里湖心站点:
    T+1天预测 - RMSE: 0.2380, MAE: 0.1651, MSE: 0.0567, MAPE: 1135.68%, 相关系数: -0.0379, R²: -0.6842
    T+2天预测 - RMSE: 0.5142, MAE: 0.3166, MSE: 0.2644, MAPE: 391.65%, 相关系数: -0.1638, R²: -0.7758
    T+3天预测 - RMSE: 0.6081, MAE: 0.4155, MSE: 0.3698, MAPE: 1031.93%, 相关系数: -0.1728, R²: -0.9010
    T+4天预测 - RMSE: 0.6714, MAE: 0.4620, MSE: 0.4507, MAPE: 775.93%, 相关系数: -0.2635, R²: -1.0658
    T+5天预测 - RMSE: 0.7917, MAE: 0.5276, MSE: 0.6268, MAPE: 383.49%, 相关系数: -0.2803, R²: -1.1911
    T+6天预测 - RMSE: 0.8584, MAE: 0.5650, MSE: 0.7368, MAPE: 524.40%, 相关系数: -0.1059, R²: -1.1088
    T+7天预测 - RMSE: 0.8037, MAE: 0.5262, MSE: 0.6459, MAPE: 358.80%, 相关系数: 0.0482, R²: -0.7483
    T+8天预测 - RMSE: 0.9179, MAE: 0.6320, MSE: 0.8426, MAPE: 306.50%, 相关系数: 0.0366, R²: -1.2547
    T+9天预测 - RMSE: 0.8630, MAE: 0.5792, MSE: 0.7448, MAPE: 411.82%, 相关系数: -0.0255, R²: -1.3337
    T+10天预测 - RMSE: 0.7535, MAE: 0.5527, MSE: 0.5678, MAPE: 326.06%, 相关系数: -0.0329, R²: -0.8484
    T+11天预测 - RMSE: 0.8209, MAE: 0.5404, MSE: 0.6738, MAPE: 226.79%, 相关系数: 0.0374, R²: -0.6858
    T+12天预测 - RMSE: 1.0545, MAE: 0.6739, MSE: 1.1119, MAPE: 456.44%, 相关系数: -0.1496, R²: -1.0374
    T+13天预测 - RMSE: 1.1594, MAE: 0.7014, MSE: 1.3441, MAPE: 231.95%, 相关系数: -0.1977, R²: -1.0186
    T+14天预测 - RMSE: 1.2087, MAE: 0.7508, MSE: 1.4609, MAPE: 862.22%, 相关系数: -0.1801, R²: -0.9255
    T+15天预测 - RMSE: 1.3353, MAE: 0.8233, MSE: 1.7831, MAPE: 273.77%, 相关系数: -0.2056, R²: -1.0288
    T+16天预测 - RMSE: 1.3653, MAE: 0.8142, MSE: 1.8639, MAPE: 632.48%, 相关系数: -0.1507, R²: -0.9175
    T+17天预测 - RMSE: 1.2973, MAE: 0.8134, MSE: 1.6829, MAPE: 500.59%, 相关系数: -0.0092, R²: -0.8885
    T+18天预测 - RMSE: 1.2583, MAE: 0.8376, MSE: 1.5832, MAPE: 1026.73%, 相关系数: -0.0116, R²: -0.7278
    T+19天预测 - RMSE: 1.2252, MAE: 0.6798, MSE: 1.5012, MAPE: 544.01%, 相关系数: 0.0706, R²: -0.6340
    T+20天预测 - RMSE: 1.1499, MAE: 0.6968, MSE: 1.3223, MAPE: 526.85%, 相关系数: 0.1419, R²: -0.5348
    T+21天预测 - RMSE: 0.9975, MAE: 0.6600, MSE: 0.9949, MAPE: 766.63%, 相关系数: 0.2443, R²: -0.3588
    T+22天预测 - RMSE: 1.1380, MAE: 0.7323, MSE: 1.2950, MAPE: 590.89%, 相关系数: -0.0757, R²: -1.0775
    T+23天预测 - RMSE: 1.2938, MAE: 0.7994, MSE: 1.6740, MAPE: 1084.62%, 相关系数: -0.0652, R²: -0.9327
    T+24天预测 - RMSE: 1.2690, MAE: 0.7671, MSE: 1.6104, MAPE: 1016.53%, 相关系数: 0.1215, R²: -0.5663
    T+25天预测 - RMSE: 1.1907, MAE: 0.8182, MSE: 1.4178, MAPE: 325.58%, 相关系数: 0.1160, R²: -0.4373
    T+26天预测 - RMSE: 1.0012, MAE: 0.7201, MSE: 1.0025, MAPE: 434.11%, 相关系数: 0.3432, R²: -0.1637
    T+27天预测 - RMSE: 0.9735, MAE: 0.7349, MSE: 0.9477, MAPE: 563.97%, 相关系数: 0.3225, R²: -0.3316
    T+28天预测 - RMSE: 0.9017, MAE: 0.7052, MSE: 0.8130, MAPE: 336.74%, 相关系数: 0.3252, R²: -0.4468
    T+29天预测 - RMSE: 0.8877, MAE: 0.7108, MSE: 0.7880, MAPE: 319.52%, 相关系数: 0.4790, R²: 0.0701
    T+30天预测 - RMSE: 1.1414, MAE: 0.8300, MSE: 1.3027, MAPE: 364.19%, 相关系数: 0.2194, R²: -0.7707
    

    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3065: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    c:\Users\hdec\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\lib\_function_base_impl.py:3066: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]
    
