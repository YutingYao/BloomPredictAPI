# 蓝藻预测系统后端API

基于多种机器学习模型的太湖流域蓝藻密度预测系统后端API实现。

## 🌟 项目特点

- **多模型支持**: 集成LSTM、GRU-D、TCN、XGBoost四种预测模型
- **多站点覆盖**: 支持太湖流域6个重要监测站点
- **高性能预测**: GRU-D模型平均改善率达50.07%
- **完整API**: RESTful API设计，支持预测和性能对比
- **异步架构**: 基于FastAPI的高性能异步Web框架

## 📊 监测站点

### 重污染与高风险区
- **胥湖心**: GRU-D模型改善率77.41%
- **锡东水厂**: GRU-D模型改善率55.13%

### 背景与参照区  
- **平台山**: XGBoost模型表现优异
- **拖山(tuoshan)**: GRU-D模型改善率79.78%

### 边界条件区
- **兰山嘴(lanshanzui)**: GRU-D模型改善率28.83%
- **五里湖心**: GRU-D模型改善率42.93%

## 🤖 支持的模型

| 模型 | 类型 | 特点 | 推荐场景 |
|------|------|------|----------|
| **GRU-D** ⭐ | 循环神经网络 | 专为缺失数据设计，整体最优 | 推荐用于所有场景 |
| **LSTM** | 循环神经网络 | 基准模型，长期记忆能力强 | 对比基准 |
| **TCN** | 卷积神经网络 | 并行计算效率高 | 长序列预测 |
| **XGBoost** | 梯度提升树 | 可解释性强，部分站点优异 | 特定站点优化 |

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 建议使用虚拟环境

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd 后端API

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 配置环境

```bash
# 复制环境配置文件
cp env.example .env

# 编辑配置文件（可选）
nano .env
```

### 启动服务

```bash
# 开发模式启动
python main.py

# 或使用uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

服务启动后访问：
- API文档: http://localhost:8000/docs
- 替代文档: http://localhost:8000/redoc
- 健康检查: http://localhost:8000/health

## 📡 API接口

### 1. 预测接口

**POST** `/api/predict`

预测指定站点的蓝藻密度增长率。

```json
{
  "station": "胥湖心",
  "model_type": "grud",
  "predict_days": 7,
  "input_data": {
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
```

**响应示例:**

```json
{
  "success": true,
  "data": {
    "station": "胥湖心",
    "model_type": "grud",
    "predict_days": 7,
    "prediction": [0.12, 0.15, 0.18, 0.21, 0.19, 0.17, 0.16],
    "confidence": 0.85,
    "rmse": 0.2567,
    "model_info": {
      "name": "GRU-D",
      "description": "专为缺失数据设计的GRU改进版",
      "improvement_over_lstm": "77.41%"
    }
  },
  "message": "预测成功",
  "timestamp": "2024-01-20T10:30:00.000Z"
}
```

### 2. 模型性能对比接口

**POST** `/api/model-performance`

获取模型性能对比信息。

```json
{
  "station": "胥湖心",
  "model_types": ["lstm", "grud", "tcn"]
}
```

### 3. 配置信息接口

- **GET** `/api/stations` - 获取支持的监测站点
- **GET** `/api/models` - 获取支持的预测模型
- **GET** `/api/input-schema` - 获取输入数据格式说明

## 🔧 输入数据格式

### 模型参数

| 参数 | 描述 | 默认值 | 备注 |
|------|------|------|------|
| seq_length | 输入序列长度 | 60 | 模型使用60天的历史数据作为输入窗口 |
| predict_days | 预测天数 | 1-30 | 支持预测未来1-30天的蓝藻密度增长率 |

### 必需字段

| 字段 | 描述 | 单位 | 范围 |
|------|------|------|------|
| temperature | 温度 | °C | -5 ~ 40 |
| oxygen | 溶解氧 | mg/L | 0 ~ 20 |
| TN | 总氮 | mg/L | 0 ~ 10 |
| TP | 总磷 | mg/L | 0 ~ 1 |
| NH | 氨氮 | mg/L | 0 ~ 5 |
| pH | pH值 | - | 5 ~ 10 |
| turbidity | 浊度 | NTU | 0 ~ 200 |
| conductivity | 电导率 | μS/cm | 0 ~ 2000 |
| permanganate | 高锰酸盐指数 | mg/L | 0 ~ 20 |
| rain_sum | 降雨量 | mm | 0 ~ 200 |
| wind_speed_10m_max | 风速 | m/s | 0 ~ 30 |
| shortwave_radiation_sum | 短波辐射 | MJ/m² | 0 ~ 50 |

## 📁 项目结构

```
后端API/
├── main.py                    # 应用入口
├── requirements.txt           # 依赖管理
├── env.example               # 环境配置示例
├── src/                      # 源代码
│   ├── config/              # 配置模块
│   │   ├── __init__.py
│   │   └── settings.py      # 应用设置
│   ├── models/              # 模型管理
│   │   ├── __init__.py
│   │   └── model_manager.py # 模型管理器
│   ├── services/            # 业务逻辑
│   │   ├── __init__.py
│   │   └── prediction_service.py # 预测服务
│   ├── schemas/             # 数据模型
│   │   ├── __init__.py
│   │   ├── request_schemas.py  # 请求模型
│   │   └── response_schemas.py # 响应模型
│   ├── utils/               # 工具函数
│   │   ├── __init__.py
│   │   ├── validators.py    # 数据验证
│   │   └── data_processor.py # 数据处理
│   └── __init__.py
├── logs/                    # 日志目录
└── *.pkl                   # 预训练模型文件
```

## 🎯 使用示例

### Python客户端示例

```python
import requests
import json

# API基础URL
BASE_URL = "http://localhost:8000"

# 预测请求
def predict_algae_density():
    url = f"{BASE_URL}/api/predict"
    
    payload = {
        "station": "胥湖心",
        "model_type": "grud", 
        "predict_days": 7,
        "input_data": {
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
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            predictions = result["data"]["prediction"]
            print(f"预测结果: {predictions}")
            return predictions
    
    print(f"请求失败: {response.text}")
    return None

# 执行预测
predictions = predict_algae_density()
```

### cURL示例

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "station": "胥湖心",
    "model_type": "grud",
    "predict_days": 7,
    "input_data": {
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
  }'
```

## ⚡ 性能优化

### 模型加载策略
- 异步模型初始化
- 多线程模型预测
- 模型缓存管理

### 配置建议
- 生产环境建议使用Gunicorn + Nginx
- 可配置模型预加载策略
- 支持负载均衡部署

## 🐳 Docker部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

构建和运行：

```bash
docker build -t algae-prediction-api .
docker run -p 8000:8000 algae-prediction-api
```

## 📈 模型性能

### 整体表现
- **GRU-D**: 平均改善率50.07%，推荐模型
- **XGBoost**: 平均改善率16.43%，在部分站点优异  
- **TCN**: 平均改善率-1.88%，整体表现一般
- **LSTM**: 基准模型，作为对比基础

### 各站点最佳模型
- 胥湖心、锡东水厂、拖山、兰山嘴、五里湖心: **GRU-D**
- 平台山: **XGBoost**

## 🔍 故障排除

### 常见问题

1. **模型文件未找到**
   ```
   错误: 模型文件不存在
   解决: 确保所有.pkl模型文件在项目根目录
   ```

2. **端口占用**
   ```bash
   # 查找占用端口的进程
   lsof -i :8000
   # 杀掉进程
   kill -9 <PID>
   ```

3. **依赖冲突**
   ```bash
   # 重新创建虚拟环境
   rm -rf venv
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### 日志查看

```bash
# 查看API日志
tail -f logs/api.log

# 查看实时日志
python main.py
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目链接: [GitHub Repository](https://github.com/your-username/algae-prediction-api)
- 问题反馈: [Issues](https://github.com/your-username/algae-prediction-api/issues)

## 🙏 致谢

- 太湖流域水环境监测数据提供方
- 深度学习模型训练数据集贡献者
- FastAPI和相关开源项目的维护者
