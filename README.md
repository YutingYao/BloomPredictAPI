# 蓝藻预测系统API - 完整版

欢迎使用蓝藻预测系统API，这是一个智能、高效、易用的蓝藻水华预测服务平台，专为太湖流域水环境监测设计。

## 🚀 项目概述

本项目提供了三个不同复杂度的API版本，满足从研究开发到生产部署的全方位需求：

- **V1版本**: 完整功能版本，适合研究和深度定制
- **V2版本**: 简化版本，平衡了功能和易用性  
- **V3版本**: 极简版本，仅需4个参数，适合快速集成

## 📊 项目结构

```
/home/devbox/project/--API/
├── 📁 src/                        # 核心源代码
│   ├── 📁 config/                 # 配置管理
│   │   ├── __init__.py
│   │   └── settings.py           # 系统配置
│   ├── 📁 models/                 # 模型管理
│   │   ├── __init__.py
│   │   └── model_manager.py      # 模型加载和管理
│   ├── 📁 schemas/                # 数据模型
│   │   ├── __init__.py
│   │   ├── request_schemas.py    # V1请求模型
│   │   ├── request_schemas_v2.py # V2请求模型
│   │   ├── request_schemas_v3.py # V3请求模型
│   │   └── response_schemas.py   # 响应模型
│   ├── 📁 services/               # 业务服务
│   │   ├── __init__.py
│   │   ├── prediction_service.py    # V1预测服务
│   │   ├── prediction_service_v2.py # V2预测服务
│   │   ├── prediction_service_v3.py # V3预测服务
│   │   ├── mock_prediction_service_v3.py # V3模拟服务
│   │   └── historical_data_service.py   # 历史数据服务
│   └── 📁 utils/                  # 工具库
│       ├── __init__.py
│       ├── data_processor.py     # 数据处理
│       └── validators.py         # 数据验证
├── 📁 models/                     # 训练好的ML模型
│   ├── 00-lstm_model_data_*.pkl  # LSTM模型文件
│   ├── 00-GRUD_model_data_*.pkl  # GRU-D模型文件
│   ├── 00-TCN_model_data_*.pkl   # TCN模型文件
│   └── 00-XGB_model_data_*.pkl   # XGBoost模型文件
├── 📁 data/                       # 历史数据文件
│   └── 002-气象-*-merged_with_weather_with_composite_features_processed.csv
├── 📁 test/                       # 测试文件
│   ├── test_api.py               # API测试
│   ├── test_model_loading.py     # 模型加载测试
│   └── test_*.py                 # 其他测试
├── 📁 docs/                       # 文档
│   ├── API简化总结-20250817.md    # API简化总结
│   ├── V3版本简化API使用指南.md   # V3使用指南
│   └── *.md                      # 其他文档
├── 📁 logs/                       # 日志文件
│   ├── api.log                   # V1日志
│   ├── api_v2.log               # V2日志
│   └── api_v3.log               # V3日志
├── main.py                        # V1主程序
├── main_v2.py                     # V2主程序
├── main_v3.py                     # V3主程序
├── start_server.py                # 服务器启动脚本
├── demo_api_v2.py                 # V2演示
├── demo_api_v3.py                 # V3演示
├── compare_versions.py            # 版本对比工具
├── requirements.txt               # 依赖列表
└── README.md                     # 本文档
```

## 🎯 快速开始

### 环境要求

- Python 3.8+
- 虚拟环境 (推荐)
- 8GB+ 内存 (用于加载ML模型)

### 安装依赖

```bash
# 使用阿里云镜像安装依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 启动服务

```bash
# 激活虚拟环境并启动服务
source venv/bin/activate && python start_server.py
```

或者选择特定版本：

```bash
# V1版本 (端口8000)
python main.py

# V2版本 (端口8001) - 推荐
python main_v2.py  

# V3版本 (端口8002) - 极简版
python main_v3.py
```

## 📋 API版本对比

| 特性 | V1版本 | V2版本 | V3版本 |
|------|--------|--------|--------|
| 参数数量 | 720+ | 6-8个 | **4个** |
| 学习成本 | 极高 | 中等 | **极低** |
| 配置复杂度 | 极复杂 | 中等 | **零配置** |
| 用户体验 | 极差 | 一般 | **极佳** |
| 业务适用性 | 研究用 | 开发用 | **生产用** |
| API端口 | 8000 | 8001 | 8002 |
| 推荐场景 | 深度研究 | 平衡使用 | 快速集成 |

## 🌟 V3版本 - 极简API (推荐)

### 核心特性

- **极简接口**: 仅需4个核心参数
- **零配置**: 全自动数据处理和模型管理
- **智能预测**: 基于当前日期自动获取历史数据

### 快速调用示例

```bash
curl -X POST 'http://localhost:8002/api/v3/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "current_date": "2024-06-01",
    "predict_days": 7,
    "station": "胥湖心",
    "model_type": "grud"
  }'
```

### Python客户端示例

```python
import requests

def predict_algae_v3():
    """V3版本预测示例"""
    url = "http://localhost:8002/api/v3/predict"
    
    payload = {
        "current_date": "2024-06-01",  # 当前日期
        "predict_days": 7,             # 预测天数
        "station": "胥湖心",            # 监测站点
        "model_type": "grud"           # 模型类型
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    if result["success"]:
        print(f"预测成功！")
        for i, pred in enumerate(result["data"]["prediction"], 1):
            print(f"第{i}天: {pred:.2f}")
    else:
        print(f"预测失败: {result['error']}")

predict_algae_v3()
```

## 🛠️ V2版本 - 平衡版

V2版本在功能完整性和易用性之间取得了很好的平衡，支持更多高级功能。

### 核心功能

- **智能历史数据获取**: 自动从CSV文件加载60天历史数据
- **混合模式**: 支持实时数据补充
- **批量预测**: 多站点并行预测
- **数据质量验证**: 内置数据检查

### 使用示例

```python
import requests

# 基础预测
def predict_v2_basic():
    url = "http://localhost:8001/api/v2/predict"
    payload = {
        "station": "胥湖心",
        "model_type": "grud",
        "predict_days": 7,
        "data_mode": "auto_historical",
        "end_date": "2024-05-31"
    }
    return requests.post(url, json=payload).json()

# 混合模式（历史 + 实时数据）
def predict_v2_hybrid():
    url = "http://localhost:8001/api/v2/predict"
    payload = {
        "station": "胥湖心",
        "model_type": "grud", 
        "predict_days": 5,
        "data_mode": "hybrid",
        "end_date": "2024-05-31",
        "supplementary_data": [
            {"date": "2024-05-30", "temperature": 27.1, "oxygen": 7.9, "pH": 7.9},
            {"date": "2024-05-31", "temperature": 28.0, "oxygen": 7.5, "pH": 8.0}
        ]
    }
    return requests.post(url, json=payload).json()
```

## 🎯 支持的站点和模型

### 监测站点 (6个)

| 站点名称 | 英文名称 | 推荐模型 | 精度提升 |
|---------|----------|----------|---------|
| 胥湖心 | Xuhu Center | grud | 77.4% |
| 锡东水厂 | Xidong Water Plant | grud | 55.1% |
| 平台山 | Pingtai Mountain | xgboost | 100% |
| tuoshan | Tuoshan Mountain | grud | 79.8% |
| lanshanzui | Lanshan Cape | grud | 28.8% |
| 五里湖心 | Wulihu Center | grud | 42.9% |

### 预测模型 (4种)

| 模型类型 | 名称 | 特点 | 推荐度 |
|---------|------|------|--------|
| grud | GRU-D | 处理缺失数据，精度高 | ⭐⭐⭐⭐⭐ |
| lstm | LSTM | 基准模型，稳定可靠 | ⭐⭐⭐ |
| tcn | TCN | 并行计算，速度快 | ⭐⭐⭐ |
| xgboost | XGBoost | 在平台山表现最佳 | ⭐⭐⭐⭐ |

## 📚 API端点总览

### V3版本端点 (推荐)

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v3/predict` | POST | 极简预测接口 |
| `/api/v3/batch-predict` | POST | 批量预测 |
| `/api/v3/validate` | POST | 请求验证 |

### V2版本端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v2/predict` | POST | 核心预测接口 |
| `/api/v2/batch-predict` | POST | 批量预测 |
| `/api/v2/validate-request` | POST | 请求验证 |
| `/api/v2/data-info/{station}` | GET | 站点数据摘要 |
| `/api/v2/historical-data` | POST | 获取历史数据 |

### 通用端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/docs` | GET | API文档 (Swagger) |
| `/redoc` | GET | API文档 (ReDoc) |

## 🔧 技术架构

### 核心技术栈

- **Web框架**: FastAPI 0.104.1
- **异步支持**: Uvicorn + AsyncIO
- **机器学习**: PyTorch 2.1.1, XGBoost 2.0.2
- **数据处理**: Pandas 2.0.3, NumPy 1.24.3
- **数据验证**: Pydantic 2.5.0

### 架构设计

```
┌─────────────────────────────────────────┐
│           FastAPI Application          │
├─────────────────────────────────────────┤
│  Router Layer (路由层)                 │
│  ├── V1: main.py                      │
│  ├── V2: main_v2.py                   │
│  └── V3: main_v3.py                   │
├─────────────────────────────────────────┤
│  Service Layer (服务层)                │
│  ├── PredictionService (预测服务)      │
│  ├── HistoricalDataService (数据服务)  │
│  └── ModelManager (模型管理)           │
├─────────────────────────────────────────┤
│  Model Layer (模型层)                  │
│  ├── LSTM Models                      │
│  ├── GRU-D Models                     │
│  ├── TCN Models                       │
│  └── XGBoost Models                   │
├─────────────────────────────────────────┤
│  Data Layer (数据层)                   │
│  └── CSV Historical Data Files        │
└─────────────────────────────────────────┘
```

### 数据流程

```
用户请求 → 参数验证 → 历史数据获取 → 数据预处理 → 模型预测 → 结果返回
    ↓         ↓           ↓           ↓           ↓           ↓
 Schema检查  模型可用性   CSV文件读取   特征工程     ML推理    JSON响应
```

## 🧪 测试和演示

### 运行测试

```bash
# 基础API测试
python test/test_api.py

# 模型加载测试
python test/test_model_loading.py

# 综合功能测试
python test/test_comprehensive_v2.py
```

### 演示脚本

```bash
# V2版本演示
python demo_api_v2.py

# V3版本演示
python demo_api_v3.py

# 版本对比演示
python compare_versions.py

# 30天预报测试
python test_30_days_forecast.py
```

## 📊 性能优化

### 模型管理
- **懒加载**: 按需加载模型，节省内存
- **缓存机制**: 已加载模型保持在内存中
- **并发控制**: 合理控制并发预测请求

### 数据处理
- **异步IO**: 非阻塞文件读取
- **数据验证**: Pydantic高效验证
- **内存优化**: 及时释放大型数据结构

### API响应
- **异步处理**: 所有端点支持async/await
- **并行预测**: 批量请求支持并行处理
- **错误处理**: 统一的异常处理机制

## 🐛 故障排除

### 常见问题

1. **模型文件缺失**
   ```bash
   # 检查模型文件
   ls -la models/
   # 确保存在24个.pkl模型文件（6个站点 × 4种模型）
   ```

2. **端口被占用**
   ```bash
   # 检查端口使用
   lsof -i :8000  # V1
   lsof -i :8001  # V2  
   lsof -i :8002  # V3
   ```

3. **依赖问题**
   ```bash
   # 重新安装依赖
   pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
   ```

4. **内存不足**
   ```bash
   # 检查系统内存
   free -h
   # 建议至少8GB内存用于加载所有模型
   ```

### 日志查看

```bash
# 查看不同版本的日志
tail -f logs/api.log      # V1日志
tail -f logs/api_v2.log   # V2日志
tail -f logs/api_v3.log   # V3日志
```

## 🔄 版本迁移指南

### 从V1迁移到V2

**V1请求**:
```json
{
  "station": "胥湖心",
  "model_type": "grud",
  "predict_days": 7,
  "input_data": { /* 720个数值 */ }
}
```

**V2请求**:
```json
{
  "station": "胥湖心",
  "model_type": "grud", 
  "predict_days": 7,
  "data_mode": "auto_historical",
  "end_date": "2024-05-31"
}
```

### 从V2迁移到V3

**V2请求**:
```json
{
  "station": "胥湖心",
  "model_type": "grud",
  "predict_days": 7, 
  "data_mode": "auto_historical",
  "end_date": "2024-05-31"
}
```

**V3请求**:
```json
{
  "current_date": "2024-06-01",
  "predict_days": 7,
  "station": "胥湖心",
  "model_type": "grud"
}
```

## 📋 开发指南

### 添加新站点

1. 在`data/`目录添加站点CSV文件
2. 在`models/`目录添加对应的4种模型文件
3. 更新`src/config/settings.py`中的站点配置
4. 运行测试确保功能正常

### 添加新模型

1. 训练新模型并保存为`.pkl`文件
2. 更新`src/models/model_manager.py`
3. 在所有schema文件中添加新的model_type选项
4. 更新文档和测试

### 代码规范

- 遵循PEP 8编码规范
- 使用类型提示 (Type Hints)
- 编写充分的文档字符串
- 保持单一职责原则
- 添加必要的单元测试

## 🤝 贡献指南

1. **Fork** 项目到您的GitHub账户
2. **创建** 功能分支 (`git checkout -b feature/amazing-feature`)
3. **提交** 您的修改 (`git commit -m 'Add amazing feature'`)
4. **推送** 到分支 (`git push origin feature/amazing-feature`)
5. **创建** Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 技术支持

- **API文档**: 访问 `/docs` 端点查看完整API文档
- **演示脚本**: 运行项目中的演示文件了解用法
- **日志文件**: 查看 `logs/` 目录了解运行状态
- **测试套件**: 运行 `test/` 目录中的测试文件

---

## 🎯 总结

蓝藻预测系统API通过三个版本的迭代，实现了从研究工具到生产级服务的完美转变：

- **V1**: 功能完整，适合深度研究和定制开发
- **V2**: 平衡易用性和功能性，适合大多数应用场景  
- **V3**: 极简设计，仅需4个参数，适合快速集成和生产部署

选择适合您需求的版本，开始预测太湖蓝藻水华吧！ 🌊