# 蓝藻预测系统API V3 - 极简版本

欢迎使用蓝藻预测系统API V3版本。本项目旨在提供一个**极简、智能、零配置**的蓝藻水华预测服务。

V3版本完美解决了之前版本中接口复杂的核心问题。用户**只需提供4个核心参数**，系统自动完成所有数据处理和模型推理工作。

- ✅ **极简设计**：只需4个核心参数
- ✅ **零配置**：无需关心数据获取、预处理等细节
- ✅ **全自动**：历史数据自动获取，实时数据自动更新
- ✅ **智能推理**：根据当前日期自动确定历史数据范围

## 🚀 快速开始

### 最简单的预测请求

只需要4个参数即可完成预测：

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

## ✨ 核心特性

### 1. 极简设计 - 只需4个参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `current_date` | 当前日期 | "2024-06-01" |
| `predict_days` | 预测天数 (1-30) | 7 |
| `station` | 预测点位 | "胥湖心" |
| `model_type` | 模型类型 | "grud" |

### 2. 全自动数据处理

- **自动历史数据获取**：根据`current_date`自动获取前60天历史数据
- **自动数据预处理**：缺失值填充、数据标准化等
- **实时数据更新**：通过`generate_fake_data.py`自动更新到CSV文件

### 3. 智能推理机制

```
用户输入: current_date = "2024-06-01"
        ↓
系统自动计算: 历史数据截止 = "2024-05-31"
        ↓ 
自动获取: 2024-04-02 到 2024-05-31 (60天历史数据)
        ↓
执行预测: 2024-06-01 到 2024-06-07 (未来7天)
```

## 📝 使用示例

### 基础预测

```python
import requests

def simple_predict():
    url = "http://localhost:8002/api/v3/predict"
    
    # 只需4个核心参数
    payload = {
        "current_date": "2024-06-01",  # 当前日期
        "predict_days": 7,             # 预测7天
        "station": "胥湖心",           # 监测站点
        "model_type": "grud"           # 推荐使用GRUD模型
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    print(f"预测结果: {result['data']['prediction']}")
    print(f"预测日期: {result['data']['prediction_dates']}")

simple_predict()
```

### 批量预测

```python
def batch_predict():
    url = "http://localhost:8002/api/v3/batch-predict"
    
    payload = {
        "requests": [
            {
                "current_date": "2024-06-01",
                "predict_days": 5,
                "station": "胥湖心",
                "model_type": "grud"
            },
            {
                "current_date": "2024-06-01", 
                "predict_days": 5,
                "station": "锡东水厂",
                "model_type": "grud"
            }
        ],
        "parallel_execution": True
    }
    
    response = requests.post(url, json=payload)
    return response.json()

result = batch_predict()
print(result)
```

### 请求验证

```python
def validate_request():
    url = "http://localhost:8002/api/v3/validate"
    
    payload = {
        "current_date": "2024-06-01",
        "predict_days": 14,
        "station": "胥湖心", 
        "model_type": "grud"
    }
    
    response = requests.post(url, json=payload)
    validation = response.json()['validation_result']
    
    print(f"请求有效: {validation['valid']}")
    print(f"自动截止日期: {validation['auto_end_date']}")
    
validate_request()
```

## 🎯 支持的站点和模型

### 监测站点 (6个)

| 站点名称 | 英文名称 | 推荐模型 |
|---------|----------|----------|
| 胥湖心 | Xuhu Center | grud |
| 锡东水厂 | Xidong Water Plant | grud |
| 平台山 | Pingtai Mountain | xgboost |
| tuoshan | Tuoshan Mountain | grud |
| lanshanzui | Lanshan Cape | grud |
| 五里湖心 | Wulihu Center | grud |

### 预测模型 (4种)

| 模型类型 | 名称 | 特点 | 推荐度 |
|---------|------|------|--------|
| grud | GRU-D | 处理缺失数据，精度高 | ⭐⭐⭐⭐⭐ |
| lstm | LSTM | 基准模型，稳定可靠 | ⭐⭐⭐ |
| tcn | TCN | 并行计算，速度快 | ⭐⭐⭐ |
| xgboost | XGBoost | 在平台山表现最佳 | ⭐⭐⭐⭐ |

## 📚 API 参考

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v3/predict` | POST | 极简预测接口 (4参数) |
| `/api/v3/batch-predict` | POST | 批量预测 |
| `/api/v3/validate` | POST | 请求验证 |
| `/api/stations` | GET | 支持的站点列表 |
| `/api/models` | GET | 支持的模型列表 |
| `/api/v3/input-schema` | GET | 输入格式说明 |

完整的API文档和在线测试工具请访问 `http://localhost:8002/docs`

## 🛠️ 本地部署

### 环境准备

```bash
# 使用阿里云镜像安装依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 启动服务

```bash
# 启动V3 API服务
source venv/bin/activate && python main_v3.py
```

服务将在 `http://localhost:8002` 启动。

## 🔄 版本对比

### API调用复杂度对比

**V1版本 (原版)**
```json
{
  "station": "胥湖心",
  "model_type": "grud",
  "predict_days": 7,
  "input_data": {
    // 需要720个数值 (60天×12特征)
    "temperature": [25.1, 25.2, ...], // 60个值
    "oxygen": [8.1, 8.2, ...],        // 60个值
    // ... 10个特征 × 60天 = 600个数值
  }
}
```
❌ **问题**: 需要手动构造720个数值，用户体验极差

**V2版本 (改进版)**
```json
{
  "station": "胥湖心",
  "model_type": "grud", 
  "predict_days": 7,
  "data_mode": "auto_historical",
  "end_date": "2024-05-31",
  "seq_length": 60,
  "fill_missing_method": "interpolation"
}
```
✅ **改进**: 自动获取历史数据  
❓ **问题**: 仍需多个配置参数

**V3版本 (极简版)**
```json
{
  "current_date": "2024-06-01",
  "predict_days": 7,
  "station": "胥湖心",
  "model_type": "grud"
}
```
✅ **优势**: 只需4个核心参数，零配置

### 功能对比表

| 特性 | V1版本 | V2版本 | V3版本 |
|------|--------|--------|--------|
| 参数数量 | 720+ | 6-8个 | **4个** |
| 用户体验 | ❌ 复杂 | ⚠️ 一般 | ✅ **极简** |
| 配置要求 | ❌ 高 | ⚠️ 中等 | ✅ **零配置** |
| 自动化程度 | ❌ 低 | ⚠️ 部分 | ✅ **全自动** |
| 学习成本 | ❌ 高 | ⚠️ 中等 | ✅ **极低** |

## 🔄 实时数据更新机制

V3版本的数据更新完全自动化，用户无需感知：

```
实时监测数据 → generate_fake_data.py → 自动更新CSV文件 → API自动使用最新数据
```

1. **数据采集**: 监测设备产生新的实时数据
2. **自动更新**: `generate_fake_data.py` 定期运行，将新数据追加到历史数据CSV文件
3. **透明使用**: API自动使用最新的历史数据，用户无需任何操作

## 🚀 演示脚本

```bash
# 运行V3版本演示
python demo_api_v3.py
```

演示脚本将展示：
- 极简预测接口使用
- 批量预测功能
- 请求验证
- API版本对比

## 📞 技术支持

- **API文档**: http://localhost:8002/docs
- **演示脚本**: `python demo_api_v3.py`
- **日志文件**: `logs/api_v3.log`

---

🎯 **设计理念**: "极简至上，智能至上"  
🚀 **核心价值**: 用最少的参数，获得最准确的预测
