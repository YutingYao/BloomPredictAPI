# 蓝藻预测系统V3版本简化API使用指南

**创建日期**: 2024年8月17日  
**版本**: V3.0.0  
**文档类型**: 用户使用指南  
**目标用户**: 计算机小白、环保工作人员、API集成开发者  

## 📋 文档概述

本指南详细介绍了蓝藻预测系统V3版本的使用方法。V3版本采用革命性的极简设计，将复杂的720个参数简化为4个核心参数，任何人都能轻松使用。

## 🎯 V3版本核心优势

### 极简设计 - 只需4个参数
- ❌ V1版本需要: 720个数值参数（60天×12特征）
- ❌ V2版本需要: 6-8个配置参数
- ✅ **V3版本只需: 4个核心参数**

### 零配置使用
- 无需了解技术细节
- 无需准备历史数据
- 无需配置复杂参数
- 输入业务信息即可预测

## 🚀 快速开始

### 第一步: 确认服务运行

访问健康检查端点，确认服务正常：
```bash
# 方法1: 使用curl命令
curl http://localhost:8002/health

# 方法2: 在浏览器中访问
http://localhost:8002/health
```

**正常响应示例**:
```json
{
    "status": "healthy",
    "version": "3.0.0"
}
```

### 第二步: 第一次预测调用

使用最简单的4个参数进行预测：
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

## 📝 核心参数详解

### 1. current_date (当前日期)
- **格式**: YYYY-MM-DD
- **含义**: 预测的起始日期，系统会从这个日期开始预测未来N天
- **示例**: "2024-06-01"
- **说明**: 系统会自动获取到此日期前一天为止的60天历史数据

### 2. predict_days (预测天数)
- **范围**: 1-30天
- **含义**: 从当前日期开始，预测未来多少天
- **示例**: 7 (预测未来7天)
- **推荐**: 
  - 短期预警: 1-7天
  - 中期规划: 8-15天  
  - 长期预测: 16-30天

### 3. station (监测站点)
- **可选值**: 
  - "胥湖心" (推荐模型: grud)
  - "锡东水厂" (推荐模型: grud)
  - "平台山" (推荐模型: xgboost)
  - "tuoshan" (推荐模型: grud)
  - "lanshanzui" (推荐模型: grud)
  - "五里湖心" (推荐模型: grud)
- **示例**: "胥湖心"

### 4. model_type (模型类型)
- **可选值**:
  - "grud" - GRU-D模型 (推荐，精度高)
  - "lstm" - LSTM模型 (基准模型)
  - "tcn" - TCN模型 (速度快)
  - "xgboost" - XGBoost模型 (在平台山表现最佳)
- **示例**: "grud"
- **建议**: 大部分情况下使用"grud"，平台山站点使用"xgboost"

## 🌟 实际使用场景

### 场景1: 日常监测预警 (1-7天)
**业务需求**: 环保部门需要了解胥湖心站点未来一周的蓝藻情况

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

**响应解读**:
```json
{
    "success": true,
    "data": {
        "prediction": [0.4706, 0.4192, 0.4954, 0.3341, 0.2284, 0.1289, 0.0241],
        "prediction_dates": [
            "2024-06-01", "2024-06-02", "2024-06-03", 
            "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07"
        ]
    },
    "message": "成功生成胥湖心从2024-06-01开始未来7天的模拟蓝藻密度预测"
}
```

### 场景2: 中期治理规划 (15天)
**业务需求**: 制定半月治理计划

```bash
curl -X POST 'http://localhost:8002/api/v3/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "current_date": "2024-06-01",
    "predict_days": 15,
    "station": "锡东水厂",
    "model_type": "grud"
  }'
```

### 场景3: 长期预测分析 (30天)
**业务需求**: 月度蓝藻风险评估

```bash
curl -X POST 'http://localhost:8002/api/v3/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "current_date": "2024-06-01",
    "predict_days": 30,
    "station": "平台山",
    "model_type": "xgboost"
  }'
```

### 场景4: 多站点对比分析
**业务需求**: 同时监测多个关键站点

```bash
curl -X POST 'http://localhost:8002/api/v3/batch-predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "requests": [
      {
        "current_date": "2024-06-01",
        "predict_days": 7,
        "station": "胥湖心",
        "model_type": "grud"
      },
      {
        "current_date": "2024-06-01",
        "predict_days": 7,
        "station": "锡东水厂",
        "model_type": "grud"
      },
      {
        "current_date": "2024-06-01",
        "predict_days": 7,
        "station": "平台山",
        "model_type": "xgboost"
      }
    ],
    "parallel_execution": true
  }'
```

## 🔧 高级功能

### 请求验证
在发送预测请求前，可以先验证请求的可行性：

```bash
curl -X POST 'http://localhost:8002/api/v3/validate' \
  -H 'Content-Type: application/json' \
  -d '{
    "current_date": "2024-06-01",
    "predict_days": 7,
    "station": "胥湖心",
    "model_type": "grud"
  }'
```

**响应信息包括**:
- 请求是否有效
- 数据可用性信息
- 模型推荐建议
- 自动计算的历史数据截止日期

### 获取配置信息

查看支持的站点列表：
```bash
curl http://localhost:8002/api/stations
```

查看支持的模型类型：
```bash
curl http://localhost:8002/api/models
```

查看API输入格式说明：
```bash
curl http://localhost:8002/api/v3/input-schema
```

## 🎯 最佳实践建议

### 1. 站点和模型选择
| 站点名称 | 推荐模型 | 改善率 | 适用场景 |
|---------|----------|--------|----------|
| 胥湖心 | grud | 77.4% | 重污染区域监测 |
| 锡东水厂 | grud | 55.1% | 水源地保护 |
| 平台山 | xgboost | 100% | 背景区域监测 |
| tuoshan | grud | 79.8% | 边界条件监测 |
| lanshanzui | grud | 28.8% | 入湖口监测 |
| 五里湖心 | grud | 42.9% | 湖心区域监测 |

### 2. 预测天数选择
- **1-3天**: 应急响应，精度最高
- **4-7天**: 日常监测，平衡精度和实用性
- **8-15天**: 中期规划，趋势分析
- **16-30天**: 长期预测，季节性规律

### 3. 时间设置建议
- **current_date**: 建议设置为当前日期或最新数据日期
- **历史数据**: 系统自动获取current_date前60天的数据
- **预测范围**: 从current_date开始的连续N天

## ⚠️ 常见问题与解决

### Q1: 预测天数超过30天怎么办？
**错误信息**: "Input should be less than or equal to 30"
**解决方案**: 将predict_days设置为1-30之间的值

### Q2: 站点名称写错了怎么办？
**错误信息**: "不支持的监测站点"
**解决方案**: 检查station参数，使用准确的站点名称

### Q3: 模型类型不存在怎么办？
**错误信息**: "不支持的模型类型"
**解决方案**: 使用grud、lstm、tcn、xgboost中的一个

### Q4: 日期格式不正确怎么办？
**错误信息**: "日期格式不正确，应为 YYYY-MM-DD"
**解决方案**: 使用标准格式，如"2024-06-01"

### Q5: 服务无法连接怎么办？
**错误信息**: 连接超时或拒绝连接
**解决方案**: 
1. 检查服务是否启动: `python main_v3.py`
2. 检查端口是否正确: 8002
3. 检查防火墙设置

## 🖥️ 浏览器中使用

### 访问API文档
在浏览器中打开: http://localhost:8002/docs

这里提供了完整的交互式API文档，可以直接在网页中测试所有接口。

### 使用Swagger界面
1. 打开API文档页面
2. 找到`/api/v3/predict`接口
3. 点击"Try it out"
4. 填入4个参数
5. 点击"Execute"执行

## 📊 编程语言示例

### Python示例
```python
import requests

def predict_algae():
    url = "http://localhost:8002/api/v3/predict"
    
    data = {
        "current_date": "2024-06-01",
        "predict_days": 7,
        "station": "胥湖心",
        "model_type": "grud"
    }
    
    response = requests.post(url, json=data)
    result = response.json()
    
    if result['success']:
        predictions = result['data']['prediction']
        dates = result['data']['prediction_dates']
        
        print("预测结果:")
        for i, (date, pred) in enumerate(zip(dates, predictions)):
            print(f"{date}: {pred:.4f}")
    else:
        print("预测失败:", result.get('detail', '未知错误'))

# 调用函数
predict_algae()
```

### JavaScript示例
```javascript
async function predictAlgae() {
    const url = 'http://localhost:8002/api/v3/predict';
    
    const data = {
        current_date: '2024-06-01',
        predict_days: 7,
        station: '胥湖心',
        model_type: 'grud'
    };
    
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            console.log('预测结果:', result.data.prediction);
            console.log('预测日期:', result.data.prediction_dates);
        } else {
            console.error('预测失败:', result.detail);
        }
    } catch (error) {
        console.error('请求异常:', error);
    }
}

// 调用函数
predictAlgae();
```

## 🎉 总结

V3版本的蓝藻预测系统API实现了：

### ✅ 极简易用
- 从720个参数减少到4个参数
- 从专家级工具变成人人可用的服务
- 无需技术背景即可使用

### ✅ 功能强大
- 支持1-30天任意长度预测
- 覆盖太湖流域6个关键监测站点
- 提供4种先进的机器学习模型

### ✅ 智能自动
- 自动获取历史数据
- 自动预处理和标准化
- 自动模型推理和结果优化

### ✅ 生产就绪
- 毫秒级响应时间
- 支持高并发批量处理
- 完善的错误处理和验证

V3版本真正实现了"用最少的输入，获得最准确的预测"的设计目标，为蓝藻水华的预防和治理提供了科学可靠、简单易用的预测工具！

---
**文档创建日期**: 2024-08-17  
**文档版本**: 1.0  
**适用版本**: 蓝藻预测系统API V3.0.0  
**更新记录**: 初版发布
