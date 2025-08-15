
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

BASE_URL = "http://localhost:8000/api"

def test_prediction(station: str, model_type: str, predict_days: int):
    """
    测试单个预测请求
    """
    url = f"{BASE_URL}/predict"
    payload = {
        "station": station,
        "model_type": model_type,
        "predict_days": predict_days,
        "input_data": {
            "temperature": 25.5,
            "oxygen": 8.2,
            "TN": 1.5,
            "TP": 0.08
        }
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=60)
        end_time = time.time()
        
        duration = end_time - start_time
        
        if response.status_code == 200:
            return {
                "days": predict_days,
                "status": "success",
                "duration": duration,
                "response": response.json()
            }
        else:
            return {
                "days": predict_days,
                "status": "failed",
                "status_code": response.status_code,
                "error": response.text,
                "duration": duration
            }
    except requests.exceptions.RequestException as e:
        return {
            "days": predict_days,
            "status": "error",
            "error": str(e)
        }

def run_prediction_tests():
    """
    运行1到30天的预测测试
    """
    station = "胥湖心"
    model_type = "grud"
    
    print(f"🚀 开始测试: {station} - {model_type.upper()} 模型")
    print("="*50)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(test_prediction, station, model_type, days): days for days in range(1, 31)}
        
        for i, future in enumerate(as_completed(futures)):
            day = futures[future]
            result = future.result()
            results.append(result)
            
            progress = (i + 1) / 30 * 100
            print(f"  [ {progress:3.0f}% ] 测试 {day} 天预测... ", end="")
            
            if result["status"] == "success":
                print(f"✅ 成功 (耗时: {result['duration']:.2f}s)")
            else:
                print(f"❌ 失败 (状态码: {result.get('status_code', 'N/A')})")

    print("="*50)
    print("📊 测试结果分析:")
    
    # 按天数排序
    results.sort(key=lambda x: x["days"])
    
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = len(results) - success_count
    
    print(f"\n  - 成功率: {success_count / len(results):.1%}")
    print(f"  - 成功: {success_count} | 失败: {failed_count}")
    
    if success_count > 0:
        avg_duration = sum(r['duration'] for r in results if r['status'] == 'success') / success_count
        max_duration_result = max((r for r in results if r['status'] == 'success'), key=lambda x: x['duration'])
        min_duration_result = min((r for r in results if r['status'] == 'success'), key=lambda x: x['duration'])
        
        print(f"  - 平均响应时间: {avg_duration:.2f}s")
        print(f"  - 最长响应时间: {max_duration_result['duration']:.2f}s (在 {max_duration_result['days']} 天预测)")
        print(f"  - 最短响应时间: {min_duration_result['duration']:.2f}s (在 {min_duration_result['days']} 天预测)")

    if failed_count > 0:
        print("\n  - 失败详情:")
        for r in results:
            if r["status"] != "success":
                print(f"    - {r['days']} 天预测: {r['status']} - {r.get('error', '无详情')[:100]}")

    # 保存结果到文件
    output_file = "test_xuhu_prediction_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 详细测试结果已保存至: {output_file}")

if __name__ == "__main__":
    run_prediction_tests()
