import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_fake_data(start_date_str, end_date_str, num_features=12):
    """
    生成指定日期范围内，包含日期和多个特征的假数据。
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    
    data = {'date': dates}
    for i in range(1, num_features + 1):
        # 假设特征值在合理范围内波动
        # 你可以根据实际数据范围调整这些值
        data[f'feature_{i}'] = np.random.uniform(low=5.0 + i, high=30.0 + i, size=len(dates))
        # 模拟一些数据的整数特性，例如水质指标可能是整数
        if i % 3 == 0:
            data[f'feature_{i}'] = np.round(data[f'feature_{i}'])

    df = pd.DataFrame(data)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d') # 格式化日期
    return df

if __name__ == "__main__":
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)

    stations = ["胥湖心", "锡东水厂", "平台山", "拖山", "兰山嘴", "五里湖心"]
    
    # 假设现有数据截止到2024-07-17，所以从2024-07-18开始生成假数据到2024-08-17
    # 你可以根据实际情况调整这个开始日期
    start_date_for_fake = "2024-07-18"
    end_date_for_fake = "2024-08-17" # 今天的日期

    print(f"正在为以下站点生成从 {start_date_for_fake} 到 {end_date_for_fake} 的假数据：")
    for station in stations:
        print(f"- {station}")
        fake_df = generate_fake_data(start_date_for_fake, end_date_for_fake)
        file_path = os.path.join(output_dir, f'{station}_fake.csv')
        fake_df.to_csv(file_path, index=False)
        print(f"  已生成并保存到: {file_path}")

    print("\n假数据生成完成。请检查 'data/' 目录。")
