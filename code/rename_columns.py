# -*- coding: utf-8 -*-
"""
@author:renshengbushexian
@time:2024/8/8:下午10:34
@email:chenzhengdong_zy@outlook.com
@IDE:PyCharm
"""
import pandas as pd

def split_column_if_not_exist(df):
    # 检查是否存在'C'和'D'字段
    if 'C' not in df.columns and 'D' not in df.columns:
        # 创建'C'和'D'字段
        df['C'] = df['A'].apply(lambda x: x if x > 0 else None)
        df['D'] = df['A'].apply(lambda x: x if x < 0 else None)
    return df

# 示例数据框
data = {
    'A': [1, -2, 3, -4, 5]
}

df = pd.DataFrame(data)

# 调用函数处理数据框
df = split_column_if_not_exist(df)

# 显示结果
print(df)
