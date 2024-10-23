# -*- coding: utf-8 -*-
# @time:2024/8/17:下午11:42
# @IDE:PyCharm
import os

# 获取当前工作目录
current_path = os.getcwd()

# 列出当前目录下的所有文件和文件夹
items = os.listdir(current_path)

# 区分并输出文件和文件夹
for item in items:
    item_path = os.path.join(current_path, item)
    if os.path.isdir(item_path):
        print(f"文件夹: {item}")
    elif os.path.isfile(item_path):
        print(f"文件: {item}")

