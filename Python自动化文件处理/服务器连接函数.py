# -*- coding: utf-8 -*-
# @time:2024-08-18:21:26
# @IDE:PyCharm
import pandas as pd
import os
from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient

# 1. 读取 CSV 文件成数据帧
df = pd.read_csv('data.csv')  # 请将 'data.csv' 替换为你的 CSV 文件路径

# 2. 设置远程服务器信息和目标路径
remote_host = 'remote_server_ip'  # 远程服务器的 IP 地址
username = 'your_username'  # 远程服务器的用户名
password = 'your_password'  # 远程服务器的密码
source_base_folder = '/remote/path/to/source/folders'  # 远程服务器上的源文件夹路径
destination_folder = '/local/path/to/destination/folder'  # 本地或目标服务器上的目标文件夹路径

# 3. 连接到远程服务器
ssh = SSHClient()
ssh.set_missing_host_key_policy(AutoAddPolicy())
ssh.connect(remote_host, username=username, password=password)

# 4. 使用 SCP 进行文件传输
with SCPClient(ssh.get_transport()) as scp:
    # 遍历数据帧，根据编码列与SN列筛选文件夹与文件
    for index, row in df.iterrows():
        code = row['编码']  # 假设编码列名为 '编码'
        sn = row['SN']  # 假设 SN 列名为 'SN'

        # 构造远程匹配的文件夹路径
        remote_folder_path = os.path.join(source_base_folder, code)

        # 执行命令检查文件夹是否存在
        stdin, stdout, stderr = ssh.exec_command(f'if [ -d "{remote_folder_path}" ]; then echo "exists"; fi')
        folder_exists = stdout.read().decode().strip()

        if folder_exists == 'exists':
            # 遍历远程文件夹下的文件
            stdin, stdout, stderr = ssh.exec_command(f'ls {remote_folder_path}')
            files = stdout.read().decode().splitlines()

            for file in files:
                if sn in file and file.endswith('.csv.gz'):  # 匹配文件名包含 SN 且为 .csv.gz 文件
                    remote_file_path = os.path.join(remote_folder_path, file)
                    local_destination_path = os.path.join(destination_folder, code)

                    # 确保本地目标文件夹存在
                    os.makedirs(local_destination_path, exist_ok=True)

                    # 复制文件到本地目标文件夹
                    scp.get(remote_file_path, os.path.join(local_destination_path, file))

                    # 打印复制完成的文件夹与文件名
                    print(f"已复制文件夹: {code}, 文件名: {file}")
        else:
            print(f"远程文件夹不存在: {remote_folder_path}")

print("所有匹配的文件已复制完成。")

# 5. 关闭连接
ssh.close()
