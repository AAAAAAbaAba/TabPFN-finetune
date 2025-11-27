#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从logs/Basic_finetune的27个文件夹中的txt文件提取4个数据集的最佳epoch
并生成Excel和CSV文件
"""

import os
import re
from pathlib import Path
import csv

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("警告: 未安装pandas，将只生成CSV文件")

# 数据集名称
datasets = ['4A', '4B', 'zhengqi', 'guangfu']

# 基础路径
base_dir = Path('/home/fit/zhangcs/WORK/chenkq/project/TabPFN-finetune/logs/Deeper_finetune_znorm')

# 存储结果
results = {}

# 遍历所有ID文件夹
for id_num in range(1, 28):
    id_folder = base_dir / f'ID_{id_num}'
    
    if not id_folder.exists():
        print(f"警告: {id_folder} 不存在")
        continue
    
    # 查找txt文件
    txt_files = list(id_folder.glob('*.txt'))
    if not txt_files:
        print(f"警告: {id_folder} 中没有找到txt文件")
        continue
    
    # 通常只有一个txt文件，取第一个
    txt_file = txt_files[0]
    
    print(f"\n处理 {txt_file.name}...")
    
    # 读取文件内容
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 为每个数据集存储最佳epoch信息
    best_epochs = {ds: None for ds in datasets}
    
    # 解析每一行
    current_epoch = None
    for i, line in enumerate(lines):
        # 匹配epoch行
        epoch_match = re.search(r'📊 Epoch (\d+) Evaluation', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        # 匹配Initial Evaluation
        if '📊 Initial Evaluation' in line:
            current_epoch = 'Initial'
        
        # 检查每个数据集的最佳结果
        for ds in datasets:
            # 匹配数据集行，包含🌟 BEST标记
            pattern = rf'\s+{ds}\s+\|.*\|.*🌟 BEST'
            if re.search(pattern, line):
                # 解析数据行，提取各个指标
                # 格式: 数据集名 | Test MSE: X, Test MAE: Y, Test R2: Z, Test max_AE: W, Test std_ERR: V | 🌟 BEST
                metrics = {}
                mse_match = re.search(r'Test MSE:\s+([\d.]+)', line)
                mae_match = re.search(r'Test MAE:\s+([\d.]+)', line)
                r2_match = re.search(r'Test R2:\s+([\d.]+)', line)
                max_ae_match = re.search(r'Test max_AE:\s+([\d.]+)', line)
                std_err_match = re.search(r'Test std_ERR:\s+([\d.]+)', line)
                
                if mse_match:
                    metrics['MSE'] = float(mse_match.group(1))
                if mae_match:
                    metrics['MAE'] = float(mae_match.group(1))
                if r2_match:
                    metrics['R2'] = float(r2_match.group(1))
                if max_ae_match:
                    metrics['max_AE'] = float(max_ae_match.group(1))
                if std_err_match:
                    metrics['std_ERR'] = float(std_err_match.group(1))
                
                # 提取完整行内容
                best_epochs[ds] = {
                    'epoch': current_epoch,
                    'line': line.strip(),
                    'line_num': i + 1,
                    'metrics': metrics
                }
    
    results[id_num] = best_epochs

# 展示结果
print("\n" + "="*100)
print("最佳Epoch结果汇总")
print("="*100)

# 按数据集分组展示
for ds in datasets:
    print(f"\n{'='*120}")
    print(f"数据集: {ds}")
    print(f"{'='*120}")
    print(f"{'ID':<6} {'Epoch':<12} {'完整数据行'}")
    print("-" * 120)
    
    for id_num in sorted(results.keys()):
        if results[id_num][ds] is not None:
            epoch = results[id_num][ds]['epoch']
            line = results[id_num][ds]['line']
            print(f"{id_num:<6} {str(epoch):<12} {line}")
        else:
            print(f"{id_num:<6} {'未找到':<12} {'-'}")

print("\n" + "="*100)
print("完成！")
print("="*100)

# 生成CSV和Excel文件
print("\n正在生成CSV和Excel文件...")

# 准备数据用于导出
export_data = []

for id_num in sorted(results.keys()):
    row = {'ID': id_num}
    
    for ds in datasets:
        if results[id_num][ds] is not None:
            epoch = results[id_num][ds]['epoch']
            metrics = results[id_num][ds].get('metrics', {})
            
            row[f'{ds}_Epoch'] = epoch
            row[f'{ds}_MSE'] = metrics.get('MSE', '')
            row[f'{ds}_MAE'] = metrics.get('MAE', '')
            row[f'{ds}_R2'] = metrics.get('R2', '')
            row[f'{ds}_max_AE'] = metrics.get('max_AE', '')
            row[f'{ds}_std_ERR'] = metrics.get('std_ERR', '')
        else:
            row[f'{ds}_Epoch'] = ''
            row[f'{ds}_MSE'] = ''
            row[f'{ds}_MAE'] = ''
            row[f'{ds}_R2'] = ''
            row[f'{ds}_max_AE'] = ''
            row[f'{ds}_std_ERR'] = ''
    
    export_data.append(row)

# 生成CSV文件
csv_file = base_dir.parent / 'best_epochs_data.csv'
fieldnames = ['ID']
for ds in datasets:
    fieldnames.extend([f'{ds}_Epoch', f'{ds}_MSE', f'{ds}_MAE', f'{ds}_R2', f'{ds}_max_AE', f'{ds}_std_ERR'])

with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:  # utf-8-sig for Excel compatibility
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(export_data)

print(f"✓ CSV文件已生成: {csv_file}")

# 生成Excel文件（如果pandas可用）
if HAS_PANDAS:
    try:
        excel_file = base_dir.parent / 'best_epochs_data.xlsx'
        df = pd.DataFrame(export_data)
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"✓ Excel文件已生成: {excel_file}")
    except Exception as e:
        print(f"⚠ Excel文件生成失败: {e}")
        print("  请确保已安装: pip install pandas openpyxl")

# 生成便于复制粘贴的格式（制表符分隔）
tsv_file = base_dir.parent / 'best_epochs_data.tsv'
with open(tsv_file, 'w', encoding='utf-8') as f:
    # 写入表头
    f.write('\t'.join(fieldnames) + '\n')
    # 写入数据
    for row in export_data:
        values = [str(row.get(field, '')) for field in fieldnames]
        f.write('\t'.join(values) + '\n')

print(f"✓ TSV文件已生成（可直接复制到Excel）: {tsv_file}")

print("\n所有文件已生成完成！")

