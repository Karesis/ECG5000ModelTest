#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ECG5000数据集加载模块
-------------------
提供ECG5000数据集的加载和预处理功能
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

def prepare_ecg5000_data(data_dir="dataset", device=None):
    """
    准备ECG5000数据集
    
    Args:
        data_dir: 数据文件所在目录
        device: 运行设备 (None则自动选择)
    
    Returns:
        X_train, y_train_cat: 训练集特征和独热编码标签
        X_test, y_test_cat: 测试集特征和独热编码标签
        y_train, y_test: 训练集和测试集原始标签
    """
    print("加载ECG5000数据集...")
    
    # 设置设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    try:
        test_data = np.loadtxt(os.path.join(data_dir, 'ECG5000_TRAIN.txt'))  # 交换：原TRAIN作为测试集
        train_data = np.loadtxt(os.path.join(data_dir, 'ECG5000_TEST.txt'))  # 交换：原TEST作为训练集
        print(f"注意：文件名与实际用途相反，使用ECG5000_TEST.txt作为训练集（样本更多）")
    except:
        raise FileNotFoundError(f"找不到ECG5000数据文件。请确保ECG5000_TRAIN.txt和ECG5000_TEST.txt在{data_dir}目录中。")
    
    # 分离特征和标签
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
    
    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 重塑为3D张量 [样本数, 特征, 时间步]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # 将标签转为分类格式（从1开始的类别标签需要-1）
    y_train = y_train.astype(int) - 1
    y_test = y_test.astype(int) - 1
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # 创建独热编码标签
    num_classes = len(np.unique(y_train))
    y_train_cat = F.one_hot(y_train_tensor, num_classes=num_classes).float()
    y_test_cat = F.one_hot(y_test_tensor, num_classes=num_classes).float()
    
    print(f"数据集形状: X_train: {X_train_tensor.shape}, y_train: {y_train_tensor.shape}")
    print(f"          X_test: {X_test_tensor.shape}, y_test: {y_test_tensor.shape}")
    print(f"类别数量: {num_classes}")
    
    return X_train_tensor, y_train_cat, X_test_tensor, y_test_cat, y_train_tensor, y_test_tensor 