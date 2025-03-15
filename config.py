#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实验配置
-------
包含ECG5000实验的所有参数配置
"""

class Config:
    # 随机种子
    RANDOM_SEED = 42
    
    # 数据集配置
    DATA_DIR = "data/dataset"
    
    # 训练配置
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    MAIN_EPOCHS = 30  # 主实验训练轮数
    ABLATION_EPOCHS = 15  # 消融研究训练轮数
    
    # 模型配置
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    NUM_PERMUTATIONS = 3
    NUM_HEADS = 4
    TEMPERATURE = 0.1
    SINKHORN_ITERATIONS = 5
    DROPOUT = 0.1
    
    # 结果目录
    RESULTS_DIR = "results"
    
    # 消融研究配置
    PERM_COUNTS = [1, 2, 3, 5]  # 测试不同置换数量
    TEMPERATURES = [0.01, 0.1, 0.5, 1.0]  # 测试不同温度参数 