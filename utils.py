#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函数模块
-----------
包含训练、评估和可视化功能
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# 配置中文字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from models.models import DoublePANN

#################################
# 训练和评估函数
#################################

def train_model(model, train_loader, criterion, optimizer, epochs, device, model_name="模型"):
    """
    训练PyTorch模型
    
    Args:
        model: PyTorch模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        epochs: 训练轮数
        device: 训练设备
        model_name: 模型名称(用于进度条显示)
        
    Returns:
        训练历史记录(损失和准确率)
    """
    model.train()
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    print(f"开始训练 {model_name}...")
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm创建进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                           ncols=100, position=0, leave=True)
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(targets, dim=1))
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, targets_idx = torch.max(targets, 1)
            total += targets.size(0)
            correct += (predicted == targets_idx).sum().item()
            
            # 更新进度条信息
            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'acc': 100 * correct / total
            })
        
        # 计算周期平均值
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # 保存历史记录
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        
        # 打印周期结果
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2f}%")
    
    return history

def evaluate_model(model, test_loader, criterion, device):
    """
    评估PyTorch模型
    
    Args:
        model: PyTorch模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 评估设备
        
    Returns:
        损失和准确率
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="评估", ncols=100)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(targets, dim=1))
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, targets_idx = torch.max(targets, 1)
            total += targets.size(0)
            correct += (predicted == targets_idx).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'acc': 100 * correct / total
            })
    
    # 计算总体评估结果
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    print(f"评估结果 - loss: {avg_loss:.4f}, accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy

#################################
# 可视化和分析函数
#################################

def plot_results(results, save_dir='results'):
    """
    可视化实验结果比较
    
    Args:
        results: 包含各模型评估结果的字典
        save_dir: 结果图保存目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. 准确率比较
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    accuracies = [results[m]['acc'] for m in models]
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, accuracies)
    plt.title('测试集准确率比较')
    plt.ylabel('准确率 (%)')
    plt.xticks(rotation=45)
    
    # 在柱状图上标注数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{accuracies[i]:.2f}%',
                ha='center', va='bottom', rotation=0)
    
    # 2. 损失比较
    losses = [results[m]['loss'] for m in models]
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(models, losses)
    plt.title('测试集损失比较')
    plt.ylabel('损失')
    plt.xticks(rotation=45)
    
    # 在柱状图上标注数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{losses[i]:.4f}',
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
    print(f"保存模型比较图到: {os.path.join(save_dir, 'model_comparison.png')}")
    
    # 3. 训练历史
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    for model in models:
        if 'history' in results[model] and 'loss' in results[model]['history']:
            plt.plot(results[model]['history']['loss'], label=model)
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for model in models:
        if 'history' in results[model] and 'accuracy' in results[model]['history']:
            plt.plot(results[model]['history']['accuracy'], label=model)
    plt.title('训练准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    print(f"保存训练历史图到: {os.path.join(save_dir, 'training_history.png')}")

def visualize_permutations(model, inputs, device, save_dir='results'):
    """
    可视化模型在输入样本上生成的置换矩阵
    
    Args:
        model: DoublePANN模型
        inputs: 输入样本
        device: 使用的设备
        save_dir: 结果图保存目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.eval()
    
    with torch.no_grad():
        # 选择第一个样本
        x = inputs[0:1].to(device)
        
        # 准备输入数据
        x_t = x.transpose(1, 2)  # [1, seq_len, feature_dim]
        
        # 提取第一个置换矩阵
        perm1 = model.generate_permutation(x_t, model.perm_generator1[0], False)
        
        # 应用第一次置换
        x_perm1 = torch.bmm(perm1, x_t)
        
        # 嵌入
        embedded = model.embedding(x_perm1)
        
        # 提取第二个置换矩阵
        perm2 = model.generate_permutation(embedded, model.perm_generator2[0], False)
    
    # 将张量转移到CPU并转换为NumPy数组进行可视化
    perm1_np = perm1[0].cpu().numpy()
    perm2_np = perm2[0].cpu().numpy()
    x_np = x[0, 0].cpu().numpy()
    x_perm1_np = x_perm1[0, :, 0].cpu().numpy()
    
    # 可视化
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始序列
    ax[0].plot(x_np)
    ax[0].set_title('原始序列')
    
    # 第一次置换矩阵
    im1 = ax[1].imshow(perm1_np, cmap='viridis')
    ax[1].set_title('第一次置换矩阵')
    plt.colorbar(im1, ax=ax[1])
    
    # 第二次置换矩阵
    im2 = ax[2].imshow(perm2_np, cmap='viridis')
    ax[2].set_title('第二次置换矩阵')
    plt.colorbar(im2, ax=ax[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'permutation_visualization.png'))
    print(f"保存置换矩阵可视化到: {os.path.join(save_dir, 'permutation_visualization.png')}")
    
    # 可视化置换前后的序列
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_np)
    plt.title('原始序列')
    
    plt.subplot(1, 2, 2)
    plt.plot(x_perm1_np)
    plt.title('第一次置换后的序列')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'permutation_effect.png'))
    print(f"保存置换效果可视化到: {os.path.join(save_dir, 'permutation_effect.png')}")

def plot_ablation_results(ablation_results, perm_counts, temperatures, save_dir='results'):
    """
    可视化消融研究结果
    
    Args:
        ablation_results: 消融研究结果字典
        perm_counts: 测试的置换数量列表
        temperatures: 测试的温度参数列表
        save_dir: 结果保存目录
    """
    plt.figure(figsize=(14, 6))
    
    # 1. 置换数量影响
    plt.subplot(1, 2, 1)
    perm_keys = [f'Perms_{p}' for p in perm_counts]
    perm_accs = [ablation_results[k]['acc'] for k in perm_keys]
    plt.plot(perm_counts, perm_accs, 'o-')
    plt.xlabel('置换数量')
    plt.ylabel('测试准确率 (%)')
    plt.title('置换数量对模型性能的影响')
    
    # 2. 温度参数影响
    plt.subplot(1, 2, 2)
    temp_keys = [f'Temp_{t}' for t in temperatures]
    temp_accs = [ablation_results[k]['acc'] for k in temp_keys]
    plt.plot(temperatures, temp_accs, 'o-')
    plt.xlabel('温度参数')
    plt.ylabel('测试准确率 (%)')
    plt.title('温度参数对模型性能的影响')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_study.png'))
    print(f"保存消融研究结果到: {os.path.join(save_dir, 'ablation_study.png')}") 