#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双置换注意力神经网络(DoublePANN)实验验证 - ECG5000数据集
------------------------------------------------------
本脚本实现了DoublePANN架构在ECG5000数据集上的验证实验。
包括:
- 基准模型(LSTM, Transformer, CNN-LSTM)评估
- DoublePANN模型(SinglePANN, DoublePANN-Basic, 完整DoublePANN)评估
- 模型性能比较
- 消融研究
- 置换可视化
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

# 导入项目模块
from models.models import (
    LSTMBaseline, TransformerBaseline, CNNLSTMBaseline,
    SinglePANNLSTM, DoublePANNBasic, DoublePANN
)
from data import prepare_ecg5000_data
from utils import (
    train_model, evaluate_model, plot_results, 
    visualize_permutations, plot_ablation_results
)
from config import Config

def set_random_seed(seed):
    """设置随机种子确保可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子已设置为: {seed}")

def run_experiments(X_train, y_train, X_test, y_test, device, config, save_dir):
    """
    运行主要实验比较
    """
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    # 获取输入维度
    input_dim = X_train.shape[1]  # 通道数/特征维度
    seq_length = X_train.shape[2]  # 序列长度
    num_classes = y_train.shape[1]  # 类别数
    
    print(f"输入维度: {input_dim}, 序列长度: {seq_length}, 类别数: {num_classes}")
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 实验结果记录
    results = {}
    
    # 1. 基准模型评估
    print("\n==========================")
    print("评估LSTM基准模型...")
    lstm_model = LSTMBaseline(input_dim, num_classes=num_classes).to(device)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=config.LEARNING_RATE)
    
    lstm_history = train_model(lstm_model, train_loader, criterion, lstm_optimizer, 
                              epochs=config.MAIN_EPOCHS, device=device, model_name="LSTM")
    
    lstm_loss, lstm_acc = evaluate_model(lstm_model, test_loader, criterion, device)
    results['LSTM'] = {'acc': lstm_acc, 'loss': lstm_loss, 'history': lstm_history}
    
    print("\n==========================")
    print("评估Transformer基准模型...")
    transformer_model = TransformerBaseline(input_dim, seq_length, num_classes).to(device)
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=config.LEARNING_RATE)
    
    transformer_history = train_model(transformer_model, train_loader, criterion, transformer_optimizer, 
                                     epochs=config.MAIN_EPOCHS, device=device, model_name="Transformer")
    
    transformer_loss, transformer_acc = evaluate_model(transformer_model, test_loader, criterion, device)
    results['Transformer'] = {'acc': transformer_acc, 'loss': transformer_loss, 'history': transformer_history}
    
    print("\n==========================")
    print("评估CNN-LSTM基准模型...")
    cnn_lstm_model = CNNLSTMBaseline(input_dim, seq_length, num_classes).to(device)
    cnn_lstm_optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=config.LEARNING_RATE)
    
    cnn_lstm_history = train_model(cnn_lstm_model, train_loader, criterion, cnn_lstm_optimizer, 
                                  epochs=config.MAIN_EPOCHS, device=device, model_name="CNN-LSTM")
    
    cnn_lstm_loss, cnn_lstm_acc = evaluate_model(cnn_lstm_model, test_loader, criterion, device)
    results['CNN-LSTM'] = {'acc': cnn_lstm_acc, 'loss': cnn_lstm_loss, 'history': cnn_lstm_history}
    
    # 2. 渐进式DoublePANN评估
    print("\n==========================")
    print("评估SinglePANN-LSTM模型...")
    single_pann_model = SinglePANNLSTM(input_dim, seq_length, num_classes).to(device)
    single_pann_optimizer = optim.Adam(single_pann_model.parameters(), lr=config.LEARNING_RATE)
    
    single_pann_history = train_model(single_pann_model, train_loader, criterion, single_pann_optimizer, 
                                     epochs=config.MAIN_EPOCHS, device=device, model_name="SinglePANN-LSTM")
    
    single_pann_loss, single_pann_acc = evaluate_model(single_pann_model, test_loader, criterion, device)
    results['SinglePANN-LSTM'] = {'acc': single_pann_acc, 'loss': single_pann_loss, 'history': single_pann_history}
    
    print("\n==========================")
    print("评估DoublePANN-Basic模型...")
    double_pann_basic_model = DoublePANNBasic(
        input_dim=input_dim,
        seq_length=seq_length,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=num_classes,
        num_permutations=config.NUM_PERMUTATIONS,
        temperature=config.TEMPERATURE
    ).to(device)
    double_pann_basic_optimizer = optim.Adam(double_pann_basic_model.parameters(), lr=config.LEARNING_RATE)
    
    double_pann_basic_history = train_model(double_pann_basic_model, train_loader, criterion, 
                                           double_pann_basic_optimizer, epochs=config.MAIN_EPOCHS, 
                                           device=device, model_name="DoublePANN-Basic")
    
    double_pann_basic_loss, double_pann_basic_acc = evaluate_model(double_pann_basic_model, test_loader, criterion, device)
    results['DoublePANN-Basic'] = {'acc': double_pann_basic_acc, 'loss': double_pann_basic_loss, 'history': double_pann_basic_history}
    
    print("\n==========================")
    print("评估完整DoublePANN模型...")
    full_double_pann_model = DoublePANN(
        input_dim=input_dim,
        seq_length=seq_length,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=num_classes,
        num_permutations=config.NUM_PERMUTATIONS,
        num_heads=config.NUM_HEADS,
        temperature=config.TEMPERATURE
    ).to(device)
    full_double_pann_optimizer = optim.Adam(full_double_pann_model.parameters(), lr=config.LEARNING_RATE)
    
    full_double_pann_history = train_model(full_double_pann_model, train_loader, criterion, 
                                          full_double_pann_optimizer, epochs=config.MAIN_EPOCHS, 
                                          device=device, model_name="Full-DoublePANN")
    
    full_double_pann_loss, full_double_pann_acc = evaluate_model(full_double_pann_model, test_loader, criterion, device)
    results['Full-DoublePANN'] = {'acc': full_double_pann_acc, 'loss': full_double_pann_loss, 'history': full_double_pann_history}
    
    # 输出结果摘要
    print("\n==========================")
    print("结果摘要:")
    for model_name, metrics in results.items():
        print(f"{model_name}: 准确率 = {metrics['acc']:.2f}%, 损失 = {metrics['loss']:.4f}")
    
    # 可视化结果
    plot_results(results, save_dir)
    
    # 找出性能最佳的模型
    best_model_name = max(results.keys(), key=lambda k: results[k]['acc'])
    best_model = None
    
    if best_model_name == 'LSTM':
        best_model = lstm_model
    elif best_model_name == 'Transformer':
        best_model = transformer_model
    elif best_model_name == 'CNN-LSTM':
        best_model = cnn_lstm_model
    elif best_model_name == 'SinglePANN-LSTM':
        best_model = single_pann_model
    elif best_model_name == 'DoublePANN-Basic':
        best_model = double_pann_basic_model
    elif best_model_name == 'Full-DoublePANN':
        best_model = full_double_pann_model
    
    print(f"性能最佳的模型是: {best_model_name}, 准确率: {results[best_model_name]['acc']:.2f}%")
    
    # 如果最佳模型是DoublePANN，则可视化置换矩阵
    if isinstance(best_model, DoublePANN):
        visualize_permutations(best_model, X_test, device, save_dir)
    
    return results, best_model

def ablation_study(X_train, y_train, X_test, y_test, device, config, save_dir):
    """
    执行消融研究
    """
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    # 获取输入维度
    input_dim = X_train.shape[1]
    seq_length = X_train.shape[2]
    num_classes = y_train.shape[1]
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 实验结果记录
    ablation_results = {}
    
    print("\n==========================")
    print("开始消融研究...")
    
    # 1. 置换数量变化
    for num_perms in config.PERM_COUNTS:
        print(f"\n测试置换数量 = {num_perms}")
        model = DoublePANN(
            input_dim=input_dim,
            seq_length=seq_length,
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=num_classes,
            num_permutations=num_perms,
            num_heads=config.NUM_HEADS,
            temperature=config.TEMPERATURE
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        history = train_model(model, train_loader, criterion, optimizer, 
                             epochs=config.ABLATION_EPOCHS, device=device, 
                             model_name=f"DoublePANN-Perms{num_perms}")
        
        loss, acc = evaluate_model(model, test_loader, criterion, device)
        ablation_results[f'Perms_{num_perms}'] = {'acc': acc, 'loss': loss}
        print(f"置换数量 = {num_perms}的模型测试准确率: {acc:.2f}%")
    
    # 2. 温度参数变化
    for temp in config.TEMPERATURES:
        print(f"\n测试温度参数 = {temp}")
        model = DoublePANN(
            input_dim=input_dim,
            seq_length=seq_length,
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=num_classes,
            num_permutations=config.NUM_PERMUTATIONS,
            num_heads=config.NUM_HEADS,
            temperature=temp
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        history = train_model(model, train_loader, criterion, optimizer, 
                             epochs=config.ABLATION_EPOCHS, device=device, 
                             model_name=f"DoublePANN-Temp{temp}")
        
        loss, acc = evaluate_model(model, test_loader, criterion, device)
        ablation_results[f'Temp_{temp}'] = {'acc': acc, 'loss': loss}
        print(f"温度参数 = {temp}的模型测试准确率: {acc:.2f}%")
    
    # 3. 关键组件移除
    # 3.1 没有注意力融合的版本
    print(f"\n测试没有注意力融合的模型")
    model_no_attention = DoublePANNBasic(
        input_dim=input_dim,
        seq_length=seq_length,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=num_classes,
        num_permutations=config.NUM_PERMUTATIONS,
        temperature=config.TEMPERATURE
    ).to(device)
    
    optimizer = optim.Adam(model_no_attention.parameters(), lr=config.LEARNING_RATE)
    history = train_model(model_no_attention, train_loader, criterion, optimizer, 
                         epochs=config.ABLATION_EPOCHS, device=device, model_name="DoublePANN-NoAttention")
    
    loss, acc = evaluate_model(model_no_attention, test_loader, criterion, device)
    ablation_results['No_Attention'] = {'acc': acc, 'loss': loss}
    print(f"没有注意力融合的模型测试准确率: {acc:.2f}%")
    
    # 可视化消融研究结果
    plot_ablation_results(ablation_results, config.PERM_COUNTS, config.TEMPERATURES, save_dir)
    
    # 打印消融研究结果摘要
    print("\n==========================")
    print("消融研究结果摘要:")
    # 置换数量影响
    print("\n置换数量影响:")
    for num_perms in config.PERM_COUNTS:
        key = f'Perms_{num_perms}'
        print(f"  置换数量 = {num_perms}: 准确率 = {ablation_results[key]['acc']:.2f}%, 损失 = {ablation_results[key]['loss']:.4f}")
    
    # 温度参数影响
    print("\n温度参数影响:")
    for temp in config.TEMPERATURES:
        key = f'Temp_{temp}'
        print(f"  温度参数 = {temp}: 准确率 = {ablation_results[key]['acc']:.2f}%, 损失 = {ablation_results[key]['loss']:.4f}")
    
    # 注意力融合影响
    print("\n注意力融合影响:")
    print(f"  有注意力融合: 准确率 = {ablation_results['Perms_3']['acc']:.2f}%, 损失 = {ablation_results['Perms_3']['loss']:.4f}")
    print(f"  无注意力融合: 准确率 = {ablation_results['No_Attention']['acc']:.2f}%, 损失 = {ablation_results['No_Attention']['loss']:.4f}")
    
    return ablation_results

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='DoublePANN在ECG5000数据集上的实验')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, 
                        help='批量大小')
    parser.add_argument('--epochs', type=int, default=Config.MAIN_EPOCHS, 
                        help='主实验训练轮数')
    parser.add_argument('--ablation_epochs', type=int, default=Config.ABLATION_EPOCHS, 
                        help='消融研究训练轮数')
    parser.add_argument('--seed', type=int, default=Config.RANDOM_SEED, 
                        help='随机种子')
    parser.add_argument('--run_ablation', action='store_true', 
                        help='是否运行消融研究')
    parser.add_argument('--results_dir', type=str, default=Config.RESULTS_DIR, 
                        help='结果保存目录')
    
    args = parser.parse_args()
    
    # 创建时间戳子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.results_dir, f"ecg5000_{timestamp}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("==========================")
    print("双置换注意力神经网络(DoublePANN)实验验证")
    print("==========================")
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建配置对象并更新参数
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.MAIN_EPOCHS = args.epochs
    config.ABLATION_EPOCHS = args.ablation_epochs
    
    # 准备数据
    print(f"\n加载ECG5000数据集...")
    X_train, y_train, X_test, y_test, y_train_raw, y_test_raw = prepare_ecg5000_data(
        data_dir=config.DATA_DIR, device=device
    )
    
    # 运行主要实验
    print(f"\n==========================")
    print(f"开始运行主要实验比较(共{config.MAIN_EPOCHS}轮)...")
    results, best_model = run_experiments(
        X_train, y_train, X_test, y_test, 
        device, config, save_dir
    )
    
    # 运行消融研究(如果需要)
    if args.run_ablation:
        print(f"\n==========================")
        print(f"开始消融研究(共{config.ABLATION_EPOCHS}轮)...")
        ablation_results = ablation_study(
            X_train, y_train, X_test, y_test, 
            device, config, save_dir
        )
    
    print("\n==========================")
    print(f"实验完成！结果保存在'{save_dir}'目录中。")

if __name__ == "__main__":
    main() 