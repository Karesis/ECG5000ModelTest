#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双置换注意力神经网络(DoublePANN)模型定义
------------------------------------------
本文件包含：
- 基准模型(LSTM, Transformer, CNN-LSTM)
- 单置换模型(SinglePANN)
- 基础双置换模型(DoublePANN-Basic)
- 完整双置换注意力神经网络(DoublePANN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#################################
# 基准模型实现
#################################

class LSTMBaseline(nn.Module):
    """
    LSTM基准模型
    """
    def __init__(self, input_size, hidden_size_1=128, hidden_size_2=64, num_classes=5):
        super(LSTMBaseline, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size_1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size_2, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x shape: [batch, feature_dim, seq_len] -> [batch, seq_len, feature_dim]
        x = x.transpose(1, 2)
        
        # LSTM层
        output1, _ = self.lstm1(x)
        output2, (h_n, _) = self.lstm2(output1)
        
        # 使用最后一个时间步的隐藏状态
        h_n = h_n.squeeze(0)
        
        # 全连接层
        x = F.relu(self.fc1(h_n))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class PositionalEncoding(nn.Module):
    """
    Transformer的位置编码
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为非模型参数
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, feature_dim]
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerBlock(nn.Module):
    """
    简化的Transformer块
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        # 转置以适应PyTorch的Multi-head Attention: [seq_len, batch_size, embed_dim]
        x_t = x.transpose(0, 1)
        
        # 自注意力
        attn_output, _ = self.attention(x_t, x_t, x_t)
        attn_output = attn_output.transpose(0, 1)  # 转回 [batch_size, seq_len, embed_dim]
        
        # 第一个残差连接和层归一化
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.ff(x)
        
        # 第二个残差连接和层归一化
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerBaseline(nn.Module):
    """
    Transformer基准模型
    """
    def __init__(self, input_size, seq_length, num_classes, head_size=64, num_heads=4, dropout=0.1):
        super(TransformerBaseline, self).__init__()
        
        # 线性投影到嵌入维度 - 确保维度是头数的倍数
        embed_dim = 64  # 使用固定维度，而不是input_size，确保可被num_heads整除
        self.embedding = nn.Linear(input_size, embed_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Transformer块
        self.transformer_block = TransformerBlock(embed_dim, num_heads, head_size*2, dropout)
        
        # 输出层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: [batch, feature_dim, seq_len] -> [batch, seq_len, feature_dim]
        x = x.transpose(1, 2)
        
        # 嵌入和位置编码
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # Transformer块
        x = self.transformer_block(x)
        
        # 全局池化
        x = x.transpose(1, 2)  # [batch, embed_dim, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch, embed_dim]
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CNNLSTMBaseline(nn.Module):
    """
    CNN-LSTM混合基准模型
    """
    def __init__(self, input_size, seq_length, num_classes):
        super(CNNLSTMBaseline, self).__init__()
        
        # CNN模块
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算CNN后的序列长度
        self.seq_len_after_cnn = seq_length // 4  # 两次池化，每次减半
        
        # LSTM模块
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        
        # 全连接层
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x shape: [batch, feature_dim(1), seq_len]
        
        # CNN层
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # [batch, channels, seq_len] -> [batch, seq_len, channels]
        x = x.transpose(1, 2)
        
        # LSTM层
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.squeeze(0)
        
        # 全连接层
        x = F.relu(self.fc1(h_n))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

#################################
# DoublePANN组件实现
#################################

class SinkhornNormalizer(nn.Module):
    """
    Sinkhorn归一化层，用于将logits转换为双随机矩阵
    """
    def __init__(self, temperature=1.0, iterations=5):
        super(SinkhornNormalizer, self).__init__()
        self.temperature = temperature
        self.iterations = iterations
        
    def forward(self, logits):
        """
        应用Sinkhorn-Knopp算法进行归一化
        
        Args:
            logits: 形状为[batch_size, seq_len, seq_len]的张量
            
        Returns:
            归一化后的双随机矩阵
        """
        # 复制输入以防止修改原始数据
        log_alpha = logits.clone()
        
        # Sinkhorn迭代
        for _ in range(self.iterations):
            # 行归一化 (对log space中的张量)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
            # 列归一化
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        
        # 应用温度参数并转换回概率空间
        return torch.exp(log_alpha / self.temperature)

class SinglePermutationLayer(nn.Module):
    """
    单置换层实现
    """
    def __init__(self, input_dim, temperature=1.0, sinkhorn_iterations=5):
        super(SinglePermutationLayer, self).__init__()
        
        # 置换生成网络
        self.perm_generator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Sinkhorn归一化器
        self.sinkhorn = SinkhornNormalizer(temperature, sinkhorn_iterations)
        
    def forward(self, x):
        """
        生成置换矩阵并应用到输入上
        
        Args:
            x: 形状为[batch_size, feature_dim, seq_len]的输入张量
            
        Returns:
            置换后的张量，形状与输入相同
        """
        # 调整张量形状以适应处理
        # x: [batch, feature_dim, seq_len] -> [batch, seq_len, feature_dim]
        x_t = x.transpose(1, 2)
        batch_size, seq_len, feature_dim = x_t.shape
        
        # 使用生成器网络产生特征
        features = self.perm_generator(x_t)  # [batch, seq_len, 64]
        
        # 计算特征之间的相似度矩阵
        similarity = torch.bmm(features, features.transpose(1, 2))  # [batch, seq_len, seq_len]
        
        # 应用Sinkhorn归一化得到双随机矩阵(软置换矩阵)
        perm_matrix = self.sinkhorn(similarity)  # [batch, seq_len, seq_len]
        
        # 应用置换到转置后的输入
        x_perm = torch.bmm(perm_matrix, x_t)  # [batch, seq_len, feature_dim]
        
        # 转回原始形状 [batch, feature_dim, seq_len]
        return x_perm.transpose(1, 2)

class SinglePANNLSTM(nn.Module):
    """
    SinglePANN-LSTM模型
    """
    def __init__(self, input_dim, seq_length, num_classes, temperature=0.1):
        super(SinglePANNLSTM, self).__init__()
        
        # 单一置换层
        self.perm_layer = SinglePermutationLayer(input_dim, temperature)
        
        # LSTM层
        self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        
        # 全连接层
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x shape: [batch, feature_dim, seq_len]
        
        # 应用置换
        x = self.perm_layer(x)
        
        # 转置为LSTM的输入形状
        x = x.transpose(1, 2)  # [batch, seq_len, feature_dim]
        
        # LSTM层
        output1, _ = self.lstm1(x)
        _, (h_n, _) = self.lstm2(output1)
        h_n = h_n.squeeze(0)
        
        # 全连接层
        x = F.relu(self.fc1(h_n))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class DoublePANNBasic(nn.Module):
    """
    基本的双置换PANN模型（不使用多头注意力融合）
    """
    def __init__(self, input_dim, seq_length, embed_dim, hidden_dim, num_classes, 
                 num_permutations=3, temperature=0.1, sinkhorn_iterations=5):
        super(DoublePANNBasic, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_permutations = num_permutations
        self.seq_length = seq_length
        
        # 嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 第一次置换（输入空间）的生成网络
        self.perm_generator1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            ) for _ in range(num_permutations)
        ])
        
        # 第二次置换（嵌入空间）的生成网络
        self.perm_generator2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_permutations)
        ])
        
        # LSTM编码器
        self.lstms = nn.ModuleList([
            nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            for _ in range(num_permutations * num_permutations)
        ])
        
        # Sinkhorn归一化器
        self.sinkhorn = SinkhornNormalizer(temperature, sinkhorn_iterations)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        
    def generate_permutation(self, x, generator, is_first_perm=False):
        """
        生成置换矩阵
        
        Args:
            x: 输入张量, [batch_size, seq_len, feature_dim]
            generator: 用于生成特征的网络
            is_first_perm: 是否为第一个置换(可选为恒等置换)
            
        Returns:
            置换矩阵, [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        if is_first_perm:
            # 恒等置换
            eye = torch.eye(seq_len, device=x.device)
            logits = eye.unsqueeze(0).expand(batch_size, -1, -1) * 10.0
        else:
            # 使用生成器网络产生特征
            features = generator(x)  # [batch, seq_len, feature_dim]
            
            # 计算特征之间的相似度矩阵
            logits = torch.bmm(features, features.transpose(1, 2))  # [batch, seq_len, seq_len]
        
        # 应用Sinkhorn归一化得到双随机矩阵
        return self.sinkhorn(logits)
        
    def forward(self, x):
        """
        模型前向传播
        
        Args:
            x: 输入张量, [batch_size, feature_dim, seq_len]
            
        Returns:
            模型输出, [batch_size, num_classes]
        """
        # 调整形状为 [batch, seq_len, feature_dim]
        x = x.transpose(1, 2)
        batch_size, seq_len, _ = x.shape
        
        # 保存所有路径的隐藏状态
        hidden_states = []
        lstm_idx = 0
        
        for i in range(self.num_permutations):
            # 第一次置换
            perm1 = self.generate_permutation(x, self.perm_generator1[i], is_first_perm=(i==0))
            x_perm1 = torch.bmm(perm1, x)  # [batch, seq_len, input_dim]
            
            # 嵌入
            embedded = self.embedding(x_perm1)  # [batch, seq_len, embed_dim]
            
            for j in range(self.num_permutations):
                # 第二次置换
                perm2 = self.generate_permutation(embedded, self.perm_generator2[j], is_first_perm=(j==0))
                x_perm2 = torch.bmm(perm2, embedded)  # [batch, seq_len, embed_dim]
                
                # LSTM处理
                _, (h_n, _) = self.lstms[lstm_idx](x_perm2)
                hidden_states.append(h_n.squeeze(0))
                lstm_idx += 1
        
        # 简单平均融合（不使用注意力机制）
        combined = torch.stack(hidden_states, dim=0)  # [num_perms^2, batch, hidden_dim]
        pooled = torch.mean(combined, dim=0)  # [batch, hidden_dim]
        
        # 输出层
        return self.output_layer(pooled)

class DoublePANN(nn.Module):
    """
    完整的双置换注意力神经网络
    """
    def __init__(self, input_dim, seq_length, embed_dim, hidden_dim, num_classes, 
                 num_permutations=3, num_heads=4, dropout=0.1, 
                 temperature=0.1, sinkhorn_iterations=5):
        super(DoublePANN, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_permutations = num_permutations
        self.num_heads = num_heads
        self.seq_length = seq_length
        
        # 嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 第一次置换（输入空间）的生成网络
        self.perm_generator1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            ) for _ in range(num_permutations)
        ])
        
        # 第二次置换（嵌入空间）的生成网络
        self.perm_generator2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_permutations)
        ])
        
        # LSTM编码器
        self.lstms = nn.ModuleList([
            nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            for _ in range(num_permutations * num_permutations)
        ])
        
        # Sinkhorn归一化器
        self.sinkhorn = SinkhornNormalizer(temperature, sinkhorn_iterations)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        
    def generate_permutation(self, x, generator, is_first_perm=False):
        """生成置换矩阵"""
        batch_size, seq_len, _ = x.shape
        
        if is_first_perm:
            # 恒等置换
            eye = torch.eye(seq_len, device=x.device)
            logits = eye.unsqueeze(0).expand(batch_size, -1, -1) * 10.0
        else:
            # 使用生成器网络产生特征
            features = generator(x)
            
            # 计算特征之间的相似度矩阵
            logits = torch.bmm(features, features.transpose(1, 2))
        
        # 应用Sinkhorn归一化得到双随机矩阵
        return self.sinkhorn(logits)
        
    def forward(self, x):
        """模型前向传播"""
        # 调整形状为 [batch, seq_len, feature_dim]
        x = x.transpose(1, 2)
        batch_size, seq_len, _ = x.shape
        
        # 保存所有路径的隐藏状态
        hidden_states = []
        lstm_idx = 0
        
        for i in range(self.num_permutations):
            # 第一次置换
            perm1 = self.generate_permutation(x, self.perm_generator1[i], is_first_perm=(i==0))
            x_perm1 = torch.bmm(perm1, x)
            
            # 嵌入
            embedded = self.embedding(x_perm1)
            
            for j in range(self.num_permutations):
                # 第二次置换
                perm2 = self.generate_permutation(embedded, self.perm_generator2[j], is_first_perm=(j==0))
                x_perm2 = torch.bmm(perm2, embedded)
                
                # LSTM处理
                _, (h_n, _) = self.lstms[lstm_idx](x_perm2)
                hidden_states.append(h_n.squeeze(0))
                lstm_idx += 1
        
        # 堆叠隐藏状态用于注意力处理
        # 从[num_perms^2, batch, hidden_dim]转为[batch, num_perms^2, hidden_dim]
        combined = torch.stack(hidden_states, dim=1)
        
        # 准备多头注意力的输入 - 需要转置为[num_perms^2, batch, hidden_dim]
        combined_t = combined.transpose(0, 1)
        
        # 应用多头注意力
        attn_output, _ = self.attention(combined_t, combined_t, combined_t)
        
        # 转回[batch, num_perms^2, hidden_dim]并平均池化
        attn_output = attn_output.transpose(0, 1)
        pooled = torch.mean(attn_output, dim=1)
        
        # 输出层
        return self.output_layer(pooled) 