#!/usr/bin/env python
# coding: utf-8
"""SOC PINN open-source training and evaluation script.

This file was converted from ``SOC_PINN_open_souce.ipynb`` and keeps the
original cell execution order. Run it from the project root with:

    python SOC_PINN_open_souce.py
"""

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

# ## 1. 环境准备

# In[1]:


import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'使用设备: {device}')
if device == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')


# In[2]:


# 导入自定义模块（强制重新加载）
import importlib
import soc_pinn_model_5
import soc_data_processor
import soc_trainer2

importlib.reload(soc_pinn_model_5)
importlib.reload(soc_data_processor)
importlib.reload(soc_trainer2)

from soc_pinn_model_5 import create_model
from soc_data_processor import SOCDataProcessor, quick_data_exploration
from soc_trainer2 import SOCTrainer, set_seed

# 设置随机种子
set_seed(42)


# ## 2. 数据探索

# In[3]:


# 数据目录 - 使用预划分数据集中的训练集进行探索
DATA_DIR = r'D:\Jupyter\SOC\PINTs_KAN_2\fangan3\Train'

# 快速数据探索
df_sample = quick_data_exploration(DATA_DIR)


# In[4]:


# 可视化单个放电片段
if df_sample is not None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # SOC变化
    ax1 = axes[0, 0]
    ax1.plot(df_sample['SOC'].values, 'b-', linewidth=1)
    ax1.set_xlabel('采样点')
    ax1.set_ylabel('SOC (%)')
    ax1.set_title('SOC变化曲线')
    ax1.grid(True, alpha=0.3)
    
    # 电流变化
    ax2 = axes[0, 1]
    if '总电流' in df_sample.columns:
        ax2.plot(df_sample['总电流'].values, 'r-', linewidth=1)
        ax2.set_xlabel('采样点')
        ax2.set_ylabel('电流 (A)')
        ax2.set_title('电流变化曲线')
        ax2.grid(True, alpha=0.3)
    
    # 电压变化
    ax3 = axes[0, 2]
    if '总电压' in df_sample.columns:
        ax3.plot(df_sample['总电压'].values, 'g-', linewidth=1)
        ax3.set_xlabel('采样点')
        ax3.set_ylabel('电压 (V)')
        ax3.set_title('电压变化曲线')
        ax3.grid(True, alpha=0.3)
    
    # 车速变化
    ax4 = axes[1, 0]
    if '车速' in df_sample.columns:
        ax4.plot(df_sample['车速'].values, 'm-', linewidth=1)
        ax4.set_xlabel('采样点')
        ax4.set_ylabel('车速 (km/h)')
        ax4.set_title('车速变化曲线')
        ax4.grid(True, alpha=0.3)
    
    # 电池单体电压最高/最低值
    ax5 = axes[1, 1]
    if '电池单体电压最高值' in df_sample.columns and '电池单体电压最低值' in df_sample.columns:
        ax5.plot(df_sample['电池单体电压最高值'].values, 'r-', linewidth=1, label='最高电压')
        ax5.plot(df_sample['电池单体电压最低值'].values, 'b-', linewidth=1, label='最低电压')
        ax5.set_xlabel('采样点')
        ax5.set_ylabel('单体电压 (mV)')
        ax5.set_title('电池单体电压变化')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 温度变化
    ax6 = axes[1, 2]
    if '最高温度值' in df_sample.columns and '最低温度值' in df_sample.columns:
        ax6.plot(df_sample['最高温度值'].values, 'r-', linewidth=1, label='最高温度')
        ax6.plot(df_sample['最低温度值'].values, 'b-', linewidth=1, label='最低温度')
        ax6.set_xlabel('采样点')
        ax6.set_ylabel('温度 (°C)')
        ax6.set_title('温度变化曲线')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # SOC与电流关系
    ax7 = axes[2, 0]
    if '总电流' in df_sample.columns:
        ax7.scatter(df_sample['总电流'].values, df_sample['SOC'].values, 
                   alpha=0.3, s=1)
        ax7.set_xlabel('电流 (A)')
        ax7.set_ylabel('SOC (%)')
        ax7.set_title('SOC vs 电流')
        ax7.grid(True, alpha=0.3)
    
    # SOC与车速关系
    ax8 = axes[2, 1]
    if '车速' in df_sample.columns:
        ax8.scatter(df_sample['车速'].values, df_sample['SOC'].values, 
                   alpha=0.3, s=1, c='m')
        ax8.set_xlabel('车速 (km/h)')
        ax8.set_ylabel('SOC (%)')
        ax8.set_title('SOC vs 车速')
        ax8.grid(True, alpha=0.3)
    
    # SOC与温度关系
    ax9 = axes[2, 2]
    if '最高温度值' in df_sample.columns:
        ax9.scatter(df_sample['最高温度值'].values, df_sample['SOC'].values, 
                   alpha=0.3, s=1, c='orange')
        ax9.set_xlabel('最高温度 (°C)')
        ax9.set_ylabel('SOC (%)')
        ax9.set_title('SOC vs 最高温度')
        ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# In[5]:


# ==================== 特征相关性热力图（所有Excel数据） ====================
import glob

# 定义要分析的特征列（包含新增特征）
correlation_features = ['SOC', '总电流', '总电压', '累计里程', '车速', 
                        '电池单体电压最高值', '电池单体电压最低值', 
                        '最高温度值', '最低温度值',
                        '环境温度', '相对湿度']  # 新增环境特征

# 加载所有Excel文件（不限于discharging）
all_files = glob.glob(os.path.join(DATA_DIR, '*.xlsx'))
print(f"找到 {len(all_files)} 个Excel数据文件")

# 合并所有数据
all_data = []
for file_path in all_files:
    try:
        df = pd.read_excel(file_path)
        # 筛选存在的特征列
        available_cols = [col for col in correlation_features if col in df.columns]
        if available_cols:
            all_data.append(df[available_cols])
    except Exception as e:
        print(f"加载 {os.path.basename(file_path)} 失败: {e}")

# 合并所有数据
df_all = pd.concat(all_data, ignore_index=True)
print(f"\n合并后总数据量: {len(df_all):,} 条记录")

# 筛选存在的列
available_features = [col for col in correlation_features if col in df_all.columns]
print(f"可用特征列: {available_features}")

# ==================== 空值检查与处理 ====================
print("\n" + "=" * 70)
print("空值检查（处理前）:")
print("=" * 70)

null_info = df_all[available_features].isnull().sum()
for col in available_features:
    null_count = null_info[col]
    null_pct = null_count / len(df_all) * 100
    status = "✅ 无空值" if null_count == 0 else f"⚠️ 有 {null_count:,} 个空值 ({null_pct:.2f}%)"
    print(f"  {col}: {status}")

# 删除含有空值的行
original_count = len(df_all)
df_all = df_all.dropna(subset=available_features)
removed_count = original_count - len(df_all)
print(f"\n删除空值后: {len(df_all):,} 条记录 (删除了 {removed_count:,} 条, {removed_count/original_count*100:.2f}%)")

# ==================== 各列数据统计 ====================
print("\n" + "=" * 70)
print("各列数据统计（所有Excel文件汇总，已清除空值）:")
print("=" * 70)
df_corr = df_all[available_features].copy()
for col in available_features:
    col_data = df_corr[col]
    std_val = col_data.std()
    print(f"  {col}:")
    print(f"    - 数据量: {len(col_data):,}")
    print(f"    - 唯一值数量: {col_data.nunique():,}")
    print(f"    - 均值: {col_data.mean():.4f}")
    print(f"    - 标准差: {std_val:.4f}" if std_val > 0 else f"    - 标准差: 0 (常量列!)")
    print(f"    - 范围: [{col_data.min():.2f}, {col_data.max():.2f}]")

# 移除常量列（标准差为0的列会导致相关性为NaN）
valid_features = [col for col in available_features if df_corr[col].std() > 0]
print(f"\n有效特征列（排除常量列）: {valid_features}")

if len(valid_features) < len(available_features):
    removed = set(available_features) - set(valid_features)
    print(f"⚠️ 移除的常量列: {removed}")

# 计算相关性矩阵（仅使用有效特征）
correlation_matrix = df_corr[valid_features].corr()

# 使用 matplotlib 绘制热力图
fig, ax = plt.subplots(figsize=(12, 10))

# 绘制热力图
im = ax.imshow(correlation_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

# 设置刻度
ax.set_xticks(np.arange(len(valid_features)))
ax.set_yticks(np.arange(len(valid_features)))
ax.set_xticklabels(valid_features, rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(valid_features, fontsize=11)

# 添加数值标注
for i in range(len(valid_features)):
    for j in range(len(valid_features)):
        value = correlation_matrix.iloc[i, j]
        text_color = 'white' if abs(value) > 0.5 else 'black'
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
               color=text_color, fontsize=10, fontweight='bold')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('相关系数', fontsize=12)

ax.set_title(f'特征相关性热力图 (所有数据, N={len(df_all):,})', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('feature_correlation_heatmap_all.png', dpi=150, bbox_inches='tight')
print("\n✅ 热力图已保存为 feature_correlation_heatmap_all.png")
plt.show()

# 打印与SOC相关性最强的特征
print("\n" + "=" * 50)
print("与SOC的相关性排序（绝对值）:")
print("=" * 50)
if 'SOC' in valid_features:
    soc_corr = correlation_matrix['SOC'].drop('SOC').abs().sort_values(ascending=False)
    for feature, corr_value in soc_corr.items():
        original_corr = correlation_matrix['SOC'][feature]
        print(f"  {feature}: {original_corr:+.4f}")


# In[6]:


# 2.3 特征统计信息表
# 使用 describe() 方法生成所有特征和标签列的统计摘要

# 选择特征列和目标列进行统计
stats_columns = correlation_features  # 使用相关性分析中的特征列

# 获取统计信息
stats_df = df_all[stats_columns].describe()

# 添加额外的统计信息
stats_df.loc['缺失值数量'] = df_all[stats_columns].isnull().sum()
stats_df.loc['缺失值比例(%)'] = (df_all[stats_columns].isnull().sum() / len(df_all) * 100).round(2)
stats_df.loc['非零值数量'] = (df_all[stats_columns] != 0).sum()
stats_df.loc['唯一值数量'] = df_all[stats_columns].nunique()

print("=" * 80)
print("特征统计信息表 (基于全部数据文件)")
print("=" * 80)
print(f"\n总样本数: {len(df_all):,} 行")
print(f"特征数量: {len(stats_columns)} 列\n")

# 显示统计表
display(stats_df.round(4))

# 以更友好的格式显示关键统计
print("\n" + "=" * 80)
print("关键统计摘要")
print("=" * 80)
for col in stats_columns:
    print(f"\n【{col}】")
    print(f"  范围: {df_all[col].min():.4f} ~ {df_all[col].max():.4f}")
    print(f"  均值 ± 标准差: {df_all[col].mean():.4f} ± {df_all[col].std():.4f}")
    print(f"  中位数: {df_all[col].median():.4f}")
    print(f"  缺失率: {df_all[col].isnull().sum() / len(df_all) * 100:.2f}%")


# ## 3. 数据准备

# In[7]:


# 配置参数
WINDOW_SIZE = 30        # 滑动窗口大小
PREDICTION_LENGTH = 30  # 预测步长
BATCH_SIZE = 32         # 批次大小

# 数据目录 - 使用预划分的文件夹
BASE_DATA_DIR = r'D:\Jupyter\SOC\PINTs_KAN_2\fangan3'
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'Train')
VAL_DIR = os.path.join(BASE_DATA_DIR, 'Vali')
TEST_DIR = os.path.join(BASE_DATA_DIR, 'Test')

# 特征列 - 新增车速、单体电压、温度、环境特征
FEATURE_COLUMNS = [
    '总电流', 
    '总电压', 
    'SOC', 
    '累计里程',
    '车速',                    # 新增：车速
    '电池单体电压最高值',       # 新增：电池单体电压最高值
    '电池单体电压最低值',       # 新增：电池单体电压最低值
    '最高温度值',              # 新增：最高温度值
    '最低温度值',              # 新增：最低温度值
    '环境温度',                # 新增：环境温度
    '相对湿度'                 # 新增：相对湿度
]
TARGET_COLUMN = 'SOC'

print(f'滑动窗口大小: {WINDOW_SIZE}')
print(f'预测步长: {PREDICTION_LENGTH}')
print(f'特征列 ({len(FEATURE_COLUMNS)} 个):')
for i, col in enumerate(FEATURE_COLUMNS, 1):
    print(f'  {i}. {col}')
print(f'目标列: {TARGET_COLUMN}')
print(f'\n数据目录:')
print(f'  训练集: {TRAIN_DIR}')
print(f'  验证集: {VAL_DIR}')
print(f'  测试集: {TEST_DIR}')


# In[8]:


# ==================== 自定义数据加载函数 ====================
# 直接从预划分的 Train/Vali/Test 文件夹加载数据
# 每个 excel 文件单独构建滑动窗口数据，然后合并

import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader

class SOCDataset(Dataset):
    """SOC预测数据集"""
    def __init__(self, X, y, current_future=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.current_future = torch.FloatTensor(current_future) if current_future is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.current_future is not None:
            return self.X[idx], self.y[idx], self.current_future[idx]
        return self.X[idx], self.y[idx]

class SimpleDataProcessor:
    """简化的数据处理器，用于兼容现有的评估代码"""
    def __init__(self, feature_scaler, target_scaler, feature_columns, target_column, 
                 window_size, prediction_length):
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.window_size = window_size
        self.prediction_length = prediction_length
    
    def inverse_transform_y(self, y_scaled):
        """反转换目标值"""
        y_flat = y_scaled.reshape(-1, 1)
        y_original = self.target_scaler.inverse_transform(y_flat)
        return y_original.reshape(y_scaled.shape)
    
    def preprocess_single_trip(self, df):
        """预处理单个文件数据（兼容原有接口）"""
        # 提取相关列（去重）
        cols_to_extract = list(dict.fromkeys(self.feature_columns + [self.target_column]))
        
        # 检查必要的列是否存在
        missing_cols = [col for col in cols_to_extract if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")
        
        data = df[cols_to_extract].copy()
        
        # 处理缺失值
        data = data.dropna()
        
        # 添加衍生特征
        if '总电流' in data.columns and '总电压' in data.columns:
            data['功率'] = data['总电流'].values * data['总电压'].values
            
        # 添加SOC变化率
        if 'SOC' in data.columns:
            soc_values = data['SOC']
            if isinstance(soc_values, pd.DataFrame):
                soc_values = soc_values.iloc[:, 0]
            data['SOC_diff'] = soc_values.diff().fillna(0).values
            
        return data
    
    def create_sliding_windows(self, data, include_current_future=True):
        """创建滑动窗口数据"""
        if isinstance(data, pd.DataFrame):
            feature_cols = [col for col in data.columns if col != self.target_column and col != 'trip_id']
            features = data[feature_cols].values
            target = data[self.target_column].values
            current = data['总电流'].values if '总电流' in data.columns else None
        else:
            features = data[:, :-1]
            target = data[:, -1]
            current = data[:, 0]
        
        X, y, current_future = [], [], []
        total_length = self.window_size + self.prediction_length
        
        for i in range(len(features) - total_length + 1):
            X.append(features[i:i + self.window_size])
            y.append(target[i + self.window_size:i + total_length])
            
            if include_current_future and current is not None:
                current_future.append(current[i + self.window_size:i + total_length])
        
        X = np.array(X)
        y = np.array(y)
        
        if include_current_future and current is not None:
            current_future = np.array(current_future)
            return X, y, current_future
            
        return X, y, None
    
    def transform(self, X, y, current_future=None):
        """转换数据"""
        n_samples, window_size, n_features = X.shape
        
        # 标准化特征
        X_flat = X.reshape(-1, n_features)
        X_scaled = self.feature_scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(n_samples, window_size, n_features)
        
        # 标准化目标
        y_flat = y.reshape(-1, 1)
        y_scaled = self.target_scaler.transform(y_flat)
        y_scaled = y_scaled.reshape(n_samples, -1)
        
        return X_scaled, y_scaled, current_future

def preprocess_single_file(df, feature_columns, target_column):
    """预处理单个文件数据"""
    # 提取相关列（去重）
    cols_to_extract = list(dict.fromkeys(feature_columns + [target_column]))
    
    # 检查必要的列是否存在
    missing_cols = [col for col in cols_to_extract if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")
    
    data = df[cols_to_extract].copy()
    
    # 处理缺失值
    data = data.dropna()
    
    # 添加衍生特征
    if '总电流' in data.columns and '总电压' in data.columns:
        data['功率'] = data['总电流'].values * data['总电压'].values
        
    # 添加SOC变化率
    if 'SOC' in data.columns:
        soc_values = data['SOC']
        if isinstance(soc_values, pd.DataFrame):
            soc_values = soc_values.iloc[:, 0]
        data['SOC_diff'] = soc_values.diff().fillna(0).values
        
    return data

def create_sliding_windows(data, window_size, prediction_length, target_column):
    """创建滑动窗口数据"""
    if isinstance(data, pd.DataFrame):
        feature_cols = [col for col in data.columns if col != target_column and col != 'trip_id']
        features = data[feature_cols].values
        target = data[target_column].values
        current = data['总电流'].values if '总电流' in data.columns else None
    else:
        features = data[:, :-1]
        target = data[:, -1]
        current = data[:, 0]
    
    X, y, current_future = [], [], []
    total_length = window_size + prediction_length
    
    for i in range(len(features) - total_length + 1):
        X.append(features[i:i + window_size])
        y.append(target[i + window_size:i + total_length])
        
        if current is not None:
            current_future.append(current[i + window_size:i + total_length])
    
    X = np.array(X)
    y = np.array(y)
    
    if current is not None:
        current_future = np.array(current_future)
        return X, y, current_future
        
    return X, y, None

def load_and_process_folder(folder_path, feature_columns, target_column, 
                            window_size, prediction_length, min_samples=10):
    """
    加载并处理一个文件夹中的所有 Excel 文件
    每个 Excel 文件单独构建滑动窗口，然后合并
    """
    files = glob.glob(os.path.join(folder_path, '*.xlsx'))
    print(f"\n加载文件夹: {folder_path}")
    print(f"  找到 {len(files)} 个 Excel 文件")
    
    all_X, all_y, all_current = [], [], []
    file_info_list = []
    skipped_short = 0
    skipped_error = 0
    
    for file_path in sorted(files):
        file_name = os.path.basename(file_path)
        try:
            # 读取 Excel 文件
            df = pd.read_excel(file_path)
            
            # 跳过样本数太少的文件
            if len(df) < min_samples:
                skipped_short += 1
                continue
            
            # 预处理
            processed = preprocess_single_file(df, feature_columns, target_column)
            
            # 检查处理后的数据是否足够创建滑动窗口
            if len(processed) < window_size + prediction_length:
                skipped_short += 1
                continue
            
            # 创建滑动窗口
            X, y, current = create_sliding_windows(
                processed, window_size, prediction_length, target_column
            )
            
            all_X.append(X)
            all_y.append(y)
            if current is not None:
                all_current.append(current)
            
            # 记录文件信息
            file_info_list.append({
                'file_name': file_name,
                'original_df': df.copy(),
                'n_samples': len(X)
            })
            
        except Exception as e:
            print(f"    处理文件 {file_name} 失败: {e}")
            skipped_error += 1
            continue
    
    if not all_X:
        raise ValueError(f"文件夹 {folder_path} 中没有有效数据")
    
    # 合并所有数据
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    current_future = np.concatenate(all_current, axis=0) if all_current else None
    
    print(f"  成功加载: {len(file_info_list)} 个文件")
    print(f"  跳过(数据太短): {skipped_short} 个")
    print(f"  跳过(处理错误): {skipped_error} 个")
    print(f"  总样本数: {len(X)}")
    
    return X, y, current_future, file_info_list

# ==================== 加载训练集、验证集、测试集 ====================
print("=" * 70)
print("加载预划分的数据集")
print("=" * 70)

# 加载训练集
X_train_raw, y_train_raw, current_train_raw, train_files_info = load_and_process_folder(
    TRAIN_DIR, FEATURE_COLUMNS, TARGET_COLUMN, WINDOW_SIZE, PREDICTION_LENGTH
)

# 加载验证集
X_val_raw, y_val_raw, current_val_raw, val_files_info = load_and_process_folder(
    VAL_DIR, FEATURE_COLUMNS, TARGET_COLUMN, WINDOW_SIZE, PREDICTION_LENGTH
)

# 加载测试集
X_test_raw, y_test_raw, current_test_raw, test_files_info = load_and_process_folder(
    TEST_DIR, FEATURE_COLUMNS, TARGET_COLUMN, WINDOW_SIZE, PREDICTION_LENGTH
)

# ==================== 数据标准化 ====================
print("\n" + "=" * 70)
print("数据标准化")
print("=" * 70)

# 创建标准化器
feature_scaler = StandardScaler()
target_scaler = MinMaxScaler(feature_range=(0, 100))

# 拟合标准化器（只用训练数据）
n_samples_train, window_size, n_features = X_train_raw.shape
X_train_flat = X_train_raw.reshape(-1, n_features)
feature_scaler.fit(X_train_flat)
target_scaler.fit(y_train_raw.reshape(-1, 1))

# 转换训练集
X_train_scaled = feature_scaler.transform(X_train_flat).reshape(n_samples_train, window_size, n_features)
y_train_scaled = target_scaler.transform(y_train_raw.reshape(-1, 1)).reshape(n_samples_train, -1)

# 转换验证集
n_samples_val = X_val_raw.shape[0]
X_val_scaled = feature_scaler.transform(X_val_raw.reshape(-1, n_features)).reshape(n_samples_val, window_size, n_features)
y_val_scaled = target_scaler.transform(y_val_raw.reshape(-1, 1)).reshape(n_samples_val, -1)

# 转换测试集
n_samples_test = X_test_raw.shape[0]
X_test_scaled = feature_scaler.transform(X_test_raw.reshape(-1, n_features)).reshape(n_samples_test, window_size, n_features)
y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1, 1)).reshape(n_samples_test, -1)

print(f"特征数: {n_features}")
print(f"训练集形状: X={X_train_scaled.shape}, y={y_train_scaled.shape}")
print(f"验证集形状: X={X_val_scaled.shape}, y={y_val_scaled.shape}")
print(f"测试集形状: X={X_test_scaled.shape}, y={y_test_scaled.shape}")

# ==================== 创建数据集和DataLoader ====================
train_dataset = SOCDataset(X_train_scaled, y_train_scaled, current_train_raw)
val_dataset = SOCDataset(X_val_scaled, y_val_scaled, current_val_raw)
test_dataset = SOCDataset(X_test_scaled, y_test_scaled, current_test_raw)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

print(f"\nDataLoader 已创建:")
print(f"  训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
print(f"  验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
print(f"  测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")

# ==================== 创建兼容的 processor 对象 ====================
# 用于与现有的评估代码兼容
processor = SimpleDataProcessor(
    feature_scaler=feature_scaler,
    target_scaler=target_scaler,
    feature_columns=FEATURE_COLUMNS,
    target_column=TARGET_COLUMN,
    window_size=WINDOW_SIZE,
    prediction_length=PREDICTION_LENGTH
)

# 保存额外信息供后续使用
processor.test_files_info = test_files_info

print("\n✅ processor 对象已创建，包含以下方法:")
print("  - preprocess_single_trip(df)")
print("  - create_sliding_windows(data)")
print("  - transform(X, y, current_future)")
print("  - inverse_transform_y(y_scaled)")


# In[9]:


# 检查数据形状
for batch in train_loader:
    if len(batch) == 3:
        X, y, current = batch
        print(f'输入形状: {X.shape} (batch, window_size, features)')
        print(f'目标形状: {y.shape} (batch, prediction_length)')
        print(f'未来电流形状: {current.shape} (batch, prediction_length)')
    else:
        X, y = batch
        print(f'输入形状: {X.shape}')
        print(f'目标形状: {y.shape}')
    INPUT_SIZE = X.shape[2]
    print(f'\n输入特征数: {INPUT_SIZE}')
    print(f'特征列表包含:')
    print(f'  - 原始特征 ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}')
    print(f'  - 衍生特征: 功率(电流*电压), SOC_diff(SOC变化率)')
    break

# 打印测试集文件信息
print(f"\n测试集包含 {len(test_files_info)} 个完整文件:")
for i, info in enumerate(test_files_info[:10]):  # 只显示前10个
    print(f"  {i+1}. {info['file_name']} ({info['n_samples']} 样本)")
if len(test_files_info) > 10:
    print(f"  ... 还有 {len(test_files_info) - 10} 个文件")


# ## 4. KAN模型定义与训练

# ### 4.1 高效KAN模型 (EfficientKAN) - 推荐
# 
# 基于 KAN_vs_Trans_vs_LSTM_comparison.py 中的实现，使用真正的 B-spline 基函数。
# 
# **与之前KAN模型的主要区别:**
# - 使用真正的 B-spline 递归计算，而非 RBF 近似
# - 包含 `grid_eps`, `scale_noise`, `scale_base`, `scale_spline` 等参数
# - 参数量更少 (~71,792 根据配置)
# - 更好的数值稳定性

# In[10]:


# 重新加载模块以使用 EfficientKAN
import importlib
import sys

# 清除缓存
modules_to_reload = [k for k in sys.modules.keys() if 'soc_pinn_model_5' in k]
for mod in modules_to_reload:
    del sys.modules[mod]

import importlib.util
spec = importlib.util.spec_from_file_location(
    "soc_pinn_model_5", 
    r"d:\Jupyter\SOC\PINTs_KAN_2\soc_pinn_model_5.py"
)
soc_pinn_model_5 = importlib.util.module_from_spec(spec)
sys.modules['soc_pinn_model_5'] = soc_pinn_model_5
spec.loader.exec_module(soc_pinn_model_5)
from soc_pinn_model_5 import (create_model, EfficientPureKAN, PhysicsInformedEfficientKAN, 
                            EnhancedPhysicsInformedEfficientKAN, EfficientKANLinear)

print("EfficientKAN 模块加载成功!")
print("\n可用的 EfficientKAN 模型类型:")
print("- efficient_kan: 高效纯KAN模型")
print("- pinn_efficient_kan: 物理信息高效KAN模型")  
print("- enhanced_pinn_efficient_kan: 增强版物理信息高效KAN模型")


# In[11]:


# EfficientKAN 模型配置 (参考 KAN_vs_Trans_vs_LSTM_comparison.py)
# 注意：hidden_dims 不应该包含最后的输出维度1，因为output_size会自动添加
EFFICIENT_KAN_CONFIG = {
    'input_size': INPUT_SIZE,
    'hidden_size': 16,           # 不用于EfficientKAN
    'num_layers': 2,             # 不用于EfficientKAN  
    'output_size': PREDICTION_LENGTH,
    'dropout': 0.1,
    'grid_size': 5,              # B-spline 网格大小 (原始使用2)
    'window_size': WINDOW_SIZE,
    'hidden_dims': [8, 8] # KAN隐藏层结构 (去掉了最后的1)
}

print('EfficientKAN 模型配置:')
for k, v in EFFICIENT_KAN_CONFIG.items():
    print(f'  {k}: {v}')
print(f'\n展平后输入维度: {INPUT_SIZE * WINDOW_SIZE}')
print(f'网络结构: [{INPUT_SIZE * WINDOW_SIZE}] -> [16] -> [8] -> [{PREDICTION_LENGTH}]')

# 参数量计算说明:
# 本项目特征配置:
# - 原始特征: 总电流、总电压、SOC、累计里程 (4个)
# - 新增特征: 车速、电池单体电压最高/最低值、最高/最低温度 (5个)
# - 衍生特征: 功率、SOC_diff (2个)
# - input_dim = 特征数 * 窗口大小 = INPUT_SIZE * 25
# - output = 180 (多步预测)
print(f'\n特征数量: {INPUT_SIZE} (原始+衍生特征)')


# In[12]:


# 创建 EfficientKAN 模型
# 【重要】在创建模型前设置随机种子，确保模型权重初始化一致
from soc_trainer2 import set_seed
set_seed(42)

efficient_kan_model = create_model(
    model_type='efficient_kan',  # 使用高效KAN模型
    device=device,
    **EFFICIENT_KAN_CONFIG
)

print(efficient_kan_model)
print(f'\nEfficientKAN 模型参数总数: {sum(p.numel() for p in efficient_kan_model.parameters()):,}')


# In[13]:


# 打印 EfficientKAN 模型的详细结构和每层参数
print("=" * 70)
print("EfficientKAN 模型结构详情")
print("=" * 70)

# 获取 KAN 层的数量
num_kan_layers = len(efficient_kan_model.kan.layers)
print(f"\nKAN 层数量: {num_kan_layers}")

# 打印每个 KAN 层的参数
print(f"\n各层参数统计:")
total_params = 0
for i, layer in enumerate(efficient_kan_model.kan.layers):
    layer_params = sum(p.numel() for p in layer.parameters())
    total_params += layer_params
    # 获取层的输入输出维度
    in_features = layer.in_features
    out_features = layer.out_features
    print(f"  KAN Layer {i}: {layer_params:,} 参数 (输入: {in_features}, 输出: {out_features})")

# 检查是否有 capacity 参数（对于 PINN 模型）
if hasattr(efficient_kan_model, 'capacity'):
    print(f"  capacity: 1 参数 (当前值: {efficient_kan_model.capacity.item():.2f} Ah)")
    total_params += 1

print(f"\n总参数数量: {total_params:,}")
print("=" * 70)

# 更详细的参数分解（每个子参数）
print("\n详细参数分解:")
for i, layer in enumerate(efficient_kan_model.kan.layers):
    print(f"\n  KAN Layer {i}:")
    for name, param in layer.named_parameters():
        print(f"    {name}: {param.shape} -> {param.numel():,} 参数")


# In[14]:


# ==================== EfficientKAN 网络结构可视化（自动从模型读取） ====================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def visualize_kan_structure(model, model_name="EfficientKAN", window_size=None, input_size=None):
    """
    自动从模型中读取网络结构并可视化
    
    Args:
        model: KAN模型对象（需要有 .kan.layers 属性）
        model_name: 模型名称
        window_size: 窗口大小（用于标注）
        input_size: 输入特征数（用于标注）
    """
    # 从模型中提取层信息
    kan_layers = model.kan.layers
    num_layers = len(kan_layers)
    
    # 提取每层的输入输出维度和grid_size
    layer_dims = []
    grid_sizes = []
    spline_orders = []
    
    for i, layer in enumerate(kan_layers):
        in_feat = layer.in_features
        out_feat = layer.out_features
        g_size = layer.grid_size if hasattr(layer, 'grid_size') else 5
        s_order = layer.spline_order if hasattr(layer, 'spline_order') else 3
        layer_dims.append((in_feat, out_feat))
        grid_sizes.append(g_size)
        spline_orders.append(s_order)
    
    # 获取输入维度（第一层的输入）
    input_dim = layer_dims[0][0]
    output_dim = layer_dims[-1][1]
    
    # 计算每层参数
    def calc_kan_params(in_feat, out_feat, grid_size, spline_order=3):
        base_weight = out_feat * in_feat
        spline_weight = out_feat * in_feat * (grid_size + spline_order)
        spline_scaler = out_feat * in_feat
        return {
            'base_weight': base_weight,
            'spline_weight': spline_weight,
            'spline_scaler': spline_scaler,
            'total': base_weight + spline_weight + spline_scaler
        }
    
    # 构建层配置
    layers_config = []
    
    # 输入层
    if window_size and input_size:
        input_label = f'输入层\n{window_size}×{input_size}={input_dim}'
    else:
        input_label = f'输入层\n{input_dim}'
    layers_config.append({'name': 'Input', 'neurons': input_dim, 'label': input_label})
    
    # KAN层
    for i, (in_f, out_f) in enumerate(layer_dims):
        if i == num_layers - 1:
            # 最后一层作为输出层
            layers_config.append({'name': f'输出层', 'neurons': out_f, 'label': f'输出层\n{out_f} 神经元'})
        else:
            layers_config.append({'name': f'KAN Layer {i}', 'neurons': out_f, 'label': f'KAN Layer {i}\n{out_f} 神经元'})
    
    # 计算所有层的参数
    all_params = []
    for i, ((in_f, out_f), g_size, s_order) in enumerate(zip(layer_dims, grid_sizes, spline_orders)):
        params = calc_kan_params(in_f, out_f, g_size, s_order)
        params['in_features'] = in_f
        params['out_features'] = out_f
        params['grid_size'] = g_size
        params['spline_order'] = s_order
        all_params.append(params)
    
    total_params = sum(p['total'] for p in all_params)
    
    # ============ 创建图形 ============
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    # ============ 左图：网络结构示意图 ============
    ax1 = axes[0]
    n_layers = len(layers_config)
    ax1.set_xlim(-0.5, n_layers + 0.5)
    ax1.set_ylim(-1, 12)
    ax1.axis('off')
    ax1.set_title(f'{model_name} 网络结构', fontsize=16, fontweight='bold', pad=20)
    
    # 动态计算层的位置
    layer_x = [i + 0.5 for i in range(n_layers)]
    
    # 颜色配置
    colors = ['#E8F5E9'] + ['#BBDEFB'] * (n_layers - 2) + ['#FFECB3']
    edge_colors = ['#4CAF50'] + ['#2196F3'] * (n_layers - 2) + ['#FFC107']
    
    # 绘制层
    for i, (lx, layer, color, edge_color) in enumerate(zip(layer_x, layers_config, colors, edge_colors)):
        n_neurons = layer['neurons']
        
        # 绘制层的方框
        box_height = 8
        box_width = 0.7
        rect = FancyBboxPatch((lx - box_width/2, 1.5), box_width, box_height,
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=color, edgecolor=edge_color, linewidth=2)
        ax1.add_patch(rect)
        
        # 层标签
        ax1.text(lx, 10.5, layer['label'], ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 神经元数量
        ax1.text(lx, 5.5, f'{n_neurons}', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 绘制连接线和参数信息
        if i > 0:
            params = all_params[i-1]
            
            # 绘制箭头
            ax1.annotate('', xy=(lx - box_width/2 - 0.03, 5.5), 
                        xytext=(layer_x[i-1] + box_width/2 + 0.03, 5.5),
                        arrowprops=dict(arrowstyle='->', color='#666666', lw=2))
            
            # 参数标签
            mid_x = (lx + layer_x[i-1]) / 2
            ax1.text(mid_x, 7.2, f'{params["total"]:,}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color='#D32F2F')
            ax1.text(mid_x, 6.5, '参数', ha='center', va='bottom', fontsize=8, color='#666')
    
    # 添加图例说明
    grid_size = grid_sizes[0]
    spline_order = spline_orders[0]
    legend_y = 0.3
    ax1.text(n_layers/2, legend_y, f'📊 参数公式: out × in × (1 + grid_size + spline_order + 1) = out × in × {2 + grid_size + spline_order}', 
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='#FFF3E0', edgecolor='#FF9800'))
    
    # ============ 右图：参数分布详情 ============
    ax2 = axes[1]
    
    # 准备数据
    layer_names = [f'Layer {i}\n{p["in_features"]}→{p["out_features"]}' for i, p in enumerate(all_params)]
    
    base_weights = [p['base_weight'] for p in all_params]
    spline_weights = [p['spline_weight'] for p in all_params]
    spline_scalers = [p['spline_scaler'] for p in all_params]
    totals = [p['total'] for p in all_params]
    
    x = np.arange(len(layer_names))
    width = 0.25
    
    # 堆叠柱状图
    bars1 = ax2.bar(x - width, base_weights, width, label='base_weight (out×in)', color='#4CAF50', alpha=0.8)
    bars2 = ax2.bar(x, spline_weights, width, label=f'spline_weight (out×in×{grid_size+spline_order})', color='#2196F3', alpha=0.8)
    bars3 = ax2.bar(x + width, spline_scalers, width, label='spline_scaler (out×in)', color='#FF9800', alpha=0.8)
    
    # 添加总数标签
    max_height = max(max(base_weights), max(spline_weights), max(spline_scalers))
    for i, (total, x_pos) in enumerate(zip(totals, x)):
        bar_max = max(base_weights[i], spline_weights[i], spline_scalers[i])
        ax2.text(x_pos, bar_max + max_height * 0.08, 
                f'总计: {total:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        # 百分比
        pct = total / sum(totals) * 100
        ax2.text(x_pos, bar_max + max_height * 0.02, 
                f'({pct:.1f}%)', ha='center', va='bottom', fontsize=9, color='#666')
    
    ax2.set_xlabel('KAN 层', fontsize=12)
    ax2.set_ylabel('参数数量', fontsize=12)
    ax2.set_title('各层参数分布详情', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_names)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加公式说明框
    formula_text = f"""参数计算公式 (grid_size={grid_size}, spline_order={spline_order}):
• base_weight: out_features × in_features
• spline_weight: out_features × in_features × (grid_size + spline_order)
                = out × in × ({grid_size} + {spline_order}) = out × in × {grid_size + spline_order}
• spline_scaler: out_features × in_features

总参数 = base_weight + spline_weight + spline_scaler
       = out × in × (1 + {grid_size + spline_order} + 1) = out × in × {2 + grid_size + spline_order}"""
    
    ax2.text(0.02, 0.98, formula_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#1976D2', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_structure.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # 打印详细参数表
    print("\n" + "="*80)
    print(f"{model_name} 各层参数详细计算")
    print("="*80)
    print(f"配置: grid_size={grid_size}, spline_order={spline_order}")
    print(f"B-spline 基函数数量 = grid_size + spline_order = {grid_size} + {spline_order} = {grid_size + spline_order}")
    print()
    
    for i, params in enumerate(all_params):
        in_f = params['in_features']
        out_f = params['out_features']
        gs = params['grid_size']
        so = params['spline_order']
        print(f"KAN Layer {i}: {in_f} → {out_f}")
        print(f"  base_weight:   {out_f} × {in_f} = {params['base_weight']:,}")
        print(f"  spline_weight: {out_f} × {in_f} × {gs + so} = {params['spline_weight']:,}")
        print(f"  spline_scaler: {out_f} × {in_f} = {params['spline_scaler']:,}")
        print(f"  小计: {params['total']:,} 参数 ({params['total']/total_params*100:.1f}%)")
        print()
    
    print(f"总参数量: {total_params:,}")
    
    # 检查是否有 capacity 参数
    if hasattr(model, 'capacity'):
        print(f"\n额外参数:")
        print(f"  capacity: 1 参数 (当前值: {model.capacity.item():.2f})")
        print(f"  真实总参数量: {total_params + 1:,}")
    
    return all_params

# ============ 调用可视化函数 ============
print("=" * 80)
print("自动检测模型结构并可视化")
print("=" * 80)

# 使用当前创建的 efficient_kan_model
params_info = visualize_kan_structure(
    model=efficient_kan_model, 
    model_name="EfficientKAN",
    window_size=WINDOW_SIZE,
    input_size=INPUT_SIZE
)


# In[15]:


# 训练 EfficientKAN 模型
efficient_kan_trainer = SOCTrainer(
    model=efficient_kan_model,
    device=device,
    save_dir='checkpoints/efficient_kan'
)

# 基础 EfficientKAN 不使用物理约束
efficient_kan_history = efficient_kan_trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=20,
    lr=0.001,
    physics_loss_weight=0.0,  # 不使用物理约束
    patience=50
)


# In[16]:


# 绘制 EfficientKAN 训练曲线
efficient_kan_trainer.plot_training_history()


# In[17]:


# 评估 EfficientKAN 模型
efficient_kan_metrics, efficient_kan_predictions, efficient_kan_targets = efficient_kan_trainer.evaluate(
    test_loader, processor
)

print("\n=== EfficientKAN 模型评估结果 ===")
for k, v in efficient_kan_metrics.items():
    print(f'{k}: {v:.4f}')


# ### 4.2 物理信息高效KAN模型 (pinn_efficient_kan)

# In[18]:


# 4.2 PINN EfficientKAN 配置（单个物理约束 - 库仑计数）
PINN_EFFICIENT_KAN_CONFIG = {
    'input_size': INPUT_SIZE,
    'hidden_size': 16,
    'num_layers': 2,
    'output_size': PREDICTION_LENGTH,
    'dropout': 0.1,
    'grid_size': 5,
    'window_size': WINDOW_SIZE,
    'hidden_dims': [8, 8],
    'capacity_init': 142.5
}

from soc_trainer2 import set_seed
set_seed(42)

# 创建 PINN EfficientKAN 模型
pinn_efficient_kan_model = create_model(
    model_type='pinn_efficient_kan',
    device=device,
    **PINN_EFFICIENT_KAN_CONFIG
)

print(pinn_efficient_kan_model)
print(f'\nPINN EfficientKAN 模型参数总数: {sum(p.numel() for p in pinn_efficient_kan_model.parameters()):,}')
print(f'初始电池容量: {pinn_efficient_kan_model.capacity.item():.2f} Ah')


# In[19]:


# 训练 PINN EfficientKAN 模型
pinn_efficient_kan_trainer = SOCTrainer(
    model=pinn_efficient_kan_model,
    device=device,
    save_dir='checkpoints/pinn_efficient_kan'
)

pinn_efficient_kan_history = pinn_efficient_kan_trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=20,
    lr=0.001,
    physics_loss_weight=50,  # 单个物理约束
    patience=50
)


# In[20]:


# 评估
pinn_efficient_kan_metrics, pinn_efficient_kan_predictions, pinn_efficient_kan_targets =     pinn_efficient_kan_trainer.evaluate(test_loader, processor)

print("\n=== PINN EfficientKAN 模型评估结果 ===")
for k, v in pinn_efficient_kan_metrics.items():
    print(f'{k}: {v:.4f}')
print(f'\n学习到的电池容量: {pinn_efficient_kan_model.capacity.item():.2f} Ah')


# ### 4.3 增强版物理信息高效KAN模型 (Enhanced PINN EfficientKAN)

# In[21]:


# Enhanced PINN EfficientKAN 配置
ENHANCED_PINN_EFFICIENT_KAN_CONFIG = {
    'input_size': INPUT_SIZE,
    'hidden_size': 16,
    'num_layers': 2,
    'output_size': PREDICTION_LENGTH,
    'dropout': 0.1,
    'grid_size': 5,
    'window_size': WINDOW_SIZE,
    'hidden_dims': [8, 8],
    'capacity_init': 142.5  # 电池容量初始值
}

from soc_trainer2 import set_seed
set_seed(42)

# 创建 Enhanced PINN EfficientKAN 模型
enhanced_pinn_efficient_kan_model = create_model(
    model_type='enhanced_pinn_efficient_kan',
    device=device,
    **ENHANCED_PINN_EFFICIENT_KAN_CONFIG
)

print(enhanced_pinn_efficient_kan_model)
print(f'\nEnhanced PINN EfficientKAN 模型参数总数: {sum(p.numel() for p in enhanced_pinn_efficient_kan_model.parameters()):,}')
print(f'初始电池容量: {enhanced_pinn_efficient_kan_model.capacity.item():.2f} Ah')


# In[22]:


# 训练 Enhanced PINN EfficientKAN 模型
enhanced_pinn_efficient_kan_trainer = SOCTrainer(
    model=enhanced_pinn_efficient_kan_model,
    device=device,
    save_dir='checkpoints/enhanced_pinn_efficient_kan'
)

# Enhanced模型使用三个独立的物理约束权重
enhanced_pinn_efficient_kan_history = enhanced_pinn_efficient_kan_trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=20,
    lr=0.001,
    # 三个独立的物理约束权重
    lambda_coulomb=50.0,     # 库仑计数约束权重
    lambda_boundary=0.01,    # SOC边界约束权重 (0-100)
    lambda_monotonic=0.01,   # 单调性约束权重
    patience=50
)


# In[23]:


# 评估 Enhanced PINN EfficientKAN 模型
enhanced_pinn_efficient_kan_metrics, enhanced_pinn_efficient_kan_predictions, enhanced_pinn_efficient_kan_targets =     enhanced_pinn_efficient_kan_trainer.evaluate(test_loader, processor)

print("\n=== Enhanced PINN EfficientKAN 模型评估结果 ===")
for k, v in enhanced_pinn_efficient_kan_metrics.items():
    print(f'{k}: {v:.4f}')
print(f'\n学习到的电池容量: {enhanced_pinn_efficient_kan_model.capacity.item():.2f} Ah')


# ## 5. KAN Model Comparison

# In[28]:


# ==================== KAN model comparison table ====================
# Only compare the three KAN variants: no physics constraint, Coulomb constraint,
# and enhanced three-constraint PINN.

import io


def get_model_size_mb(model):
    """Compute model state_dict size in MB."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 * 1024)


kan_models_info = {
    'EfficientKAN': {
        'metrics': efficient_kan_metrics,
        'model': efficient_kan_model,
        'physics_constraint': 'None',
    },
    'PINN EfficientKAN': {
        'metrics': pinn_efficient_kan_metrics,
        'model': pinn_efficient_kan_model,
        'physics_constraint': 'Coulomb counting',
    },
    'Enhanced PINN EfficientKAN': {
        'metrics': enhanced_pinn_efficient_kan_metrics,
        'model': enhanced_pinn_efficient_kan_model,
        'physics_constraint': 'Coulomb + boundary + monotonic',
    },
}

kan_comparison_data = []
for name, info in kan_models_info.items():
    metrics = info['metrics']
    model = info['model']
    kan_comparison_data.append({
        'Model': name,
        'Architecture': 'KAN',
        'Physics Constraint': info['physics_constraint'],
        'MSE': metrics['mse'],
        'RMSE': metrics['rmse'],
        'MAE': metrics['mae'],
        'R2': metrics['r2'],
        'MAPE (%)': metrics['mape'],
        'Parameters': sum(p.numel() for p in model.parameters()),
        'Model Size (MB)': get_model_size_mb(model),
    })

df_kan_comparison = pd.DataFrame(kan_comparison_data)

print('\n' + '=' * 120)
print('KAN Model Performance Comparison')
print('=' * 120)
header = (
    f"{'Model':<28} {'Constraint':<32} {'MSE':>10} {'RMSE':>10} "
    f"{'MAE':>10} {'R2':>8} {'MAPE(%)':>10} {'Params':>12} {'Size(MB)':>10}"
)
print(header)
print('-' * 120)
for _, row in df_kan_comparison.iterrows():
    print(
        f"{row['Model']:<28} {row['Physics Constraint']:<32} "
        f"{row['MSE']:>10.4f} {row['RMSE']:>10.4f} {row['MAE']:>10.4f} "
        f"{row['R2']:>8.4f} {row['MAPE (%)']:>10.2f} "
        f"{row['Parameters']:>12,} {row['Model Size (MB)']:>10.3f}"
    )
print('=' * 120)

df_kan_comparison.to_csv('kan_model_comparison_results.csv', index=False, encoding='utf-8-sig')
print('\nKAN comparison results saved to kan_model_comparison_results.csv')

display(df_kan_comparison)

