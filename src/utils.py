#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
utils.py - 工具函數模組
本模組提供銲錫接點疲勞壽命預測系統中使用的各種工具函數，
包括數據可視化、結果評估、檔案處理和模型輔助工具。

主要功能:
1. 數據可視化工具 - 生成各類圖表用於結果分析和報告
2. 結果評估工具 - 計算各類評估指標並生成評估報告
3. 檔案和路徑處理工具 - 提供檔案操作的輔助函數
4. 日誌配置工具 - 提供統一的日誌配置功能
5. 模型輔助工具 - 提供模型訓練和評估的輔助功能
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import explained_variance_score, max_error, median_absolute_error
from sklearn.model_selection import learning_curve

# 設定默認風格
plt.style.use('seaborn-whitegrid')
sns.set_palette("muted")

# 配置日誌
def setup_logger(log_file: Optional[str] = None, 
                level: int = logging.INFO) -> logging.Logger:
    """
    配置並返回日誌記錄器

    參數:
        log_file (str, optional): 日誌檔案路徑，默認為None，表示僅輸出到控制台
        level (int): 日誌級別，默認為INFO

    返回:
        logging.Logger: 配置好的日誌記錄器
    """
    # 創建日誌記錄器
    logger = logging.getLogger('solder_fatigue_prediction')
    logger.setLevel(level)
    
    # 創建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 創建控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果提供了日誌檔案路徑，添加檔案處理器
    if log_file:
        # 確保日誌目錄存在
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 檔案和路徑處理工具
def ensure_dir(directory: str) -> str:
    """
    確保目錄存在，如果不存在則創建它
    
    參數:
        directory (str): 目錄路徑
        
    返回:
        str: 目錄路徑
    """
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    return directory

def get_project_root() -> Path:
    """
    獲取專案根目錄
    
    返回:
        Path: 專案根目錄路徑
    """
    return Path(__file__).parent.parent

def save_json(data: Dict, filepath: str) -> None:
    """
    將字典數據保存為JSON檔案
    
    參數:
        data (Dict): 要保存的數據
        filepath (str): 保存路徑
    """
    # 確保目錄存在
    ensure_dir(os.path.dirname(os.path.abspath(filepath)))
    
    # 保存數據
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(filepath: str) -> Dict:
    """
    從JSON檔案載入字典數據
    
    參數:
        filepath (str): JSON檔案路徑
        
    返回:
        Dict: 載入的數據
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_pickle(obj: Any, filepath: str) -> None:
    """
    使用pickle保存物件
    
    參數:
        obj (Any): 要保存的物件
        filepath (str): 保存路徑
    """
    # 確保目錄存在
    ensure_dir(os.path.dirname(os.path.abspath(filepath)))
    
    # 保存物件
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str) -> Any:
    """
    使用pickle載入物件
    
    參數:
        filepath (str): pickle檔案路徑
        
    返回:
        Any: 載入的物件
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def create_timestamp_str() -> str:
    """
    創建時間戳記字串，用於檔案命名
    
    返回:
        str: 格式化的時間戳記字串
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# 數據可視化工具
def plot_feature_importance(feature_names: List[str], 
                          importance: np.ndarray, 
                          title: str = "特徵重要性",
                          filepath: Optional[str] = None,
                          top_n: int = 20,
                          figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    繪製特徵重要性圖表
    
    參數:
        feature_names (List[str]): 特徵名稱列表
        importance (numpy.ndarray): 特徵重要性數值
        title (str): 圖表標題
        filepath (str, optional): 保存路徑，默認為None表示不保存
        top_n (int): 顯示的最重要特徵數量
        figsize (Tuple[int, int]): 圖表尺寸
    """
    # 創建特徵重要性數據框
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # 按重要性排序
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 取前N個特徵
    if len(importance_df) > top_n:
        importance_df = importance_df.head(top_n)
    
    # 調整特徵名稱長度，確保圖表美觀
    importance_df['feature'] = importance_df['feature'].apply(
        lambda x: x if len(x) < 30 else x[:27] + '...'
    )
    
    # 繪製圖表
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='importance', y='feature', data=importance_df)
    
    # 設置標題和標籤
    plt.title(title, fontsize=16)
    plt.xlabel('重要性', fontsize=14)
    plt.ylabel('特徵', fontsize=14)
    plt.tight_layout()
    
    # 保存圖表
    if filepath:
        ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_correlation_heatmap(data: pd.DataFrame, 
                            title: str = "特徵相關性熱圖",
                            filepath: Optional[str] = None,
                            figsize: Tuple[int, int] = (14, 12),
                            mask_upper: bool = True,
                            cmap: str = 'coolwarm') -> None:
    """
    繪製相關性熱圖
    
    參數:
        data (pandas.DataFrame): 數據框
        title (str): 圖表標題
        filepath (str, optional): 保存路徑，默認為None表示不保存
        figsize (Tuple[int, int]): 圖表尺寸
        mask_upper (bool): 是否遮蓋上三角形
        cmap (str): 顏色映射
    """
    # 計算相關性矩陣
    corr = data.corr()
    
    # 設置掩碼（可選，用於隱藏上三角形）
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # 繪製相關性熱圖
    plt.figure(figsize=figsize)
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=False, 
               square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    
    # 設置標題
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # 保存圖表
    if filepath:
        ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_actual_vs_predicted(y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           title: str = "實際值 vs. 預測值",
                           filepath: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    繪製實際值與預測值對比圖
    
    參數:
        y_true (numpy.ndarray): 實際目標值
        y_pred (numpy.ndarray): 模型預測值
        title (str): 圖表標題
        filepath (str, optional): 保存路徑，默認為None表示不保存
        figsize (Tuple[int, int]): 圖表尺寸
    """
    # 創建數據框
    plot_df = pd.DataFrame({
        '實際值': y_true.flatten(),
        '預測值': y_pred.flatten()
    })
    
    # 計算評估指標
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # 繪製散點圖
    plt.figure(figsize=figsize)
    sns.scatterplot(x='實際值', y='預測值', data=plot_df, alpha=0.6)
    
    # 添加對角線（理想預測線）
    min_val = min(plot_df['實際值'].min(), plot_df['預測值'].min())
    max_val = max(plot_df['實際值'].max(), plot_df['預測值'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    # 添加評估指標文本
    plt.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nR²: {r2:.4f}',
            transform=plt.gca().transAxes, 
            fontsize=12, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 設置標題和標籤
    plt.title(title, fontsize=16)
    plt.xlabel('實際值', fontsize=14)
    plt.ylabel('預測值', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    # 保存圖表
    if filepath:
        ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_residuals(y_true: np.ndarray, 
                 y_pred: np.ndarray, 
                 title: str = "殘差分析",
                 filepath: Optional[str] = None,
                 figsize: Tuple[int, int] = (16, 6)) -> None:
    """
    繪製殘差分析圖
    
    參數:
        y_true (numpy.ndarray): 實際目標值
        y_pred (numpy.ndarray): 模型預測值
        title (str): 圖表標題
        filepath (str, optional): 保存路徑，默認為None表示不保存
        figsize (Tuple[int, int]): 圖表尺寸
    """
    # 計算殘差
    residuals = y_true.flatten() - y_pred.flatten()
    
    # 創建數據框
    plot_df = pd.DataFrame({
        '預測值': y_pred.flatten(),
        '殘差': residuals
    })
    
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 繪製殘差散點圖
    sns.scatterplot(x='預測值', y='殘差', data=plot_df, ax=ax1, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_title('殘差 vs. 預測值', fontsize=14)
    ax1.set_xlabel('預測值', fontsize=12)
    ax1.set_ylabel('殘差', fontsize=12)
    ax1.grid(True)
    
    # 繪製殘差分佈圖
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_title('殘差分佈', fontsize=14)
    ax2.set_xlabel('殘差', fontsize=12)
    ax2.set_ylabel('頻率', fontsize=12)
    ax2.grid(True)
    
    # 設置整體標題
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    # 保存圖表
    if filepath:
        ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_learning_curve(estimator: Any, 
                      X: np.ndarray, 
                      y: np.ndarray,
                      title: str = "學習曲線",
                      filepath: Optional[str] = None,
                      cv: int = 5,
                      n_jobs: int = -1,
                      train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
                      figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    繪製學習曲線
    
    參數:
        estimator (Any): 估計器/模型
        X (numpy.ndarray): 特徵數據
        y (numpy.ndarray): 目標變數
        title (str): 圖表標題
        filepath (str, optional): 保存路徑，默認為None表示不保存
        cv (int): 交叉驗證折數
        n_jobs (int): 並行作業數
        train_sizes (numpy.ndarray): 訓練集大小比例
        figsize (Tuple[int, int]): 圖表尺寸
    """
    # 計算學習曲線
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, 
        train_sizes=train_sizes,
        scoring='neg_mean_squared_error',
        shuffle=True
    )
    
    # 計算平均值和標準差
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # 轉換為RMSE
    train_scores_mean = np.sqrt(train_scores_mean)
    train_scores_std = np.sqrt(train_scores_std)
    test_scores_mean = np.sqrt(test_scores_mean)
    test_scores_std = np.sqrt(test_scores_std)
    
    # 繪製學習曲線
    plt.figure(figsize=figsize)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="訓練集RMSE")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="驗證集RMSE")
    
    # 設置標題和標籤
    plt.title(title, fontsize=16)
    plt.xlabel("訓練樣本數", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    # 保存圖表
    if filepath:
        ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_time_series_data(time_points: List[int], 
                        features: Dict[str, List[float]],
                        title: str = "時間序列特徵分析",
                        filepath: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    繪製時間序列特徵分析圖
    
    參數:
        time_points (List[int]): 時間點列表
        features (Dict[str, List[float]]): 特徵名稱和時間序列數據的字典
        title (str): 圖表標題
        filepath (str, optional): 保存路徑，默認為None表示不保存
        figsize (Tuple[int, int]): 圖表尺寸
    """
    # 創建圖表
    plt.figure(figsize=figsize)
    
    # 繪製各特徵的時間序列
    for feature_name, values in features.items():
        if len(values) != len(time_points):
            continue
        plt.plot(time_points, values, 'o-', linewidth=2, label=feature_name)
    
    # 設置標題和標籤
    plt.title(title, fontsize=16)
    plt.xlabel("時間點", fontsize=14)
    plt.ylabel("數值", fontsize=14)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    # 保存圖表
    if filepath:
        ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_model_comparison(model_names: List[str], 
                        metrics: Dict[str, Dict[str, float]],
                        metric_name: str = "rmse",
                        title: str = "模型性能比較",
                        filepath: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    繪製模型比較圖表
    
    參數:
        model_names (List[str]): 模型名稱列表
        metrics (Dict[str, Dict[str, float]]): 模型評估指標字典
        metric_name (str): 要比較的指標名稱
        title (str): 圖表標題
        filepath (str, optional): 保存路徑，默認為None表示不保存
        figsize (Tuple[int, int]): 圖表尺寸
    """
    # 提取指定指標的值和標準差
    values = []
    errors = []
    
    for model in model_names:
        if model in metrics and metric_name in metrics[model]:
            values.append(metrics[model][metric_name])
            # 檢查標準差是否存在
            if f"{metric_name}_std" in metrics[model]:
                errors.append(metrics[model][f"{metric_name}_std"])
            else:
                errors.append(0)
    
    # 創建數據框
    df = pd.DataFrame({
        '模型': model_names,
        '指標值': values,
        '標準差': errors
    })
    
    # 按指標值排序
    df = df.sort_values('指標值')
    
    # 繪製條形圖
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='指標值', y='模型', data=df, 
                    xerr=df['標準差'], 
                    palette='viridis')
    
    # 設置標題和標籤
    plt.title(title, fontsize=16)
    plt.xlabel(f"{metric_name.upper()}", fontsize=14)
    plt.ylabel("模型", fontsize=14)
    plt.tight_layout()
    
    # 保存圖表
    if filepath:
        ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.show()

# 結果評估工具
def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    計算回歸評估指標
    
    參數:
        y_true (numpy.ndarray): 實際目標值
        y_pred (numpy.ndarray): 模型預測值
        
    返回:
        Dict[str, float]: 包含各類評估指標的字典
    """
    # 確保輸入為一維數組
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # 計算各類評估指標
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    explained_variance = explained_variance_score(y_true, y_pred)
    max_err = max_error(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    
    # 計算平均絕對百分比誤差 (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # 計算均方根對數誤差 (RMSLE)
    # 對於負值，先進行處理
    y_true_positive = np.maximum(y_true, 1e-10)
    y_pred_positive = np.maximum(y_pred, 1e-10)
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_true_positive), np.log1p(y_pred_positive)))
    
    # 返回評估指標字典
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'explained_variance': explained_variance,
        'max_error': max_err,
        'median_absolute_error': median_ae,
        'mape': mape,
        'rmsle': rmsle
    }

def generate_evaluation_report(y_true: np.ndarray, 
                            y_pred: np.ndarray, 
                            model_name: str = "未命名模型",
                            filepath: Optional[str] = None) -> Dict[str, float]:
    """
    生成模型評估報告
    
    參數:
        y_true (numpy.ndarray): 實際目標值
        y_pred (numpy.ndarray): 模型預測值
        model_name (str): 模型名稱
        filepath (str, optional): 保存報告的文件路徑，默認為None表示不保存
        
    返回:
        Dict[str, float]: 評估指標字典
    """
    # 計算評估指標
    metrics = calculate_regression_metrics(y_true, y_pred)
    
    # 打印報告
    print(f"\n===== {model_name} 評估報告 =====")
    print(f"均方誤差 (MSE):             {metrics['mse']:.6f}")
    print(f"均方根誤差 (RMSE):           {metrics['rmse']:.6f}")
    print(f"平均絕對誤差 (MAE):          {metrics['mae']:.6f}")
    print(f"決定係數 (R²):              {metrics['r2']:.6f}")
    print(f"解釋方差分數:                {metrics['explained_variance']:.6f}")
    print(f"最大誤差:                   {metrics['max_error']:.6f}")
    print(f"中位數絕對誤差:              {metrics['median_absolute_error']:.6f}")
    print(f"平均絕對百分比誤差 (MAPE):    {metrics['mape']:.6f}%")
    print(f"均方根對數誤差 (RMSLE):       {metrics['rmsle']:.6f}")
    
    # 如果提供了文件路徑，保存報告
    if filepath:
        # 確保目錄存在
        ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        
        # 創建報告內容
        report = {
            'model_name': model_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics': metrics
        }
        
        # 保存為JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        
        print(f"\n評估報告已保存到: {filepath}")
    
    return metrics

def compare_models(models_metrics: Dict[str, Dict[str, float]], 
                 metric_names: List[str] = ['rmse', 'r2'],
                 filepath: Optional[str] = None) -> pd.DataFrame:
    """
    比較多個模型的性能
    
    參數:
        models_metrics (Dict[str, Dict[str, float]]): 模型名稱到評估指標的映射字典
        metric_names (List[str]): 要比較的指標名稱列表
        filepath (str, optional): 保存比較結果的文件路徑，默認為None表示不保存
        
    返回:
        pandas.DataFrame: 模型比較結果數據框
    """
    # 創建空列表存儲比較結果
    results = []
    
    # 遍歷每個模型的指標
    for model_name, metrics in models_metrics.items():
        result = {'模型': model_name}
        
        # 添加每個指定的指標
        for metric in metric_names:
            if metric in metrics:
                # 對特定指標進行格式化
                if metric.lower() in ['mse', 'rmse', 'mae', 'rmsle']:
                    result[metric.upper()] = f"{metrics[metric]:.6f}"
                elif metric.lower() == 'r2':
                    result['R²'] = f"{metrics[metric]:.6f}"
                elif metric.lower() == 'mape':
                    result['MAPE (%)'] = f"{metrics[metric]:.2f}%"
                else:
                    result[metric] = f"{metrics[metric]:.6f}"
        
        results.append(result)
    
    # 創建比較結果數據框
    comparison_df = pd.DataFrame(results)
    
    # 打印比較結果
    print("\n===== 模型比較 =====")
    print(comparison_df.to_string(index=False))
    
    # 如果提供了文件路徑，保存比較結果
    if filepath:
        # 確保目錄存在
        ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        
        # 保存為CSV
        comparison_df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"\n比較結果已保存到: {filepath}")
    
    return comparison_df

# 模型輔助工具
def get_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    計算最佳閾值
    
    參數:
        y_true (numpy.ndarray): 實際目標值 (二分類問題中的0/1標籤)
        y_score (numpy.ndarray): 模型預測的分數或概率
        
    返回:
        float: 最佳閾值
    """
    from sklearn.metrics import roc_curve, f1_score
    
    # 計算ROC曲線上的不同閾值
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # 嘗試不同閾值，選擇F1分數最高的閾值
    optimal_threshold = 0.5  # 默認閾值
    best_f1 = 0
    
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = threshold
    
    return optimal_threshold

def bootstrap_confidence_interval(y_true: np.ndarray, 
                               y_pred: np.ndarray, 
                               metric_func: Callable[[np.ndarray, np.ndarray], float],
                               n_bootstraps: int = 1000,
                               confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    使用bootstrap方法計算評估指標的置信區間
    
    參數:
        y_true (numpy.ndarray): 實際目標值
        y_pred (numpy.ndarray): 模型預測值
        metric_func (Callable): 計算評估指標的函數，接受y_true和y_pred作為參數
        n_bootstraps (int): bootstrap樣本數量
        confidence_level (float): 置信水平，範圍(0, 1)
        
    返回:
        Tuple[float, float]: 置信區間下界和上界
    """
    # 確保輸入為一維數組
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # 樣本數量
    n_samples = len(y_true)
    
    # 存儲bootstrap樣本的指標結果
    bootstrap_results = []
    
    # 生成bootstrap樣本並計算指標
    for _ in range(n_bootstraps):
        # 隨機抽樣（有放回）
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # 計算當前bootstrap樣本的指標
        bootstrap_metric = metric_func(y_true[indices], y_pred[indices])
        bootstrap_results.append(bootstrap_metric)
    
    # 計算置信區間
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_results, lower_percentile)
    upper_bound = np.percentile(bootstrap_results, upper_percentile)
    
    return lower_bound, upper_bound

def create_feature_groups(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    根據特徵名稱自動將特徵分組
    
    參數:
        feature_names (List[str]): 特徵名稱列表
        
    返回:
        Dict[str, List[str]]: 特徵組名稱到特徵列表的映射
    """
    # 定義可能的特徵組關鍵字
    group_keywords = {
        '幾何特徵': ['die', 'stud', 'mold', 'pcb', 'geometry', 'dimension', 'size', 'height', 'width', 'thickness'],
        '翹曲特徵': ['warpage', 'warp', 'bend', 'deflection', 'deformation'],
        '應力特徵': ['stress', 'nlplwk', 'pressure', 'force', 'load'],
        '應變特徵': ['strain', 'deformation', 'equi', 'equivalent', 'plastic', 'elastic'],
        '時間序列特徵': ['3600', '7200', '10800', '14400', 'time', 'series', 'temporal'],
        '材料特徵': ['material', 'property', 'modulus', 'strength', 'coefficient', 'thermal'],
        '交互特徵': ['interaction', 'product', 'ratio', 'combined'],
        '統計特徵': ['mean', 'average', 'std', 'min', 'max', 'median', 'var', 'sum', 'count']
    }
    
    # 初始化特徵組
    feature_groups = {group: [] for group in group_keywords.keys()}
    ungrouped_features = []
    
    # 將特徵分配到相應的組
    for feature in feature_names:
        feature_lower = feature.lower()
        assigned = False
        
        for group, keywords in group_keywords.items():
            if any(keyword in feature_lower for keyword in keywords):
                feature_groups[group].append(feature)
                assigned = True
                break
        
        if not assigned:
            ungrouped_features.append(feature)
    
    # 添加未分組的特徵
    if ungrouped_features:
        feature_groups['其他特徵'] = ungrouped_features
    
    # 移除空組
    feature_groups = {k: v for k, v in feature_groups.items() if v}
    
    return feature_groups

def generate_physical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    根據物理模型生成特徵
    
    參數:
        data (pandas.DataFrame): 原始數據框
        
    返回:
        pandas.DataFrame: 添加物理特徵後的數據框
    """
    # 複製數據避免修改原始數據
    df = data.copy()
    
    # 檢測相關列的存在
    structure_cols = ['Die', 'stud', 'mold', 'PCB']
    stress_cols = [col for col in df.columns if 'NLPLWK' in col]
    strain_cols = [col for col in df.columns if 'Strain' in col]
    warpage_cols = [col for col in df.columns if 'warpage' in col.lower()]
    
    # 1. 基於變形能理論的特徵
    if stress_cols and strain_cols:
        # 計算應力應變比（類似楊氏模量）
        for stress_col in stress_cols:
            for strain_col in strain_cols:
                col_name = f"energy_density_{stress_col}_{strain_col}"
                df[col_name] = df[stress_col] * df[strain_col] / 2  # 彈性應變能密度
    
    # 2. 基於熱力學的特徵
    if warpage_cols and structure_cols:
        if all(col in df.columns for col in ['Total_warpage', 'PCB', 'Die']):
            # 計算規一化翹曲
            df['normalized_warpage'] = df['Total_warpage'] / (df['PCB'] + df['Die'])
            
            # 計算翹曲曲率 (近似)
            if 'PCB' in df.columns:
                df['warpage_curvature'] = 2 * df['Total_warpage'] / (df['PCB'] ** 2)
    
    # 3. 基於疲勞理論的特徵 (Coffin-Manson模型相關)
    if 'Acc_Equi_Strain_max' in df.columns:
        # 典型的Coffin-Manson模型: Nf = C * (Δεp)^(-n)
        # 其中C和n是材料常數，Δεp是塑性應變幅度
        
        # 首先定義幾個典型的C和n值範圍
        C_values = [0.5, 1.0, 1.5]
        n_values = [0.5, 0.6, 0.7]
        
        for C in C_values:
            for n in n_values:
                col_name = f"CM_pred_C{C}_n{n}"
                df[col_name] = C * (df['Acc_Equi_Strain_max'] ** (-n))
    
    # 4. 基於應力集中的特徵
    if all(col in df.columns for col in structure_cols):
        # 計算應力集中因子的近似
        df['stress_concentration_factor'] = df['Die'] / df['mold']
    
    return df

def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """
    檢查數據質量，包括缺失值、異常值、分佈情況等
    
    參數:
        data (pandas.DataFrame): 數據框
        
    返回:
        Dict[str, Any]: 數據質量檢查結果
    """
    # 初始化結果字典
    results = {
        'missing_values': {},
        'outliers': {},
        'distribution': {},
        'correlation': {},
        'warnings': []
    }
    
    # 1. 檢查缺失值
    missing_count = data.isnull().sum()
    missing_percent = (missing_count / len(data) * 100).round(2)
    
    results['missing_values'] = {
        'count': missing_count.to_dict(),
        'percent': missing_percent.to_dict(),
        'total_rows_with_missing': data.isnull().any(axis=1).sum(),
        'total_percent_rows_with_missing': (data.isnull().any(axis=1).sum() / len(data) * 100).round(2)
    }
    
    # 發出警告：缺失值比例過高的特徵
    high_missing_cols = missing_percent[missing_percent > 5].index.tolist()
    if high_missing_cols:
        results['warnings'].append(f"以下特徵的缺失值比例超過5%: {', '.join(high_missing_cols)}")
    
    # 2. 檢查異常值 (使用IQR方法)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    outliers_dict = {}
    
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
        outliers_count = len(outliers)
        outliers_percent = (outliers_count / len(data) * 100).round(2)
        
        outliers_dict[col] = {
            'count': outliers_count,
            'percent': outliers_percent,
            'min': data[col].min(),
            'max': data[col].max(),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    results['outliers'] = outliers_dict
    
    # 發出警告：異常值比例過高的特徵
    high_outliers_cols = [col for col, info in outliers_dict.items() if info['percent'] > 10]
    if high_outliers_cols:
        results['warnings'].append(f"以下特徵的異常值比例超過10%: {', '.join(high_outliers_cols)}")
    
    # 3. 檢查分佈情況
    distribution_dict = {}
    
    for col in numeric_cols:
        skewness = data[col].skew()
        kurtosis = data[col].kurtosis()
        
        distribution_dict[col] = {
            'mean': data[col].mean(),
            'median': data[col].median(),
            'std': data[col].std(),
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
        # 高度偏斜的特徵可能需要轉換
        if abs(skewness) > 2:
            results['warnings'].append(f"特徵 '{col}' 具有高度偏斜 (skewness={skewness:.2f})，可能需要進行轉換")
    
    results['distribution'] = distribution_dict
    
    # 4. 檢查相關性
    # 計算數值特徵的相關矩陣
    corr_matrix = data[numeric_cols].corr().abs()
    
    # 查找高度相關的特徵對
    high_corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            if corr_matrix.iloc[i, j] > 0.9:  # 相關係數閾值
                high_corr_pairs.append({
                    'feature1': numeric_cols[i],
                    'feature2': numeric_cols[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    results['correlation']['high_correlation_pairs'] = high_corr_pairs
    
    # 警告高度相關的特徵對
    if high_corr_pairs:
        results['warnings'].append(f"發現 {len(high_corr_pairs)} 對高度相關的特徵 (r > 0.9)，可能導致多重共線性問題")
    
    # 5. 檢查數據集大小是否適合機器學習
    if len(data) < 50:
        results['warnings'].append(f"數據集僅有 {len(data)} 個樣本，這對於許多機器學習模型來說可能不足")
    
    if len(numeric_cols) > len(data) / 5:
        results['warnings'].append(f"特徵數量 ({len(numeric_cols)}) 相對於樣本數量 ({len(data)}) 較高，可能導致過擬合")
    
    return results

# 主函數示例
if __name__ == "__main__":
    # 設置日誌
    logger = setup_logger(log_file="logs/utils_test.log")
    logger.info("工具函數模組測試開始")
    
    try:
        # 創建測試數據
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y_true = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * X[:, 3] - X[:, 4] + np.random.randn(n_samples) * 0.5
        
        # 模擬預測結果
        y_pred = y_true + np.random.randn(n_samples) * 1.2
        
        # 創建特徵名稱
        feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        
        # 測試評估指標計算
        logger.info("測試評估指標計算")
        metrics = calculate_regression_metrics(y_true, y_pred)
        print(metrics)
        
        # 測試評估報告生成
        logger.info("測試評估報告生成")
        generate_evaluation_report(y_true, y_pred, model_name="測試模型", 
                                 filepath="output/test_model_report.json")
        
        # 測試作圖功能
        logger.info("測試作圖功能")
        plot_actual_vs_predicted(y_true, y_pred, filepath="figures/test_actual_vs_pred.png")
        plot_residuals(y_true, y_pred, filepath="figures/test_residuals.png")
        
        # 測試bootstrap置信區間
        logger.info("測試bootstrap置信區間")
        rmse_func = lambda y_t, y_p: np.sqrt(mean_squared_error(y_t, y_p))
        lower, upper = bootstrap_confidence_interval(y_true, y_pred, rmse_func, n_bootstraps=100)
        print(f"RMSE 95% 置信區間: [{lower:.4f}, {upper:.4f}]")
        
        logger.info("工具函數模組測試完成")
        
    except Exception as e:
        logger.error(f"測試過程中出錯: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())