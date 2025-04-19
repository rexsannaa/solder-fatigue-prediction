#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
evaluation.py - 模型評估模組
本模組提供了全面的模型評估功能，用於評估銲錫接點疲勞壽命預測模型的性能，
包括性能指標計算、結果視覺化、模型比較和敏感性分析。

主要功能:
1. 計算多種模型評估指標
2. 生成評估報告與結果視覺化
3. 不同模型的性能比較
4. 預測結果的敏感性分析
5. 模型輸出的不確定性量化
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error, explained_variance_score
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.inspection import permutation_importance
import logging
import json
from datetime import datetime
import pickle
# 設置中文字體支持
import matplotlib as mpl
import platform

# 根據不同操作系統設置不同的字體
system = platform.system()
if system == 'Windows':
    # Windows系統使用微軟雅黑或新細明體
    mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'NSimSun', 'Arial Unicode MS']
elif system == 'Darwin':  # macOS
    # macOS使用蘋方或儷黑Pro
    mpl.rcParams['font.sans-serif'] = ['PingFang TC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS']
else:  # Linux/其他
    # Linux使用文泉驛或思源黑體
    mpl.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK TC', 'Noto Sans TC', 'Arial Unicode MS']

# 確保負號正確顯示
mpl.rcParams['axes.unicode_minus'] = False

# 設置全局字體大小
mpl.rcParams['font.size'] = 12
# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    模型評估器，提供全面的模型評估功能
    """
    
    def __init__(self, output_dir: str = 'figures'):
        """
        初始化模型評估器
        
        參數:
            output_dir (str): 評估結果輸出目錄
        """
        self.output_dir = output_dir
        self.metrics_history = {}
        self.comparison_results = {}
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        model_name: str = None, log_metrics: bool = True) -> Dict[str, float]:
        """
        計算多種評估指標
        
        參數:
            y_true (numpy.ndarray): 真實值
            y_pred (numpy.ndarray): 預測值
            model_name (str, optional): 模型名稱，用於記錄
            log_metrics (bool): 是否記錄指標
            
        返回:
            Dict[str, float]: 評估指標字典
        """
        # 確保輸入是一維數組
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)
        
        # 計算各種評估指標
        metrics = {}
        
        # 均方誤差及其變種
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # 平均絕對誤差及其變種
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # 平均絕對百分比誤差
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
        except:
            # 處理低於sklearn 0.24版本的相容性
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-10, np.abs(y_true)))) * 100
        
        # 決定係數
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # 解釋方差分數
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
        
        # 最大誤差
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        
        # 如果提供了模型名稱，記錄評估指標
        if model_name and log_metrics:
            self.metrics_history[model_name] = metrics
            
            if log_metrics:
                logger.info(f"模型 {model_name} 的評估指標:")
                logger.info(f"  RMSE = {metrics['rmse']:.4f}")
                logger.info(f"  MAE = {metrics['mae']:.4f}")
                logger.info(f"  MAPE = {metrics['mape']:.2f}%")
                logger.info(f"  R² = {metrics['r2']:.4f}")
                logger.info(f"  最大誤差 = {metrics['max_error']:.4f}")
        
        return metrics
    
    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str, 
                               save_fig: bool = True, 
                               show_fig: bool = True) -> plt.Figure:
        """
        繪製實際值與預測值的對比圖
        
        參數:
            y_true (numpy.ndarray): 真實值
            y_pred (numpy.ndarray): 預測值
            model_name (str): 模型名稱
            save_fig (bool): 是否保存圖像
            show_fig (bool): 是否顯示圖像
            
        返回:
            matplotlib.figure.Figure: 圖像對象
        """
        # 確保輸入是一維數組
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)
        
        # 計算評估指標
        metrics = self.calculate_metrics(y_true, y_pred, log_metrics=False)
        
        # 創建圖像
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 繪製散點圖
        scatter = ax.scatter(y_true, y_pred, alpha=0.7, s=50, c='#1f77b4', edgecolors='k', linewidths=0.5)
        
        # 添加完美預測線
        max_val = max(np.max(y_true), np.max(y_pred))
        min_val = min(np.min(y_true), np.min(y_pred))
        buffer = (max_val - min_val) * 0.05  # 添加5%的緩衝區
        line_vals = [min_val - buffer, max_val + buffer]
        ax.plot(line_vals, line_vals, 'r--', linewidth=2, label='完美預測線')
        
        # 設置坐標軸範圍
        ax.set_xlim(min_val - buffer, max_val + buffer)
        ax.set_ylim(min_val - buffer, max_val + buffer)
        
        # 添加標籤和標題
        ax.set_xlabel('實際疲勞壽命 (循環次數)', fontsize=14)
        ax.set_ylabel('預測疲勞壽命 (循環次數)', fontsize=14)
        ax.set_title(f'{model_name} 模型預測結果對比', fontsize=16)
        
        # 添加網格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加評估指標文本
        metrics_text = (f"RMSE = {metrics['rmse']:.2f}\n"
                       f"MAE = {metrics['mae']:.2f}\n"
                       f"MAPE = {metrics['mape']:.2f}%\n"
                       f"R² = {metrics['r2']:.4f}")
        
        # 在圖像右下角添加文本框
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        # 添加圖例
        ax.legend(loc='upper left')
        
        # 調整布局
        fig.tight_layout()
        
        # 保存圖像
        if save_fig:
            filename = f"actual_vs_predicted_{model_name.replace(' ', '_').lower()}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至 {filepath}")
        
        # 顯示圖像
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_permutation_importance(self, model: Any, X: np.ndarray, y: np.ndarray, 
                                 feature_names: List[str], model_name: str = None,
                                 top_n: int = None, n_repeats: int = 10, random_state: int = 42,
                                 save_fig: bool = True, 
                                 show_fig: bool = True) -> plt.Figure:
        """
        繪製置換重要性圖，這是一種與模型無關的特徵重要性評估方法
        
        參數:
            model (Any): 訓練好的模型
            X (numpy.ndarray): 特徵數據
            y (numpy.ndarray): 目標變數
            feature_names (List[str]): 特徵名稱列表
            model_name (str, optional): 模型名稱
            top_n (int, optional): 只顯示前N個重要特徵
            n_repeats (int): 置換重複次數
            random_state (int): 隨機種子
            save_fig (bool): 是否保存圖像
            show_fig (bool): 是否顯示圖像
            
        返回:
            matplotlib.figure.Figure: 圖像對象
        """
        # 計算置換重要性
        try:
            perm_importance = permutation_importance(model, X, y, n_repeats=n_repeats,
                                                 random_state=random_state)
        except Exception as e:
            logger.error(f"計算置換重要性時出錯: {str(e)}")
            logger.info("嘗試使用自定義方法計算置換重要性...")
            
            # 如果sklearn的置換重要性失敗，使用自定義實現
            perm_importance = self._custom_permutation_importance(model, X, y, n_repeats, random_state)
        
        # 將特徵重要性與特徵名稱配對
        features = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        })
        
        # 按重要性降序排序
        features = features.sort_values('importance_mean', ascending=False)
        
        # 只取前N個特徵
        if top_n is not None and top_n < len(features):
            features = features.head(top_n)
        
        # 創建圖像
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
        
        # 繪製條形圖及誤差線
        bars = ax.barh(features['feature'], features['importance_mean'], 
                     xerr=features['importance_std'], 
                     color='skyblue', edgecolor='k', alpha=0.8,
                     error_kw={'ecolor': 'k', 'capsize': 5, 'elinewidth': 1})
        
        # 為每個條形添加數值標籤
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width * 1.01  # 稍微偏離條形的末端
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                  va='center', ha='left', fontsize=10)
        
        # 添加標籤和標題
        ax.set_xlabel('置換重要性', fontsize=14)
        ax.set_ylabel('特徵', fontsize=14)
        title = f'特徵置換重要性' if model_name is None else f'{model_name} 模型特徵置換重要性'
        ax.set_title(title, fontsize=16)
        
        # 添加網格
        ax.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        # 調整布局
        fig.tight_layout()
        
        # 保存圖像
        if save_fig:
            model_str = "" if model_name is None else f"_{model_name.replace(' ', '_').lower()}"
            filename = f"permutation_importance{model_str}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至 {filepath}")
        
        # 顯示圖像
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _custom_permutation_importance(self, model: Any, X: np.ndarray, y: np.ndarray, 
                                      n_repeats: int, random_state: int) -> Any:
        """
        自定義實現的置換重要性，當sklearn的實現失敗時使用
        
        參數:
            model (Any): 訓練好的模型
            X (numpy.ndarray): 特徵數據
            y (numpy.ndarray): 目標變數
            n_repeats (int): 置換重複次數
            random_state (int): 隨機種子
            
        返回:
            Any: 類似於sklearn的置換重要性結果
        """
        class PermutationImportanceResult:
            def __init__(self, importances_mean, importances_std, importances):
                self.importances_mean = importances_mean
                self.importances_std = importances_std
                self.importances = importances
        
        # 設置隨機種子
        np.random.seed(random_state)
        
        # 獲取基準分數
        baseline_score = r2_score(y, model.predict(X))
        
        # 初始化重要性數組
        n_features = X.shape[1]
        importances = np.zeros((n_repeats, n_features))
        
        # 對每個特徵進行置換
        for feature_idx in range(n_features):
            for repeat_idx in range(n_repeats):
                # 複製原始數據
                X_permuted = X.copy()
                
                # 隨機置換特定特徵
                X_permuted[:, feature_idx] = np.random.permutation(X[:, feature_idx])
                
                # 計算置換後的分數
                permuted_score = r2_score(y, model.predict(X_permuted))
                
                # 計算重要性 (基準分數 - 置換後分數)
                importances[repeat_idx, feature_idx] = baseline_score - permuted_score
        
        # 計算平均值和標準差
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        
        # 創建結果對象
        result = PermutationImportanceResult(importances_mean, importances_std, importances)
        
        return result
    
    def perform_sensitivity_analysis(self, model: Any, X: np.ndarray, feature_names: List[str],
                                    feature_idx: int, feature_range: np.ndarray = None, 
                                    num_points: int = 50, model_name: str = None,
                                    save_fig: bool = True, 
                                    show_fig: bool = True) -> plt.Figure:
        """
        對指定特徵進行敏感性分析，其他特徵保持不變
        
        參數:
            model (Any): 訓練好的模型
            X (numpy.ndarray): 特徵數據，用一個樣本作為基準
            feature_names (List[str]): 特徵名稱列表
            feature_idx (int): 要分析的特徵索引
            feature_range (numpy.ndarray, optional): 特徵值範圍，如果為None則自動生成範圍
            num_points (int): 分析點數
            model_name (str, optional): 模型名稱
            save_fig (bool): 是否保存圖像
            show_fig (bool): 是否顯示圖像
            
        返回:
            matplotlib.figure.Figure: 圖像對象
        """
        # 使用數據的中值樣本作為基準
        X_base = np.median(X, axis=0).reshape(1, -1)
        
        # 獲取特徵名稱
        feature_name = feature_names[feature_idx]
        
        # 如果未提供特徵範圍，自動生成範圍
        if feature_range is None:
            # 獲取特徵的最小值和最大值
            feature_min = np.min(X[:, feature_idx])
            feature_max = np.max(X[:, feature_idx])
            
            # 擴展範圍以便更好地顯示趨勢
            range_width = feature_max - feature_min
            feature_min = max(0, feature_min - range_width * 0.2)  # 避免負值（如適用）
            feature_max = feature_max + range_width * 0.2
            
            # 生成均勻分佈的特徵值
            feature_range = np.linspace(feature_min, feature_max, num_points)
        
        # 為每個特徵值創建一個樣本
        X_samples = np.tile(X_base, (len(feature_range), 1))
        X_samples[:, feature_idx] = feature_range
        
        # 進行預測
        y_pred = model.predict(X_samples)
        
        # 創建圖像
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 繪製敏感性曲線
        ax.plot(feature_range, y_pred, 'o-', color='blue', linewidth=2, markersize=5)
        
        # 添加標記表示當前特徵的中值
        median_value = np.median(X[:, feature_idx])
        median_prediction = model.predict(np.array([X_base[0].copy()]).reshape(1, -1))[0]
        ax.scatter([median_value], [median_prediction], color='red', s=100, zorder=5, 
                 label=f'中值 ({median_value:.4f}, {median_prediction:.4f})')
        
        # 添加標籤和標題
        ax.set_xlabel(f'特徵值: {feature_name}', fontsize=14)
        ax.set_ylabel('預測疲勞壽命', fontsize=14)
        title = f'{feature_name} 敏感性分析' if model_name is None else f'{model_name} 模型 {feature_name} 敏感性分析'
        ax.set_title(title, fontsize=16)
        
        # 添加網格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加圖例
        ax.legend(loc='best')
        
        # 調整布局
        fig.tight_layout()
        
        # 保存圖像
        if save_fig:
            model_str = "" if model_name is None else f"_{model_name.replace(' ', '_').lower()}"
            feature_str = feature_name.replace(' ', '_').lower()
            filename = f"sensitivity_{feature_str}{model_str}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至 {filepath}")
        
        # 顯示圖像
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def multi_feature_sensitivity_analysis(self, model: Any, X: np.ndarray, feature_names: List[str],
                                         feature_indices: List[int], model_name: str = None,
                                         num_points: int = 20, save_fig: bool = True, 
                                         show_fig: bool = True) -> Dict[str, plt.Figure]:
        """
        對多個指定特徵進行敏感性分析，生成多個圖像
        
        參數:
            model (Any): 訓練好的模型
            X (numpy.ndarray): 特徵數據
            feature_names (List[str]): 特徵名稱列表
            feature_indices (List[int]): 要分析的特徵索引列表
            model_name (str, optional): 模型名稱
            num_points (int): 每個特徵的分析點數
            save_fig (bool): 是否保存圖像
            show_fig (bool): 是否顯示圖像
            
        返回:
            Dict[str, matplotlib.figure.Figure]: 特徵名稱與其敏感性分析圖的字典
        """
        # 存儲每個特徵的敏感性分析圖
        sensitivity_figures = {}
        
        # 對每個特徵進行敏感性分析
        for feature_idx in feature_indices:
            # 確保特徵索引有效
            if feature_idx < 0 or feature_idx >= len(feature_names):
                logger.warning(f"特徵索引 {feature_idx} 超出範圍，跳過")
                continue
            
            # 進行敏感性分析
            fig = self.perform_sensitivity_analysis(
                model, X, feature_names, feature_idx,
                feature_range=None, num_points=num_points,
                model_name=model_name, save_fig=save_fig, show_fig=show_fig
            )
            
            # 存儲結果
            feature_name = feature_names[feature_idx]
            sensitivity_figures[feature_name] = fig
        
        return sensitivity_figures
    
    def plot_two_feature_interaction(self, model: Any, X: np.ndarray, feature_names: List[str],
                                  feature_idx1: int, feature_idx2: int, 
                                  resolution: int = 20, model_name: str = None,
                                  save_fig: bool = True, 
                                  show_fig: bool = True) -> plt.Figure:
        """
        繪製兩個特徵之間的交互影響圖
        
        參數:
            model (Any): 訓練好的模型
            X (numpy.ndarray): 特徵數據
            feature_names (List[str]): 特徵名稱列表
            feature_idx1 (int): 第一個特徵的索引
            feature_idx2 (int): 第二個特徵的索引
            resolution (int): 網格解析度
            model_name (str, optional): 模型名稱
            save_fig (bool): 是否保存圖像
            show_fig (bool): 是否顯示圖像
            
        返回:
            matplotlib.figure.Figure: 圖像對象
        """
        # 使用數據的中值樣本作為基準
        X_base = np.median(X, axis=0).reshape(1, -1)
        
        # 獲取特徵名稱
        feature_name1 = feature_names[feature_idx1]
        feature_name2 = feature_names[feature_idx2]
        
        # 獲取特徵的範圍
        feature_min1 = np.min(X[:, feature_idx1])
        feature_max1 = np.max(X[:, feature_idx1])
        feature_min2 = np.min(X[:, feature_idx2])
        feature_max2 = np.max(X[:, feature_idx2])
        
        # 擴展範圍
        range_width1 = feature_max1 - feature_min1
        range_width2 = feature_max2 - feature_min2
        feature_min1 = max(0, feature_min1 - range_width1 * 0.1)
        feature_max1 = feature_max1 + range_width1 * 0.1
        feature_min2 = max(0, feature_min2 - range_width2 * 0.1)
        feature_max2 = feature_max2 + range_width2 * 0.1
        
        # 創建二維網格
        x1_grid = np.linspace(feature_min1, feature_max1, resolution)
        x2_grid = np.linspace(feature_min2, feature_max2, resolution)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        
        # 為網格中的每個點創建樣本
        X_samples = np.tile(X_base, (resolution * resolution, 1))
        X_samples[:, feature_idx1] = X1.flatten()
        X_samples[:, feature_idx2] = X2.flatten()
        
        # 進行預測
        y_pred = model.predict(X_samples).reshape(resolution, resolution)
        
        # 創建圖像
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 繪製熱力圖
        contour = ax.contourf(X1, X2, y_pred, cmap='viridis', levels=50)
        
        # 添加顏色條
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('預測疲勞壽命', fontsize=12)
        
        # 添加等高線
        contour_lines = ax.contour(X1, X2, y_pred, colors='white', alpha=0.5, levels=10)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        
        # 添加標記表示當前特徵的中值
        median_value1 = np.median(X[:, feature_idx1])
        median_value2 = np.median(X[:, feature_idx2])
        ax.scatter([median_value1], [median_value2], color='red', s=100, marker='x', 
                 label=f'中值點 ({median_value1:.2f}, {median_value2:.2f})')
        
        # 添加標籤和標題
        ax.set_xlabel(feature_name1, fontsize=14)
        ax.set_ylabel(feature_name2, fontsize=14)
        title = f'{feature_name1} 和 {feature_name2} 交互影響' if model_name is None else f'{model_name} 模型 {feature_name1} 和 {feature_name2} 交互影響'
        ax.set_title(title, fontsize=16)
        
        # 添加圖例
        ax.legend(loc='best')
        
        # 調整布局
        fig.tight_layout()
        
        # 保存圖像
        if save_fig:
            model_str = "" if model_name is None else f"_{model_name.replace(' ', '_').lower()}"
            feature_str1 = feature_name1.replace(' ', '_').lower()
            feature_str2 = feature_name2.replace(' ', '_').lower()
            filename = f"interaction_{feature_str1}_{feature_str2}{model_str}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至 {filepath}")
        
        # 顯示圖像
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                model_name: str, feature_importance: pd.DataFrame = None,
                                report_dir: str = None) -> str:
        """
        生成評估報告，包括性能指標和可選的圖像
        
        參數:
            y_true (numpy.ndarray): 真實值
            y_pred (numpy.ndarray): 預測值
            model_name (str): 模型名稱
            feature_importance (pandas.DataFrame, optional): 特徵重要性數據框
            report_dir (str, optional): 報告輸出目錄，默認為self.output_dir
            
        返回:
            str: 報告檔案路徑
        """
        # 如果未指定報告目錄，使用默認輸出目錄
        if report_dir is None:
            report_dir = self.output_dir
        
        # 確保目錄存在
        os.makedirs(report_dir, exist_ok=True)
        
        # 計算評估指標
        metrics = self.calculate_metrics(y_true, y_pred, model_name, log_metrics=False)
        
        # 創建報告字典
        report = {
            'model_name': model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics
        }
        
        # 添加特徵重要性（如果提供）
        if feature_importance is not None:
            report['feature_importance'] = feature_importance.to_dict(orient='records')
        
        # 生成圖像
        self.plot_actual_vs_predicted(y_true, y_pred, model_name, save_fig=True, show_fig=False)
        self.plot_residuals(y_true, y_pred, model_name, save_fig=True, show_fig=False)
        self.plot_error_distribution(y_true, y_pred, model_name, error_type='percentage', save_fig=True, show_fig=False)
        
        # 保存報告為JSON
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        report_filename = f"evaluation_report_{model_name.replace(' ', '_').lower()}_{timestamp}.json"
        report_filepath = os.path.join(report_dir, report_filename)
        
        with open(report_filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        
        logger.info(f"評估報告已保存至 {report_filepath}")
        
        return report_filepath
    
    def save_evaluator(self, filepath: str) -> None:
        """
        保存評估器狀態
        
        參數:
            filepath (str): 保存檔案路徑
        """
        # 確保目錄存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # 保存評估器
        with open(filepath, 'wb') as f:
            pickle.dump({
                'output_dir': self.output_dir,
                'metrics_history': self.metrics_history,
                'comparison_results': self.comparison_results
            }, f)
        
        logger.info(f"評估器狀態已保存至 {filepath}")
    
    @classmethod
    def load_evaluator(cls, filepath: str) -> 'ModelEvaluator':
        """
        載入評估器狀態
        
        參數:
            filepath (str): 保存檔案路徑
            
        返回:
            ModelEvaluator: 載入的評估器實例
        """
        # 檢查檔案是否存在
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"檔案 {filepath} 不存在")
        
        # 載入評估器
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # 創建新的評估器實例
        evaluator = cls(output_dir=data['output_dir'])
        evaluator.metrics_history = data['metrics_history']
        evaluator.comparison_results = data['comparison_results']
        
        logger.info(f"已從 {filepath} 載入評估器狀態")
        
        return evaluator


# 用法示例
if __name__ == "__main__":
    try:
        import os
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        # 嘗試載入示例數據
        data_path = os.path.join('data', 'processed', 'Training_data_warpage_final.csv')
        
        if os.path.exists(data_path):
            # 載入數據
            data = pd.read_csv(data_path)
            print(f"載入數據: {data_path}, 形狀: {data.shape}")
            
            # 分離特徵和目標變數
            if 'Nf_pred (cycles)' in data.columns:
                target_col = 'Nf_pred (cycles)'
            elif 'Nf_pred' in data.columns:
                target_col = 'Nf_pred'
            else:
                raise ValueError("找不到目標變數列 'Nf_pred (cycles)' 或 'Nf_pred'")
            
            y = data[target_col]
            X = data.drop(target_col, axis=1)
            feature_names = X.columns.tolist()
            
            # 分割訓練集和測試集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"訓練集: {X_train.shape}, 測試集: {X_test.shape}")
            
            # 初始化評估器
            evaluator = ModelEvaluator(output_dir='figures')
            
            # 訓練簡單模型
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 進行預測
            y_pred = model.predict(X_test)
            
            # 計算評估指標
            metrics = evaluator.calculate_metrics(y_test, y_pred, model_name='RandomForest')
            print("\n評估指標:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.6f}")
            
            # 繪製實際值與預測值對比圖
            evaluator.plot_actual_vs_predicted(y_test, y_pred, model_name='RandomForest')
            
            # 繪製殘差分析圖
            evaluator.plot_residuals(y_test, y_pred, model_name='RandomForest')
            
            # 繪製誤差分布圖
            evaluator.plot_error_distribution(y_test, y_pred, model_name='RandomForest')
            
            # 繪製特徵重要性圖
            evaluator.plot_feature_importance(model, feature_names, model_name='RandomForest', top_n=10)
            
            # 計算置換重要性
            try:
                evaluator.plot_permutation_importance(model, X_test.values, y_test.values, 
                                           feature_names, model_name='RandomForest', 
                                           top_n=10, n_repeats=5)
            except Exception as e:
                print(f"繪製置換重要性時出錯: {str(e)}")
            
            # 對重要特徵進行敏感性分析
            try:
                # 找出前3個重要特徵
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[-3:]
                
                # 進行敏感性分析
                for idx in top_indices:
                    evaluator.perform_sensitivity_analysis(model, X_test.values, feature_names, 
                                                idx, model_name='RandomForest')
                
                # 分析前兩個重要特徵的交互影響
                evaluator.plot_two_feature_interaction(model, X_test.values, feature_names,
                                         top_indices[-1], top_indices[-2], 
                                         model_name='RandomForest')
            except Exception as e:
                print(f"進行敏感性分析時出錯: {str(e)}")
            
            # 生成評估報告
            report_path = evaluator.generate_evaluation_report(y_test, y_pred, 'RandomForest')
            print(f"\n評估報告已生成: {report_path}")
            
            # 假設有多個模型進行比較
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'RandomForest_Small': RandomForestRegressor(n_estimators=50, random_state=42),
                'RandomForest_Large': RandomForestRegressor(n_estimators=200, random_state=42)
            }
            
            # 訓練模型並收集預測結果
            predictions = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                predictions[name] = model.predict(X_test)
            
            # 比較模型性能
            comparison_df, _ = evaluator.compare_models(y_test, predictions)
            print("\n模型比較結果:")
            print(comparison_df)
            
            print("\n評估模組示範完成。您可以使用這些函數來評估您的模型性能。")
            
        else:
            print(f"示例數據文件 {data_path} 不存在。")
            print("您可以按照以下方式使用本模組:")
            print("""
# 初始化評估器
evaluator = ModelEvaluator(output_dir='figures')

# 載入測試數據
X_test = ...  # 測試特徵
y_test = ...  # 真實值
y_pred = ...  # 預測值
feature_names = ...  # 特徵名稱列表

# 計算評估指標
metrics = evaluator.calculate_metrics(y_test, y_pred, model_name='MyModel')

# 繪製實際值與預測值對比圖
evaluator.plot_actual_vs_predicted(y_test, y_pred, model_name='MyModel')

# 繪製殘差分析圖
evaluator.plot_residuals(y_test, y_pred, model_name='MyModel')

# 繪製特徵重要性圖 (如果模型支援)
evaluator.plot_feature_importance(model, feature_names, model_name='MyModel')

# 進行敏感性分析
evaluator.perform_sensitivity_analysis(model, X_test, feature_names, feature_idx=0, model_name='MyModel')

# 比較多個模型
predictions = {
    'Model1': y_pred1,
    'Model2': y_pred2,
    ...
}
comparison_df, _ = evaluator.compare_models(y_test, predictions)
            """)
    
    except Exception as e:
        print(f"運行示例時出錯: {str(e)}")
        import traceback
        traceback.print_exc()
        print("請確保所需的依賴包已正確安裝，並提供有效的數據文件。")
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     model_name: str, 
                     save_fig: bool = True, 
                     show_fig: bool = True) -> plt.Figure:
        """
        繪製殘差分析圖
        
        參數:
            y_true (numpy.ndarray): 真實值
            y_pred (numpy.ndarray): 預測值
            model_name (str): 模型名稱
            save_fig (bool): 是否保存圖像
            show_fig (bool): 是否顯示圖像
            
        返回:
            matplotlib.figure.Figure: 圖像對象
        """
        # 確保輸入是一維數組
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)
        
        # 計算殘差
        residuals = y_true - y_pred
        
        # 創建圖像 (2x2的子圖)
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 殘差散點圖
        axs[0, 0].scatter(y_pred, residuals, alpha=0.7, s=50, c='#1f77b4', edgecolors='k', linewidths=0.5)
        axs[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axs[0, 0].set_xlabel('預測值', fontsize=12)
        axs[0, 0].set_ylabel('殘差', fontsize=12)
        axs[0, 0].set_title('殘差 vs 預測值', fontsize=14)
        axs[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 2. 殘差直方圖
        bins = np.min([int(len(residuals) / 5), 30])  # 根據數據量動態調整柱狀數
        bins = np.max([bins, 10])  # 至少10個柱狀
        axs[0, 1].hist(residuals, bins=bins, alpha=0.7, color='#1f77b4', edgecolor='k')
        axs[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axs[0, 1].set_xlabel('殘差', fontsize=12)
        axs[0, 1].set_ylabel('頻率', fontsize=12)
        axs[0, 1].set_title('殘差分布', fontsize=14)
        axs[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # 3. Q-Q圖 (檢查正態性)
        from scipy import stats
        stats.probplot(residuals, plot=axs[1, 0])
        axs[1, 0].set_title('殘差Q-Q圖', fontsize=14)
        axs[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 4. 殘差的絕對值 vs 預測值 (檢查異方差性)
        axs[1, 1].scatter(y_pred, np.abs(residuals), alpha=0.7, s=50, c='#1f77b4', edgecolors='k', linewidths=0.5)
        axs[1, 1].set_xlabel('預測值', fontsize=12)
        axs[1, 1].set_ylabel('殘差絕對值', fontsize=12)
        axs[1, 1].set_title('殘差絕對值 vs 預測值', fontsize=14)
        axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        # 添加總標題
        fig.suptitle(f'{model_name} 模型殘差分析', fontsize=16)
        
        # 調整布局
        fig.tight_layout(rect=[0, 0, 1, 0.97])  # 為總標題留出空間
        
        # 保存圖像
        if save_fig:
            filename = f"residuals_analysis_{model_name.replace(' ', '_').lower()}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至 {filepath}")
        
        # 顯示圖像
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              model_name: str, error_type: str = 'percentage',
                              save_fig: bool = True, 
                              show_fig: bool = True) -> plt.Figure:
        """
        繪製誤差分布圖
        
        參數:
            y_true (numpy.ndarray): 真實值
            y_pred (numpy.ndarray): 預測值
            model_name (str): 模型名稱
            error_type (str): 誤差類型，'absolute' 或 'percentage'
            save_fig (bool): 是否保存圖像
            show_fig (bool): 是否顯示圖像
            
        返回:
            matplotlib.figure.Figure: 圖像對象
        """
        # 確保輸入是一維數組
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)
        
        # 計算誤差
        if error_type == 'absolute':
            errors = y_true - y_pred
            error_label = '絕對誤差'
        else:  # percentage
            errors = 100 * (y_true - y_pred) / np.maximum(1e-10, np.abs(y_true))
            error_label = '百分比誤差 (%)'
        
        # 創建圖像
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 繪製誤差分布直方圖
        bins = np.min([int(len(errors) / 5), 30])  # 根據數據量動態調整柱狀數
        bins = np.max([bins, 10])  # 至少10個柱狀
        
        n, bins, patches = ax.hist(errors, bins=bins, alpha=0.7, color='#1f77b4', edgecolor='k')
        
        # 添加平均誤差線
        mean_error = np.mean(errors)
        ax.axvline(x=mean_error, color='r', linestyle='--', linewidth=2,
                 label=f'平均{error_label} = {mean_error:.2f}')
        
        # 添加標準差線
        std_error = np.std(errors)
        ax.axvline(x=mean_error + std_error, color='g', linestyle=':', linewidth=2,
                 label=f'+1 標準差 = {(mean_error + std_error):.2f}')
        ax.axvline(x=mean_error - std_error, color='g', linestyle=':', linewidth=2,
                 label=f'-1 標準差 = {(mean_error - std_error):.2f}')
        
        # 添加中位數線
        median_error = np.median(errors)
        ax.axvline(x=median_error, color='purple', linestyle='-.', linewidth=2,
                 label=f'中位數 = {median_error:.2f}')
        
        # 添加零誤差線
        ax.axvline(x=0, color='k', linestyle='-', linewidth=1.5, alpha=0.5)
        
        # 添加標籤和標題
        ax.set_xlabel(error_label, fontsize=14)
        ax.set_ylabel('頻率', fontsize=14)
        ax.set_title(f'{model_name} 模型{error_label}分布', fontsize=16)
        
        # 添加網格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加圖例
        ax.legend(loc='upper right')
        
        # 計算誤差統計量
        stats_text = (f"平均{error_label}: {mean_error:.2f}\n"
                     f"中位數{error_label}: {median_error:.2f}\n"
                     f"標準差: {std_error:.2f}\n"
                     f"最大{error_label}: {np.max(errors):.2f}\n"
                     f"最小{error_label}: {np.min(errors):.2f}")
        
        # 在圖像左上角添加文本框
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='left', bbox=props)
        
        # 調整布局
        fig.tight_layout()
        
        # 保存圖像
        if save_fig:
            filename = f"error_distribution_{error_type}_{model_name.replace(' ', '_').lower()}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至 {filepath}")
        
        # 顯示圖像
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_learning_curve(self, estimator: Any, X: np.ndarray, y: np.ndarray, 
                          model_name: str, cv: int = 5, n_jobs: int = -1,
                          train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
                          save_fig: bool = True, 
                          show_fig: bool = True) -> plt.Figure:
        """
        繪製學習曲線，用於評估模型的偏差-方差權衡
        
        參數:
            estimator (Any): 未訓練的模型估計器
            X (numpy.ndarray): 特徵數據
            y (numpy.ndarray): 目標變數
            model_name (str): 模型名稱
            cv (int): 交叉驗證折數
            n_jobs (int): 並行作業數
            train_sizes (numpy.ndarray): 訓練集大小比例
            save_fig (bool): 是否保存圖像
            show_fig (bool): 是否顯示圖像
            
        返回:
            matplotlib.figure.Figure: 圖像對象
        """
        # 創建圖像
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 計算學習曲線
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs,
            train_sizes=train_sizes,
            scoring='neg_mean_squared_error',
            shuffle=True
        )
        
        # 計算均值和標準差
        train_scores_mean = -np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        # 繪製學習曲線
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                       train_scores_mean + train_scores_std, alpha=0.1, color='blue')
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                       test_scores_mean + test_scores_std, alpha=0.1, color='orange')
        ax.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='訓練集MSE')
        ax.plot(train_sizes, test_scores_mean, 'o-', color='orange', label='驗證集MSE')
        
        # 添加標籤和標題
        ax.set_xlabel('訓練樣本數', fontsize=14)
        ax.set_ylabel('均方誤差 (MSE)', fontsize=14)
        ax.set_title(f'{model_name} 模型學習曲線', fontsize=16)
        
        # 添加網格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加圖例
        ax.legend(loc='best', fontsize=12)
        
        # 調整布局
        fig.tight_layout()
        
        # 保存圖像
        if save_fig:
            filename = f"learning_curve_{model_name.replace(' ', '_').lower()}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至 {filepath}")
        
        # 顯示圖像
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def compare_models(self, y_true: np.ndarray, predictions_dict: Dict[str, np.ndarray], 
                     save_fig: bool = True, 
                     show_fig: bool = True) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        比較多個模型的性能
        
        參數:
            y_true (numpy.ndarray): 真實值
            predictions_dict (Dict[str, numpy.ndarray]): 模型名稱與其預測值的字典
            save_fig (bool): 是否保存圖像
            show_fig (bool): 是否顯示圖像
            
        返回:
            Tuple[pandas.DataFrame, matplotlib.figure.Figure]: 比較結果數據框和圖像對象
        """
        # 確保輸入是一維數組
        y_true = np.ravel(y_true)
        
        # 計算每個模型的評估指標
        comparison_data = []
        metrics = ['rmse', 'mae', 'mape', 'r2', 'explained_variance', 'max_error']
        
        for model_name, y_pred in predictions_dict.items():
            y_pred = np.ravel(y_pred)
            model_metrics = self.calculate_metrics(y_true, y_pred, model_name=None, log_metrics=False)
            comparison_data.append([model_name] + [model_metrics[m] for m in metrics])
        
        # 創建比較結果數據框
        columns = ['模型'] + [
            'RMSE (均方根誤差)',
            'MAE (平均絕對誤差)',
            'MAPE (平均絕對百分比誤差 %)',
            'R² (決定係數)',
            '解釋方差分數',
            '最大誤差'
        ]
        comparison_df = pd.DataFrame(comparison_data, columns=columns)
        
        # 排序，根據RMSE升序
        comparison_df = comparison_df.sort_values('RMSE (均方根誤差)')
        
        # 保存比較結果
        self.comparison_results = comparison_df.set_index('模型').to_dict()
        
        # 繪製比較圖
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. RMSE 比較
        axs[0, 0].bar(comparison_df['模型'], comparison_df['RMSE (均方根誤差)'], color='skyblue')
        axs[0, 0].set_xlabel('模型', fontsize=12)
        axs[0, 0].set_ylabel('RMSE', fontsize=12)
        axs[0, 0].set_title('模型RMSE比較', fontsize=14)
        axs[0, 0].tick_params(axis='x', rotation=45)
        axs[0, 0].grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 2. MAE 比較
        axs[0, 1].bar(comparison_df['模型'], comparison_df['MAE (平均絕對誤差)'], color='lightgreen')
        axs[0, 1].set_xlabel('模型', fontsize=12)
        axs[0, 1].set_ylabel('MAE', fontsize=12)
        axs[0, 1].set_title('模型MAE比較', fontsize=14)
        axs[0, 1].tick_params(axis='x', rotation=45)
        axs[0, 1].grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 3. MAPE 比較
        axs[1, 0].bar(comparison_df['模型'], comparison_df['MAPE (平均絕對百分比誤差 %)'], color='salmon')
        axs[1, 0].set_xlabel('模型', fontsize=12)
        axs[1, 0].set_ylabel('MAPE (%)', fontsize=12)
        axs[1, 0].set_title('模型MAPE比較', fontsize=14)
        axs[1, 0].tick_params(axis='x', rotation=45)
        axs[1, 0].grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 4. R² 比較
        axs[1, 1].bar(comparison_df['模型'], comparison_df['R² (決定係數)'], color='mediumpurple')
        axs[1, 1].set_xlabel('模型', fontsize=12)
        axs[1, 1].set_ylabel('R²', fontsize=12)
        axs[1, 1].set_title('模型R²比較', fontsize=14)
        axs[1, 1].tick_params(axis='x', rotation=45)
        axs[1, 1].grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 添加總標題
        fig.suptitle('模型性能比較', fontsize=16)
        
        # 調整布局
        fig.tight_layout(rect=[0, 0, 1, 0.97])  # 為總標題留出空間
        
        # 保存圖像
        if save_fig:
            filename = "model_comparison.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至 {filepath}")
            
            # 保存比較結果到CSV
            csv_filepath = os.path.join(self.output_dir, "model_comparison.csv")
            comparison_df.to_csv(csv_filepath, index=False, float_format='%.6f')
            logger.info(f"比較結果已保存至 {csv_filepath}")
        
        # 顯示圖像
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return comparison_df, fig
    
    def plot_feature_importance(self, model: Any, feature_names: List[str], 
                              model_name: str = None,
                              top_n: int = None,
                              save_fig: bool = True, 
                              show_fig: bool = True) -> plt.Figure:
        """
        繪製特徵重要性圖
        
        參數:
            model (Any): 訓練好的模型，必須有feature_importances_屬性
            feature_names (List[str]): 特徵名稱列表
            model_name (str, optional): 模型名稱
            top_n (int, optional): 只顯示前N個重要特徵
            save_fig (bool): 是否保存圖像
            show_fig (bool): 是否顯示圖像
            
        返回:
            matplotlib.figure.Figure: 圖像對象
        """
        # 檢查模型是否具有feature_importances_屬性
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("模型必須具有feature_importances_屬性")
        
        # 獲取特徵重要性
        importances = model.feature_importances_
        
        # 將特徵重要性與特徵名稱配對
        features = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # 按重要性降序排序
        features = features.sort_values('importance', ascending=False)
        
        # 只取前N個特徵
        if top_n is not None and top_n < len(features):
            features = features.head(top_n)
        
        # 創建圖像
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
        
        # 繪製條形圖
        bars = ax.barh(features['feature'], features['importance'], color='skyblue', edgecolor='k', alpha=0.8)
        
        # 為每個條形添加數值標籤
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width * 1.01  # 稍微偏離條形的末端
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                  va='center', ha='left', fontsize=10)
        
        # 添加標籤和標題
        ax.set_xlabel('重要性', fontsize=14)
        ax.set_ylabel('特徵', fontsize=14)
        title = f'特徵重要性' if model_name is None else f'{model_name} 模型特徵重要性'
        ax.set_title(title, fontsize=16)
        
        # 添加網格
        ax.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        # 調整布局
        fig.tight_layout()
        
        # 保存圖像
        if save_fig:
            model_str = "" if model_name is None else f"_{model_name.replace(' ', '_').lower()}"
            filename = f"feature_importance{model_str}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至 {filepath}")
        
        # 顯示圖像
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig