#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
data_processing.py - 數據處理模組
本模組提供數據加載、清理、特徵工程和數據分割等功能，
用於準備銲錫接點疲勞壽命預測模型的訓練和測試數據。

主要功能:
1. 加載原始CAE模擬數據和實驗數據
2. 數據清理與預處理
3. 特徵工程與選擇
4. 數據正規化與標準化
5. 訓練集和測試集分割
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
import logging

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    數據處理類，提供數據處理的各種功能
    """
    
    def __init__(self, random_state=42):
        """
        初始化數據處理類
        
        參數:
            random_state (int): 隨機種子，用於可重複的隨機操作
        """
        self.random_state = random_state
        self.scaler_X = None  # 特徵標準化器
        self.scaler_y = None  # 目標變數標準化器
        self.feature_selector = None  # 特徵選擇器
        self.pca = None  # PCA降維
        self.selected_features = None  # 選擇的特徵列表
        
    def load_data(self, filepath, sheet_name=0):
        """
        加載數據文件
        
        參數:
            filepath (str): 數據文件路徑
            sheet_name (str/int, optional): Excel表名稱或索引，默認為0
        
        返回:
            pandas.DataFrame: 加載的數據框
        """
        try:
            file_ext = os.path.splitext(filepath)[1].lower()
            if file_ext == '.csv':
                data = pd.read_csv(filepath)
            elif file_ext in ['.xls', '.xlsx']:
                data = pd.read_excel(filepath, sheet_name=sheet_name)
            else:
                raise ValueError(f"不支援的文件格式: {file_ext}")
            
            logger.info(f"成功從 {filepath} 載入數據，形狀: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"載入數據時出錯: {str(e)}")
            raise
            
    def clean_data(self, data):
        """
        數據清理
        
        參數:
            data (pandas.DataFrame): 原始數據框
            
        返回:
            pandas.DataFrame: 清理後的數據框
        """
        # 複製數據避免修改原始數據
        df = data.copy()
        
        # 記錄初始形狀
        initial_shape = df.shape
        
        # 處理缺失值
        df = df.dropna()
        
        # 移除重複項
        df = df.drop_duplicates()
        
        # 檢測並處理異常值 (使用IQR方法)
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # 定義異常值邊界
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 過濾異常值
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # 記錄數據清理的結果
        final_shape = df.shape
        
        logger.info(f"數據清理完成: 原始形狀 {initial_shape} -> 清理後形狀 {final_shape}")
        logger.info(f"移除了 {initial_shape[0] - final_shape[0]} 行")
        
        return df
    
    def engineer_features(self, data):
        """
        特徵工程
        
        參數:
            data (pandas.DataFrame): 原始數據框
            
        返回:
            pandas.DataFrame: 添加新特徵後的數據框
        """
        # 複製數據避免修改原始數據
        df = data.copy()
        
        # 記錄初始特徵數
        initial_feature_count = df.shape[1]
        
        # 根據物理知識添加特徵
        # 1. 如果存在相關列，計算應力與應變的比率
        stress_cols = [col for col in df.columns if 'NLPLWK' in col]
        strain_cols = [col for col in df.columns if 'Strain' in col]
        
        if stress_cols and strain_cols:
            # 添加最大應力應變比
            max_stress_col = max(stress_cols, key=lambda x: df[x].max())
            max_strain_col = max(strain_cols, key=lambda x: df[x].max())
            df['stress_strain_ratio'] = df[max_stress_col] / df[max_strain_col].replace(0, np.nan)
            
            # 添加平均應力
            df['avg_stress'] = df[stress_cols].mean(axis=1)
            
            # 添加應力變化率（如果有時間序列數據）
            time_series_cols = [col for col in stress_cols if any(str(t) in col for t in [3600, 7200, 10800, 14400])]
            if len(time_series_cols) >= 2:
                sorted_cols = sorted(time_series_cols, 
                                    key=lambda x: int(''.join(filter(str.isdigit, x))))
                for i in range(1, len(sorted_cols)):
                    df[f'stress_change_{i}'] = df[sorted_cols[i]] - df[sorted_cols[i-1]]
        
        # 2. 添加翹曲相關特徵
        warpage_cols = [col for col in df.columns if 'warpage' in col.lower()]
        if warpage_cols:
            df['warpage_ratio'] = df['Total_warpage'] / df['Unit_warpage'] if 'Total_warpage' in df.columns and 'Unit_warpage' in df.columns else np.nan
            
        # 3. 添加物理意義的特徵組合
        # 如果有材料和幾何參數，可以添加複合特徵
        geometry_cols = ['Die', 'stud', 'mold', 'PCB']
        if all(col in df.columns for col in geometry_cols):
            # 幾何形狀比例
            df['die_pcb_ratio'] = df['Die'] / df['PCB']
            df['stud_mold_ratio'] = df['stud'] / df['mold']
            
            # 物理複合特徵 (基於材料力學)
            if 'avg_stress' in df.columns:
                df['geometric_stress_factor'] = df['avg_stress'] * (df['Die'] / df['PCB'])
        
        # 記錄特徵工程結果
        final_feature_count = df.shape[1]
        logger.info(f"特徵工程完成: 原始特徵數 {initial_feature_count} -> 新特徵數 {final_feature_count}")
        logger.info(f"新增了 {final_feature_count - initial_feature_count} 個特徵")
        
        # 移除具有高度相關性的特徵
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # 找出相關性大於0.95的特徵
        high_corr_features = [column for column in upper.columns if any(upper[column] > 0.95)]
        
        if high_corr_features:
            df = df.drop(high_corr_features, axis=1)
            logger.info(f"移除了 {len(high_corr_features)} 個高相關性特徵: {high_corr_features}")
        
        return df
    
    def select_features(self, X, y, method='k_best', n_features=None):
        """
        特徵選擇
        
        參數:
            X (pandas.DataFrame): 特徵數據框
            y (pandas.Series): 目標變數
            method (str): 特徵選擇方法，'k_best' 或 'pca'
            n_features (int, optional): 選擇的特徵數量，默認為None，表示使用所有特徵的一半
            
        返回:
            pandas.DataFrame: 選擇特徵後的數據框
        """
        # 如果未指定特徵數量，預設使用一半的特徵
        if n_features is None:
            n_features = X.shape[1] // 2
            n_features = max(n_features, 1)  # 至少選擇一個特徵
        
        if method == 'k_best':
            logger.info(f"使用 SelectKBest 方法選擇 {n_features} 個最佳特徵")
            self.feature_selector = SelectKBest(f_regression, k=n_features)
            X_new = self.feature_selector.fit_transform(X, y)
            
            # 獲取選擇的特徵名稱
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_features = X.columns[selected_indices].tolist()
            
            # 創建具有正確特徵名稱的DataFrame
            X_selected = pd.DataFrame(X_new, columns=self.selected_features, index=X.index)
            
        elif method == 'pca':
            logger.info(f"使用 PCA 方法將特徵降維至 {n_features} 個主成分")
            self.pca = PCA(n_components=n_features, random_state=self.random_state)
            X_new = self.pca.fit_transform(X)
            
            # 創建具有新特徵名稱的DataFrame
            X_selected = pd.DataFrame(X_new, columns=[f'PC{i+1}' for i in range(n_features)], index=X.index)
            self.selected_features = X_selected.columns.tolist()
            
        else:
            raise ValueError(f"不支援的特徵選擇方法: {method}")
        
        logger.info(f"特徵選擇完成，選擇的特徵: {self.selected_features}")
        return X_selected
    
    def normalize_data(self, X, y=None, method='standard', fit_scalers=True):
        """
        數據標準化/正規化
        
        參數:
            X (pandas.DataFrame): 特徵數據框
            y (pandas.Series, optional): 目標變數，默認為None
            method (str): 標準化方法，'standard' 或 'minmax'
            fit_scalers (bool): 是否擬合標準化器，用於訓練集
            
        返回:
            tuple: (標準化後的X, 標準化後的y)，如果y為None則返回(標準化後的X, None)
        """
        if method == 'standard':
            if fit_scalers:
                self.scaler_X = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler_X.fit_transform(X) if fit_scalers else self.scaler_X.transform(X),
                columns=X.columns,
                index=X.index
            )
        elif method == 'minmax':
            if fit_scalers:
                self.scaler_X = MinMaxScaler()
            X_scaled = pd.DataFrame(
                self.scaler_X.fit_transform(X) if fit_scalers else self.scaler_X.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            raise ValueError(f"不支援的標準化方法: {method}")
        
        # 如果提供了目標變數，也對其進行標準化
        y_scaled = None
        if y is not None:
            if fit_scalers:
                self.scaler_y = StandardScaler()
            y_scaled = pd.Series(
                self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten() if fit_scalers else self.scaler_y.transform(y.values.reshape(-1, 1)).flatten(),
                index=y.index
            )
        
        logger.info(f"使用 {method} 方法完成數據標準化")
        return X_scaled, y_scaled
    
    def inverse_transform_y(self, y_scaled):
        """
        將標準化後的目標變數轉換回原始尺度
        
        參數:
            y_scaled (numpy.ndarray or pandas.Series): 標準化後的目標變數
            
        返回:
            numpy.ndarray: 原始尺度的目標變數
        """
        if self.scaler_y is None:
            raise ValueError("標準化器尚未擬合，請先調用normalize_data方法")
        
        # 將輸入轉換為2D數組
        if isinstance(y_scaled, pd.Series):
            y_scaled = y_scaled.values
        
        y_scaled_2d = y_scaled.reshape(-1, 1)
        y_orig = self.scaler_y.inverse_transform(y_scaled_2d).flatten()
        
        return y_orig
    
    def split_data(self, X, y, test_size=0.2, stratify=None):
        """
        分割訓練集和測試集
        
        參數:
            X (pandas.DataFrame): 特徵數據框
            y (pandas.Series): 目標變數
            test_size (float): 測試集比例，默認為0.2
            stratify (array-like): 分層抽樣的依據，默認為None
            
        返回:
            tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=self.random_state,
                stratify=stratify
            )
            
            logger.info(f"數據分割完成: 訓練集 {X_train.shape[0]} 樣本, 測試集 {X_test.shape[0]} 樣本")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"數據分割時出錯: {str(e)}")
            raise
    
    def prepare_data_pipeline(self, data, target_col='Nf_pred', test_size=0.2, 
                              clean=True, feature_engineering=True, 
                              feature_selection=False, normalization='standard'):
        """
        完整的數據處理流程
        
        參數:
            data (pandas.DataFrame): 原始數據框
            target_col (str): 目標變數列名，默認為'Nf_pred'
            test_size (float): 測試集比例，默認為0.2
            clean (bool): 是否進行數據清理，默認為True
            feature_engineering (bool): 是否進行特徵工程，默認為True
            feature_selection (bool or str): 是否進行特徵選擇，可為False、'k_best'或'pca'，默認為False
            normalization (str or False): 標準化方法，可為'standard'、'minmax'或False，默認為'standard'
            
        返回:
            tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        try:
            # 複製數據避免修改原始數據
            df = data.copy()
            
            # 1. 數據清理
            if clean:
                logger.info("開始數據清理...")
                df = self.clean_data(df)
            
            # 2. 分離特徵和目標變數
            if target_col not in df.columns:
                raise ValueError(f"目標變數列 '{target_col}' 不在數據中")
            
            y = df[target_col]
            X = df.drop(target_col, axis=1)
            
            # 3. 特徵工程
            if feature_engineering:
                logger.info("開始特徵工程...")
                X = self.engineer_features(X)
            
            # 4. 數據分割 (在標準化和特徵選擇之前，以避免數據洩漏)
            logger.info(f"分割數據，測試集比例: {test_size}...")
            X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=test_size)
            
            # 5. 特徵選擇 (僅在訓練集上進行)
            if feature_selection:
                logger.info(f"使用 {feature_selection} 方法進行特徵選擇...")
                X_train = self.select_features(X_train, y_train, method=feature_selection)
                # 應用相同的特徵選擇到測試集
                if feature_selection == 'k_best':
                    # 對於SelectKBest，選擇相同的特徵
                    X_test = X_test[self.selected_features]
                elif feature_selection == 'pca':
                    # 對於PCA，應用相同的轉換
                    X_test = pd.DataFrame(
                        self.pca.transform(X_test),
                        columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
                        index=X_test.index
                    )
            
            # 6. 數據標準化 (僅擬合訓練集，然後轉換兩者)
            if normalization:
                logger.info(f"使用 {normalization} 方法進行數據標準化...")
                X_train, y_train = self.normalize_data(X_train, y_train, method=normalization, fit_scalers=True)
                X_test, y_test = self.normalize_data(X_test, y_test, method=normalization, fit_scalers=False)
            
            # 獲取最終特徵名稱
            feature_names = X_train.columns.tolist()
            
            logger.info("數據處理流程完成")
            logger.info(f"訓練集形狀: X_train {X_train.shape}, y_train {y_train.shape}")
            logger.info(f"測試集形狀: X_test {X_test.shape}, y_test {y_test.shape}")
            
            return X_train, X_test, y_train, y_test, feature_names
            
        except Exception as e:
            logger.error(f"數據處理流程出錯: {str(e)}")
            raise

# 用法示例 (如果直接運行此腳本)
if __name__ == "__main__":
    # 數據處理示例
    processor = DataProcessor(random_state=42)
    
    # 加載數據
    try:
        # 嘗試載入示例數據
        data_path = os.path.join('data', 'processed', 'Training_data_warpage_final.csv')
        data = processor.load_data(data_path)
        
        print("\n數據概述:")
        print(data.head())
        print("\n數據形狀:", data.shape)
        print("\n數據類型:\n", data.dtypes)
        
        # 展示完整數據處理流程
        X_train, X_test, y_train, y_test, feature_names = processor.prepare_data_pipeline(
            data, 
            target_col='Nf_pred (cycles)',  # 請根據實際的目標列名調整
            test_size=0.2,
            clean=True,
            feature_engineering=True,
            feature_selection='k_best',
            normalization='standard'
        )
        
        print("\n處理後的特徵列表:", feature_names)
        print("\n處理後的訓練集形狀:", X_train.shape)
        print("\n處理後的測試集形狀:", X_test.shape)
        
    except Exception as e:
        print(f"運行示例時出錯: {str(e)}")
        print("這可能是因為示例數據路徑不存在。請確保數據文件位於正確的位置。")