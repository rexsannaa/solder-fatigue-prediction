#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
feature_engineering.py - 特徵工程模組
本模組提供針對銲錫接點疲勞壽命預測的特徵工程功能，根據物理模型和經驗知識，
從原始數據中提取和生成更具預測價值的特徵。

主要功能:
1. 基於材料力學的特徵生成
2. 多變量特徵交互生成
3. 時間序列特徵提取
4. 特徵重要性評估與選擇
5. 特徵縮放與轉換
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import r2_score
import logging

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    特徵工程類，提供銲錫接點疲勞壽命預測的特徵生成和處理方法
    """
    
    def __init__(self, random_state=42):
        """
        初始化特徵工程類
        
        參數:
            random_state (int): 隨機種子，用於可重複的隨機操作
        """
        self.random_state = random_state
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.poly = None
        self.selected_features = None
        
    def create_material_features(self, data):
        """
        基於材料力學知識生成特徵
        
        參數:
            data (pandas.DataFrame): 輸入數據框
        
        返回:
            pandas.DataFrame: 添加材料力學特徵後的數據框
        """
        df = data.copy()
        
        # 檢測相關列的存在
        structure_cols = ['Die', 'stud', 'mold', 'PCB']
        stress_cols = [col for col in df.columns if 'NLPLWK' in col]
        strain_cols = [col for col in df.columns if 'Strain' in col]
        warpage_cols = [col for col in df.columns if 'warpage' in col.lower()]
        
        logger.info(f"檢測到 {len(stress_cols)} 個應力相關列、{len(strain_cols)} 個應變相關列")
        
        # 1. 基於彈塑性力學的特徵
        if stress_cols and strain_cols:
            logger.info("生成彈塑性力學特徵...")
            
            # 計算最大應力應變比
            for stress_col in stress_cols:
                for strain_col in strain_cols:
                    if ('up' in stress_col and 'up' in strain_col) or ('down' in stress_col and 'down' in strain_col):
                        col_name = f"ratio_{stress_col}_{strain_col}"
                        df[col_name] = df[stress_col] / df[strain_col].replace(0, np.nan)
            
            # 計算塑性功密度變化率
            up_stress_cols = sorted([col for col in stress_cols if 'up' in col], 
                                  key=lambda x: int(''.join(filter(str.isdigit, x))))
            down_stress_cols = sorted([col for col in stress_cols if 'down' in col], 
                                    key=lambda x: int(''.join(filter(str.isdigit, x))))
            
            if len(up_stress_cols) > 1:
                for i in range(1, len(up_stress_cols)):
                    df[f'up_stress_change_{i}'] = df[up_stress_cols[i]] - df[up_stress_cols[i-1]]
                    if i > 1:  # 加速度特徵
                        df[f'up_stress_accel_{i-1}'] = df[f'up_stress_change_{i}'] - df[f'up_stress_change_{i-1}']
            
            if len(down_stress_cols) > 1:
                for i in range(1, len(down_stress_cols)):
                    df[f'down_stress_change_{i}'] = df[down_stress_cols[i]] - df[down_stress_cols[i-1]]
                    if i > 1:  # 加速度特徵
                        df[f'down_stress_accel_{i-1}'] = df[f'down_stress_change_{i}'] - df[f'down_stress_change_{i-1}']
        
        # 2. 基於結構力學的特徵
        if all(col in df.columns for col in structure_cols):
            logger.info("生成結構力學特徵...")
            
            # 計算幾何比例
            df['die_pcb_ratio'] = df['Die'] / df['PCB']
            df['stud_mold_ratio'] = df['stud'] / df['mold']
            df['die_mold_ratio'] = df['Die'] / df['mold']
            
            # 計算剛度相關特徵
            df['bending_resistance'] = df['Die'] * (df['mold']**3) / 12  # 基於梁彎曲理論
            df['stiffness_factor'] = (df['Die'] * df['mold']) / df['PCB']  # 剛度因子
        
        # 3. 基於熱力學的特徵
        if warpage_cols:
            logger.info("生成熱力學和翹曲特徵...")
            
            if 'Total_warpage' in df.columns and 'Unit_warpage' in df.columns:
                df['warpage_ratio'] = df['Total_warpage'] / df['Unit_warpage']
                
                # 考慮翹曲的規一化
                if all(col in df.columns for col in structure_cols):
                    df['normalized_warpage'] = df['Total_warpage'] / (df['Die'] * df['PCB'])
                    
            # 如果有應力數據，結合翹曲特徵
            if stress_cols and 'Total_warpage' in df.columns:
                max_stress_col = max(stress_cols, key=lambda x: df[x].max())
                df['stress_warpage_product'] = df[max_stress_col] * df['Total_warpage']
        
        # 4. 基於疲勞理論的特徵
        if stress_cols:
            logger.info("生成疲勞理論特徵...")
            
            # 計算應力幅度 (上升和下降的差異)
            if 'NLPLWK_up_14400' in df.columns and 'NLPLWK_down_14400' in df.columns:
                df['stress_amplitude_14400'] = abs(df['NLPLWK_up_14400'] - df['NLPLWK_down_14400'])
            
            if 'NLPLWK_up_10800' in df.columns and 'NLPLWK_down_10800' in df.columns:
                df['stress_amplitude_10800'] = abs(df['NLPLWK_up_10800'] - df['NLPLWK_down_10800'])
            
            # 計算應力累積特徵
            up_stress_cols = [col for col in stress_cols if 'up' in col]
            down_stress_cols = [col for col in stress_cols if 'down' in col]
            
            if up_stress_cols:
                df['cumulative_up_stress'] = df[up_stress_cols].sum(axis=1)
            
            if down_stress_cols:
                df['cumulative_down_stress'] = df[down_stress_cols].sum(axis=1)
            
            if 'cumulative_up_stress' in df.columns and 'cumulative_down_stress' in df.columns:
                df['total_cycle_stress'] = df['cumulative_up_stress'] + df['cumulative_down_stress']
        
        # 5. 計算熱循環中的累積損傷特徵
        if 'Acc_Equi_Strain_max' in df.columns:
            if 'Die' in df.columns:
                df['damage_factor'] = df['Acc_Equi_Strain_max'] * df['Die']
            
            # 根據Coffin-Manson關係簡化的疲勞特徵
            df['fatigue_index'] = df['Acc_Equi_Strain_max']**1.5  # 假設疲勞指數接近1.5
            
        # 檢測特徵增加的數量
        new_features = set(df.columns) - set(data.columns)
        logger.info(f"共生成 {len(new_features)} 個新特徵: {new_features}")
        
        return df
    
    def create_interaction_features(self, data, degree=2, interaction_only=True):
        """
        生成特徵交互項
        
        參數:
            data (pandas.DataFrame): 輸入數據框
            degree (int): 多項式特徵的最高次數，默認為2
            interaction_only (bool): 是否只包含交互項，而不包含單個特徵的冪，默認為True
            
        返回:
            pandas.DataFrame: 添加交互特徵後的數據框
        """
        logger.info(f"生成特徵交互項，最高次數: {degree}, 僅交互: {interaction_only}")
        
        # 只對數值特徵生成交互項
        numeric_data = data.select_dtypes(include=[np.number])
        
        # 初始化PolynomialFeatures
        self.poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        
        # 轉換數據
        poly_features = self.poly.fit_transform(numeric_data)
        
        # 獲取特徵名稱
        feature_names = self.poly.get_feature_names_out(numeric_data.columns)
        
        # 過濾掉原始特徵
        original_features = set(numeric_data.columns)
        interaction_features = [name for name in feature_names if name not in original_features]
        
        # 只取交互特徵部分
        poly_df = pd.DataFrame(
            poly_features[:, len(original_features):], 
            columns=interaction_features,
            index=data.index
        )
        
        logger.info(f"生成了 {len(interaction_features)} 個交互特徵")
        
        # 合併原始數據和交互特徵
        result = pd.concat([data, poly_df], axis=1)
        
        return result
    
    def create_time_series_features(self, data):
        """
        從時間序列數據中生成特徵
        
        參數:
            data (pandas.DataFrame): 輸入數據框
            
        返回:
            pandas.DataFrame: 添加時間序列特徵後的數據框
        """
        df = data.copy()
        
        # 識別時間序列特徵
        time_points = [3600, 7200, 10800, 14400]  # 假設這些是時間點
        
        # 尋找帶有時間點的特徵
        time_series_prefixes = []
        for col in df.columns:
            for time in time_points:
                if str(time) in col:
                    prefix = col.replace(str(time), "")
                    if prefix not in time_series_prefixes:
                        time_series_prefixes.append(prefix)
        
        logger.info(f"識別到 {len(time_series_prefixes)} 個時間序列前綴: {time_series_prefixes}")
        
        for prefix in time_series_prefixes:
            # 尋找該前綴的所有時間點特徵
            ts_cols = [col for col in df.columns if prefix in col and any(str(t) in col for t in time_points)]
            
            if len(ts_cols) < 2:
                continue
                
            # 按時間點排序
            ts_cols.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            
            # 特徵名稱基礎
            base_name = prefix.replace("_", "").strip()
            
            # 計算差分特徵 (速率)
            for i in range(1, len(ts_cols)):
                df[f'{base_name}_diff_{i}'] = df[ts_cols[i]] - df[ts_cols[i-1]]
            
            # 計算二階差分特徵 (加速度)
            for i in range(2, len(ts_cols)):
                df[f'{base_name}_diff2_{i}'] = df[f'{base_name}_diff_{i}'] - df[f'{base_name}_diff_{i-1}']
            
            # 計算相對變化率
            for i in range(1, len(ts_cols)):
                denominator = df[ts_cols[i-1]].replace(0, np.nan)  # 避免除以零
                df[f'{base_name}_rel_change_{i}'] = (df[ts_cols[i]] - df[ts_cols[i-1]]) / denominator
            
            # 計算統計特徵
            df[f'{base_name}_max'] = df[ts_cols].max(axis=1)
            df[f'{base_name}_min'] = df[ts_cols].min(axis=1)
            df[f'{base_name}_mean'] = df[ts_cols].mean(axis=1)
            df[f'{base_name}_std'] = df[ts_cols].std(axis=1)
            
            # 計算峰值與谷值
            df[f'{base_name}_peak_to_valley'] = df[f'{base_name}_max'] - df[f'{base_name}_min']
        
        # 檢測特徵增加的數量
        new_features = set(df.columns) - set(data.columns)
        logger.info(f"共生成 {len(new_features)} 個時間序列特徵")
        
        return df
    
    def select_features(self, X, y, method='mutual_info', n_features=None, threshold=None):
        """
        特徵選擇
        
        參數:
            X (pandas.DataFrame): 特徵數據框
            y (pandas.Series): 目標變數
            method (str): 特徵選擇方法，'f_regression' 或 'mutual_info'
            n_features (int, optional): 選擇的特徵數量，默認為None
            threshold (float, optional): 特徵重要性閾值，默認為None
            
        返回:
            pandas.DataFrame: 選擇特徵後的數據框和選擇的特徵列表
        """
        # 如果未指定特徵數量或閾值，預設選擇一半的特徵
        if n_features is None and threshold is None:
            n_features = X.shape[1] // 2
            n_features = max(n_features, 1)  # 至少選擇一個特徵
        
        logger.info(f"使用 {method} 方法進行特徵選擇")
        
        # 選擇特徵選擇器
        if method == 'f_regression':
            if threshold is not None:
                self.feature_selector = SelectKBest(f_regression, k='all')
                self.feature_selector.fit(X, y)
                # 基於閾值選擇特徵
                mask = self.feature_selector.pvalues_ < threshold
                selected_features = X.columns[mask].tolist()
            else:
                self.feature_selector = SelectKBest(f_regression, k=n_features)
                self.feature_selector.fit(X, y)
                selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        elif method == 'mutual_info':
            if threshold is not None:
                self.feature_selector = SelectKBest(mutual_info_regression, k='all')
                self.feature_selector.fit(X, y)
                # 基於閾值選擇特徵
                scores = self.feature_selector.scores_
                mask = scores > threshold * max(scores)
                selected_features = X.columns[mask].tolist()
            else:
                self.feature_selector = SelectKBest(mutual_info_regression, k=n_features)
                self.feature_selector.fit(X, y)
                selected_features = X.columns[self.feature_selector.get_support()].tolist()
                
        else:
            raise ValueError(f"不支援的特徵選擇方法: {method}")
        
        self.selected_features = selected_features
        logger.info(f"選擇了 {len(selected_features)} 個特徵: {selected_features}")
        
        # 返回僅包含選定特徵的數據框
        return X[selected_features]
    
    def get_feature_importance(self, X, y, method='mutual_info'):
        """
        獲取特徵重要性
        
        參數:
            X (pandas.DataFrame): 特徵數據框
            y (pandas.Series): 目標變數
            method (str): 特徵重要性計算方法，'f_regression' 或 'mutual_info'
            
        返回:
            pandas.DataFrame: 包含特徵及其重要性的數據框，按重要性降序排列
        """
        logger.info(f"使用 {method} 方法計算特徵重要性")
        
        if method == 'f_regression':
            # 使用F值評估特徵重要性
            selector = SelectKBest(f_regression, k='all')
            selector.fit(X, y)
            # F值越大表示特徵越重要
            importance = selector.scores_
            # p值越小表示特徵越顯著
            pvalues = selector.pvalues_
            
            # 創建特徵重要性數據框
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'f_value': importance,
                'p_value': pvalues
            })
            
            # 按F值降序排序
            importance_df = importance_df.sort_values('f_value', ascending=False)
            
        elif method == 'mutual_info':
            # 使用互信息評估特徵重要性
            selector = SelectKBest(mutual_info_regression, k='all')
            selector.fit(X, y)
            # 互信息值越大表示特徵越重要
            importance = selector.scores_
            
            # 創建特徵重要性數據框
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'mutual_info': importance
            })
            
            # 按互信息值降序排序
            importance_df = importance_df.sort_values('mutual_info', ascending=False)
            
        else:
            raise ValueError(f"不支援的特徵重要性計算方法: {method}")
        
        logger.info(f"已計算 {len(X.columns)} 個特徵的重要性")
        
        return importance_df
    
    def dimensionality_reduction(self, X, n_components=None, variance_ratio=0.95):
        """
        使用PCA進行降維
        
        參數:
            X (pandas.DataFrame): 特徵數據框
            n_components (int, optional): 主成分數量，默認為None
            variance_ratio (float, optional): 解釋方差比例，默認為0.95
            
        返回:
            pandas.DataFrame: 降維後的數據框
        """
        # 如果未指定主成分數量，則基於解釋方差比例確定
        if n_components is None:
            self.pca = PCA(n_components=variance_ratio, random_state=self.random_state)
        else:
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
        
        # 轉換數據
        X_pca = self.pca.fit_transform(X)
        
        # 創建包含主成分的數據框
        pca_df = pd.DataFrame(
            X_pca,
            columns=[f'PC{i+1}' for i in range(X_pca.shape[1])],
            index=X.index
        )
        
        logger.info(f"使用PCA將 {X.shape[1]} 個特徵降維至 {X_pca.shape[1]} 個主成分")
        logger.info(f"解釋方差比例: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        return pca_df
    
    def scale_features(self, X, method='standard', fit=True):
        """
        特徵縮放
        
        參數:
            X (pandas.DataFrame): 特徵數據框
            method (str): 縮放方法，'standard' 或 'minmax'
            fit (bool): 是否擬合縮放器，用於訓練集
            
        返回:
            pandas.DataFrame: 縮放後的數據框
        """
        if method == 'standard':
            if fit or self.scaler is None:
                self.scaler = StandardScaler()
                scaled_X = self.scaler.fit_transform(X)
            else:
                scaled_X = self.scaler.transform(X)
        elif method == 'minmax':
            if fit or self.scaler is None:
                self.scaler = MinMaxScaler()
                scaled_X = self.scaler.fit_transform(X)
            else:
                scaled_X = self.scaler.transform(X)
        else:
            raise ValueError(f"不支援的縮放方法: {method}")
        
        # 創建包含原始特徵名稱的數據框
        scaled_df = pd.DataFrame(
            scaled_X,
            columns=X.columns,
            index=X.index
        )
        
        logger.info(f"使用 {method} 方法完成特徵縮放")
        
        return scaled_df
    
    def apply_feature_engineering_pipeline(self, X, y=None, material_features=True, 
                                         interaction_features=False, time_series_features=True,
                                         feature_selection=True, selection_method='mutual_info',
                                         n_selected_features=None, scaling=True, scaling_method='standard'):
        """
        應用完整的特徵工程流程
        
        參數:
            X (pandas.DataFrame): 特徵數據框
            y (pandas.Series, optional): 目標變數，用於特徵選擇，默認為None
            material_features (bool): 是否生成材料特徵，默認為True
            interaction_features (bool): 是否生成交互特徵，默認為False
            time_series_features (bool): 是否生成時間序列特徵，默認為True
            feature_selection (bool): 是否進行特徵選擇，默認為True
            selection_method (str): 特徵選擇方法，默認為'mutual_info'
            n_selected_features (int, optional): 選擇的特徵數量，默認為None
            scaling (bool): 是否進行特徵縮放，默認為True
            scaling_method (str): 縮放方法，默認為'standard'
            
        返回:
            pandas.DataFrame: 特徵工程處理後的數據框
        """
        # 複製數據避免修改原始數據
        X_processed = X.copy()
        
        logger.info("開始應用特徵工程流程...")
        
        # 1. 生成材料力學特徵
        if material_features:
            logger.info("生成材料力學特徵...")
            X_processed = self.create_material_features(X_processed)
        
        # 2. 生成特徵交互項
        if interaction_features:
            logger.info("生成特徵交互項...")
            X_processed = self.create_interaction_features(X_processed, degree=2, interaction_only=True)
        
        # 3. 生成時間序列特徵
        if time_series_features:
            logger.info("生成時間序列特徵...")
            X_processed = self.create_time_series_features(X_processed)
        
        # 4. 特徵選擇 (需要目標變數)
        if feature_selection and y is not None:
            logger.info(f"使用 {selection_method} 方法進行特徵選擇...")
            X_processed = self.select_features(X_processed, y, method=selection_method, n_features=n_selected_features)
        
        # 5. 特徵縮放
        if scaling:
            logger.info(f"使用 {scaling_method} 方法進行特徵縮放...")
            X_processed = self.scale_features(X_processed, method=scaling_method, fit=True)
        
        logger.info(f"特徵工程流程完成。原始特徵數: {X.shape[1]}, 處理後特徵數: {X_processed.shape[1]}")
        
        return X_processed
    
    def transform_new_data(self, X_new, feature_selection=True, scaling=True):
        """
        轉換新數據，應用與訓練數據相同的特徵工程步驟
        
        參數:
            X_new (pandas.DataFrame): 新的特徵數據框
            feature_selection (bool): 是否應用特徵選擇，默認為True
            scaling (bool): 是否應用特徵縮放，默認為True
            
        返回:
            pandas.DataFrame: 轉換後的數據框
        """
        if not hasattr(self, 'selected_features') and feature_selection:
            logger.warning("尚未執行特徵選擇，無法應用於新數據")
            feature_selection = False
        
        if not hasattr(self, 'scaler') and scaling:
            logger.warning("尚未執行特徵縮放，無法應用於新數據")
            scaling = False
        
        # 複製數據避免修改原始數據
        X_transformed = X_new.copy()
        
        logger.info("開始轉換新數據...")
        
        # 1. 生成材料力學特徵
        X_transformed = self.create_material_features(X_transformed)
        
        # 2. 應用特徵選擇 (如果已經訓練過)
        if feature_selection and hasattr(self, 'selected_features'):
            # 確保所有選定的特徵都在新數據中
            missing_features = [f for f in self.selected_features if f not in X_transformed.columns]
            if missing_features:
                logger.warning(f"新數據中缺少以下特徵: {missing_features}")
                # 為缺失的特徵填充零值
                for feature in missing_features:
                    X_transformed[feature] = 0
            
            X_transformed = X_transformed[self.selected_features]
        
        # 3. 應用特徵縮放 (如果已經訓練過)
        if scaling and hasattr(self, 'scaler'):
            X_transformed = pd.DataFrame(
                self.scaler.transform(X_transformed),
                columns=X_transformed.columns,
                index=X_transformed.index
            )
        
        logger.info(f"新數據轉換完成。特徵數: {X_transformed.shape[1]}")
        
        return X_transformed


# 用法示例
if __name__ == "__main__":
    # 導入示例數據 (如果有的話)
    try:
        import os
        import pandas as pd
        
        # 嘗試載入示例數據
        data_path = os.path.join('data', 'processed', 'Training_data_warpage_final.csv')
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            
            # 分離特徵和目標變數
            y = data['Nf_pred (cycles)'] if 'Nf_pred (cycles)' in data.columns else None
            X = data.drop(['Nf_pred (cycles)'], axis=1) if y is not None else data
            
            # 初始化特徵工程類
            engineer = FeatureEngineer(random_state=42)
            
            # 應用特徵工程流程
            X_processed = engineer.apply_feature_engineering_pipeline(
                X, y, 
                material_features=True, 
                interaction_features=False, 
                time_series_features=True,
                feature_selection=True if y is not None else False
            )
            
            # 顯示處理結果
            print("\n原始數據形狀:", X.shape)
            print("處理後數據形狀:", X_processed.shape)
            
            if hasattr(engineer, 'selected_features'):
                print("\n選擇的重要特徵 (前10個):")
                for i, feature in enumerate(engineer.selected_features[:10]):
                    print(f"{i+1}. {feature}")
                    
            # 如果有目標變數，計算特徵重要性
            if y is not None:
                importance_df = engineer.get_feature_importance(X, y)
                print("\n特徵重要性 (前10個):")
                print(importance_df.head(10))
                
        else:
            print(f"示例數據文件 {data_path} 不存在。")
    except Exception as e:
        print(f"運行示例時出錯: {str(e)}")
        print("如果您想使用此模組，請提供適當的數據文件或通過API進行調用。")