#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
model.py - 焊點疲勞壽命預測模型
本模組提供多種機器學習模型用於預測銲錫接點的疲勞壽命，
支援模型訓練、評估、儲存和載入功能。

主要功能:
1. 提供多種回歸模型用於壽命預測
2. 混合模型整合，提高預測準確性
3. 支援物理信息神經網絡(PINN)與機器學習結合
4. 提供模型訓練和評估功能
5. 支援模型序列化與反序列化
"""

import os
import pickle
import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Union, Optional, Any

# 機器學習模型
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

# 神經網絡模型
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FatigueLifePredictor:
    """
    焊點疲勞壽命預測器，提供多種模型的訓練、評估和預測功能
    """
    
    def __init__(self, random_state: int = 42):
        """
        初始化疲勞壽命預測器
        
        參數:
            random_state (int): 隨機種子，用於可重複的隨機操作
        """
        self.random_state = random_state
        self.models = {}  # 儲存訓練好的模型
        self.best_model = None  # 最佳模型
        self.best_model_name = None  # 最佳模型名稱
        self.feature_names = None  # 特徵名稱列表
        self.scaler_X = None  # 特徵縮放器
        self.scaler_y = None  # 目標變數縮放器
        self.model_metrics = {}  # 模型評估指標
        
    def _create_models(self) -> Dict[str, Any]:
        """
        創建各種回歸模型
        
        返回:
            Dict[str, Any]: 模型名稱與模型對象的字典
        """
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'lasso': Lasso(alpha=0.1, random_state=self.random_state),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
            'svr': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'knn': KNeighborsRegressor(n_neighbors=5),
            'gpr': GaussianProcessRegressor(
                kernel=RBF() + Matern(nu=1.5) + RationalQuadratic(),
                random_state=self.random_state
            )
        }
        
        # 如果PyTorch可用，添加神經網絡模型
        if TORCH_AVAILABLE:
            models['neural_network'] = MLPRegressor(
                input_size=10,  # 將在訓練時調整
                hidden_sizes=[64, 32],
                output_size=1,
                random_state=self.random_state
            )
            
            # 如果資料量足夠，添加PINN模型
            models['pinn'] = PhysicsInformedNN(
                input_size=10,  # 將在訓練時調整
                hidden_sizes=[64, 32],
                output_size=1,
                random_state=self.random_state
            )
        
        return models
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, cv: int = 5,
                     tune_hyperparams: bool = True) -> Dict[str, Dict[str, float]]:
        """
        訓練多個回歸模型並評估其性能
        
        參數:
            X (pandas.DataFrame): 特徵數據框
            y (pandas.Series): 目標變數
            cv (int): 交叉驗證的折數
            tune_hyperparams (bool): 是否調整超參數
            
        返回:
            Dict[str, Dict[str, float]]: 模型名稱與其評估指標的字典
        """
        # 保存特徵名稱
        self.feature_names = X.columns.tolist()
        
        # 創建各種模型
        models = self._create_models()
        
        # 保存評估指標
        metrics = {}
        
        # 使用交叉驗證評估每個模型
        for name, model in models.items():
            logger.info(f"訓練模型: {name}")
            
            try:
                # 檢查模型是否為神經網絡
                if TORCH_AVAILABLE and isinstance(model, (MLPRegressor, PhysicsInformedNN)):
                    # 針對神經網絡模型的特殊處理
                    model.input_size = X.shape[1]  # 設置輸入特徵數量
                    model_metrics = self._train_neural_network(model, X, y, cv)
                else:
                    # 針對一般機器學習模型
                    if tune_hyperparams:
                        model = self._tune_hyperparameters(name, model, X, y, cv)
                    
                    # 使用交叉驗證評估模型
                    start_time = time.time()
                    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv, 
                                                      scoring='neg_mean_squared_error', 
                                                      n_jobs=-1))
                    cv_mae = -cross_val_score(model, X, y, cv=cv, 
                                            scoring='neg_mean_absolute_error', 
                                            n_jobs=-1)
                    cv_r2 = cross_val_score(model, X, y, cv=cv, 
                                          scoring='r2', 
                                          n_jobs=-1)
                    
                    # 在整個數據集上訓練最終模型
                    model.fit(X, y)
                    self.models[name] = model
                    
                    # 記錄評估指標
                    model_metrics = {
                        'rmse': cv_rmse.mean(),
                        'rmse_std': cv_rmse.std(),
                        'mae': cv_mae.mean(),
                        'mae_std': cv_mae.std(),
                        'r2': cv_r2.mean(),
                        'r2_std': cv_r2.std(),
                        'training_time': time.time() - start_time
                    }
                
                # 保存模型評估指標
                metrics[name] = model_metrics
                logger.info(f"模型 {name} 的評估指標: RMSE={model_metrics['rmse']:.4f}±{model_metrics['rmse_std']:.4f}, "
                          f"R²={model_metrics['r2']:.4f}±{model_metrics['r2_std']:.4f}")
                
            except Exception as e:
                logger.error(f"訓練模型 {name} 時出錯: {str(e)}")
                continue
        
        # 找出性能最佳的模型
        if metrics:
            self.best_model_name = min(metrics.items(), key=lambda x: x[1]['rmse'])[0]
            self.best_model = self.models[self.best_model_name]
            logger.info(f"最佳模型是 {self.best_model_name}, RMSE={metrics[self.best_model_name]['rmse']:.4f}")
        
        # 創建投票回歸器
        if len(self.models) >= 3:
            try:
                logger.info("創建投票回歸器...")
                estimators = [(name, model) for name, model in self.models.items() 
                             if not (TORCH_AVAILABLE and isinstance(model, (MLPRegressor, PhysicsInformedNN)))]
                
                if estimators:
                    voting_regressor = VotingRegressor(estimators=estimators)
                    
                    # 評估投票回歸器
                    start_time = time.time()
                    cv_rmse = np.sqrt(-cross_val_score(voting_regressor, X, y, cv=cv, 
                                                      scoring='neg_mean_squared_error', 
                                                      n_jobs=-1))
                    cv_mae = -cross_val_score(voting_regressor, X, y, cv=cv, 
                                            scoring='neg_mean_absolute_error', 
                                            n_jobs=-1)
                    cv_r2 = cross_val_score(voting_regressor, X, y, cv=cv, 
                                          scoring='r2', 
                                          n_jobs=-1)
                    
                    # 在整個數據集上訓練最終模型
                    voting_regressor.fit(X, y)
                    self.models['voting'] = voting_regressor
                    
                    # 記錄評估指標
                    voting_metrics = {
                        'rmse': cv_rmse.mean(),
                        'rmse_std': cv_rmse.std(),
                        'mae': cv_mae.mean(),
                        'mae_std': cv_mae.std(),
                        'r2': cv_r2.mean(),
                        'r2_std': cv_r2.std(),
                        'training_time': time.time() - start_time
                    }
                    
                    metrics['voting'] = voting_metrics
                    logger.info(f"投票回歸器的評估指標: RMSE={voting_metrics['rmse']:.4f}±{voting_metrics['rmse_std']:.4f}, "
                              f"R²={voting_metrics['r2']:.4f}±{voting_metrics['r2_std']:.4f}")
                    
                    # 如果投票回歸器是最佳模型，則更新最佳模型
                    if voting_metrics['rmse'] < metrics[self.best_model_name]['rmse']:
                        self.best_model_name = 'voting'
                        self.best_model = voting_regressor
                        logger.info(f"投票回歸器是最佳模型, RMSE={voting_metrics['rmse']:.4f}")
            except Exception as e:
                logger.error(f"創建投票回歸器時出錯: {str(e)}")
        
        # 保存所有模型的評估指標
        self.model_metrics = metrics
        
        # 如果資料量允許，嘗試創建與訓練混合模型
        if X.shape[0] >= 50 and X.shape[1] >= 5 and TORCH_AVAILABLE:
            try:
                logger.info("創建混合PINN-LSTM模型...")
                self._train_hybrid_model(X, y, cv)
            except Exception as e:
                logger.error(f"創建混合模型時出錯: {str(e)}")
        
        return metrics
    
    def _tune_hyperparameters(self, name: str, model: Any, X: pd.DataFrame, y: pd.Series, cv: int) -> Any:
        """
        調整模型超參數
        
        參數:
            name (str): 模型名稱
            model (Any): 模型對象
            X (pandas.DataFrame): 特徵數據框
            y (pandas.Series): 目標變數
            cv (int): 交叉驗證的折數
            
        返回:
            Any: 調整超參數後的模型對象
        """
        logger.info(f"調整 {name} 模型的超參數...")
        
        # 為不同模型定義超參數網格
        param_grids = {
            'ridge': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0]
            },
            'elastic_net': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            },
            'svr': {
                'C': [0.1, 1.0, 10.0],
                'epsilon': [0.01, 0.1, 1.0],
                'kernel': ['linear', 'rbf']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # 1: Manhattan, 2: Euclidean
            }
        }
        
        # 檢查是否存在此模型的超參數網格
        if name not in param_grids:
            logger.info(f"沒有為 {name} 定義超參數網格，使用默認參數")
            return model
        
        # 數據量小於50時，簡化超參數網格以加快調優速度
        if X.shape[0] < 50:
            for key in param_grids[name].keys():
                param_grids[name][key] = param_grids[name][key][:2]  # 只使用前兩個值
        
        # 處理小樣本數據集：如果樣本數小於5*cv，減少交叉驗證折數
        actual_cv = min(cv, max(2, X.shape[0] // 5))
        if actual_cv < cv:
            logger.warning(f"樣本數較少，交叉驗證折數從 {cv} 減少到 {actual_cv}")
        
        # 使用網格搜索調整超參數
        grid_search = GridSearchCV(
            model, param_grids[name], 
            cv=actual_cv, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"最佳超參數: {grid_search.best_params_}")
        logger.info(f"最佳RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
        
        return grid_search.best_estimator_
    
    def _train_neural_network(self, model: Any, X: pd.DataFrame, y: pd.Series, cv: int) -> Dict[str, float]:
        """
        訓練神經網絡模型
        
        參數:
            model (Any): 神經網絡模型對象
            X (pandas.DataFrame): 特徵數據框
            y (pandas.Series): 目標變數
            cv (int): 交叉驗證的折數
            
        返回:
            Dict[str, float]: 模型評估指標
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安裝，無法訓練神經網絡模型")
        
        # 轉換為numpy數組
        X_np = X.values
        y_np = y.values.reshape(-1, 1)
        
        # 設置交叉驗證
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # 記錄評估指標
        rmse_list = []
        mae_list = []
        r2_list = []
        
        # 開始計時
        start_time = time.time()
        
        # 交叉驗證
        for train_idx, test_idx in kf.split(X_np):
            X_train, X_test = X_np[train_idx], X_np[test_idx]
            y_train, y_test = y_np[train_idx], y_np[test_idx]
            
            # 訓練模型
            model.fit(X_train, y_train)
            
            # 預測並評估
            y_pred = model.predict(X_test)
            
            # 計算評估指標
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            rmse_list.append(rmse)
            mae_list.append(mae)
            r2_list.append(r2)
        
        # 在整個數據集上訓練最終模型
        model.fit(X_np, y_np)
        self.models[model.__class__.__name__] = model
        
        # 計算評估指標平均值和標準差
        metrics = {
            'rmse': np.mean(rmse_list),
            'rmse_std': np.std(rmse_list),
            'mae': np.mean(mae_list),
            'mae_std': np.std(mae_list),
            'r2': np.mean(r2_list),
            'r2_std': np.std(r2_list),
            'training_time': time.time() - start_time
        }
        
        return metrics
    
    def _train_hybrid_model(self, X: pd.DataFrame, y: pd.Series, cv: int) -> None:
        """
        訓練混合PINN-LSTM模型
        
        參數:
            X (pandas.DataFrame): 特徵數據框
            y (pandas.Series): 目標變數
            cv (int): 交叉驗證的折數
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安裝，無法訓練混合模型")
        
        # 創建混合模型
        hybrid_model = HybridPINNLSTM(
            input_size=X.shape[1],
            hidden_sizes=[64, 32],
            output_size=1,
            lstm_hidden_size=32,
            random_state=self.random_state
        )
        
        # 訓練和評估混合模型
        metrics = self._train_neural_network(hybrid_model, X, y, cv)
        
        # 保存模型和評估指標
        self.models['hybrid_pinn_lstm'] = hybrid_model
        self.model_metrics['hybrid_pinn_lstm'] = metrics
        
        logger.info(f"混合PINN-LSTM模型評估指標: RMSE={metrics['rmse']:.4f}±{metrics['rmse_std']:.4f}, "
                  f"R²={metrics['r2']:.4f}±{metrics['r2_std']:.4f}")
        
        # 如果混合模型是最佳模型，則更新最佳模型
        if metrics['rmse'] < self.model_metrics[self.best_model_name]['rmse']:
            self.best_model_name = 'hybrid_pinn_lstm'
            self.best_model = hybrid_model
            logger.info(f"混合PINN-LSTM是最佳模型, RMSE={metrics['rmse']:.4f}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], model_name: Optional[str] = None) -> np.ndarray:
        """
        使用指定模型或最佳模型進行預測
        
        參數:
            X (pandas.DataFrame or numpy.ndarray): 特徵數據
            model_name (str, optional): 要使用的模型名稱，默認為None表示使用最佳模型
            
        返回:
            numpy.ndarray: 預測結果
        """
        # 確保模型已經訓練
        if not self.models:
            raise ValueError("模型尚未訓練，請先調用train_models方法")
        
        # 確定要使用的模型
        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(f"模型 {model_name} 不存在，可用的模型有: {list(self.models.keys())}")
            model = self.models[model_name]
        else:
            if self.best_model is None:
                raise ValueError("尚未確定最佳模型，請先調用train_models方法")
            model = self.best_model
        
        # 確保輸入是pandas DataFrame
        if isinstance(X, np.ndarray):
            if self.feature_names is None:
                raise ValueError("特徵名稱未知，無法將numpy數組轉換為DataFrame")
            if X.shape[1] != len(self.feature_names):
                raise ValueError(f"輸入特徵數量 {X.shape[1]} 與訓練時的特徵數量 {len(self.feature_names)} 不匹配")
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # 檢查特徵名稱是否匹配
        if not all(feature in X.columns for feature in self.feature_names):
            missing_features = [feature for feature in self.feature_names if feature not in X.columns]
            raise ValueError(f"輸入數據缺少以下特徵: {missing_features}")
        
        # 只保留訓練時使用的特徵
        X = X[self.feature_names]
        
        # 進行預測
        try:
            # 檢查模型是否為神經網絡
            if TORCH_AVAILABLE and isinstance(model, (MLPRegressor, PhysicsInformedNN, HybridPINNLSTM)):
                return model.predict(X.values)
            else:
                return model.predict(X)
        except Exception as e:
            logger.error(f"預測時出錯: {str(e)}")
            raise
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, model_name: Optional[str] = None) -> Dict[str, float]:
        """
        評估指定模型或最佳模型在測試集上的性能
        
        參數:
            X (pandas.DataFrame): 測試集特徵
            y (pandas.Series): 測試集目標變數
            model_name (str, optional): 要評估的模型名稱，默認為None表示評估最佳模型
            
        返回:
            Dict[str, float]: 評估指標
        """
        # 進行預測
        y_pred = self.predict(X, model_name)
        
        # 計算評估指標
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # 建立評估報告
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # 獲取使用的模型名稱
        model_name_used = model_name if model_name is not None else self.best_model_name
        
        logger.info(f"模型 {model_name_used} 在測試集上的評估指標: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        return metrics
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        獲取特徵重要性（僅適用於具有feature_importances_屬性的模型）
        
        參數:
            model_name (str, optional): 要獲取特徵重要性的模型名稱，默認為None表示使用最佳模型
            
        返回:
            pandas.DataFrame: 包含特徵名稱和重要性的數據框
        """
        # 確定要使用的模型
        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(f"模型 {model_name} 不存在，可用的模型有: {list(self.models.keys())}")
            model = self.models[model_name]
        else:
            if self.best_model is None:
                raise ValueError("尚未確定最佳模型，請先調用train_models方法")
            model = self.best_model
        
        # 檢查模型是否具有feature_importances_屬性
        if not hasattr(model, 'feature_importances_'):
            # 檢查是否為投票回歸器
            if hasattr(model, 'estimators_'):
                # 嘗試從第一個具有feature_importances_的估計器獲取
                for estimator_name, estimator in model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        logger.info(f"使用估計器 {estimator_name} 的特徵重要性")
                        importances = estimator.feature_importances_
                        break
                else:
                    raise ValueError("沒有找到具有feature_importances_屬性的估計器")
            else:
                raise ValueError(f"模型 {model_name if model_name is not None else self.best_model_name} 不支援獲取特徵重要性")
        else:
            importances = model.feature_importances_
        
        # 創建特徵重要性數據框
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        # 按重要性降序排序
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save(self, filepath: str) -> None:
        """
        保存模型到檔案
        
        參數:
            filepath (str): 檔案路徑
        """
        # 確保目錄存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # 將模型和相關資訊保存到檔案
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'best_model_name': self.best_model_name,
                'feature_names': self.feature_names,
                'model_metrics': self.model_metrics,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'random_state': self.random_state
            }, f)
            
        logger.info(f"模型已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FatigueLifePredictor':
        """
        從檔案載入模型
        
        參數:
            filepath (str): 檔案路徑
            
        返回:
            FatigueLifePredictor: 載入的預測器實例
        """
        # 檢查檔案是否存在
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"檔案 {filepath} 不存在")
        
        # 從檔案載入模型和相關資訊
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # 創建新的預測器實例
        predictor = cls(random_state=data['random_state'])
        predictor.models = data['models']
        predictor.best_model_name = data['best_model_name']
        predictor.best_model = data['models'][data['best_model_name']] if data['best_model_name'] else None
        predictor.feature_names = data['feature_names']
        predictor.model_metrics = data['model_metrics']
        predictor.scaler_X = data['scaler_X']
        predictor.scaler_y = data['scaler_y']
        
        logger.info(f"從 {filepath} 載入模型")
        
        return predictor


# 為了支援小樣本資料集，我們實現一個簡單的神經網絡回歸器
class MLPRegressor:
    """
    多層感知器回歸器，適用於小樣本資料集的神經網絡模型
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 random_state: int = 42, learning_rate: float = 0.001, 
                 max_epochs: int = 1000, batch_size: int = 16, 
                 early_stopping_patience: int = 50, l2_reg: float = 0.0001):
        """
        初始化多層感知器回歸器
        
        參數:
            input_size (int): 輸入特徵數量
            hidden_sizes (List[int]): 隱藏層神經元數量列表
            output_size (int): 輸出維度
            random_state (int): 隨機種子
            learning_rate (float): 學習率
            max_epochs (int): 最大訓練周期數
            batch_size (int): 批次大小
            early_stopping_patience (int): 早停耐心值
            l2_reg (float): L2正則化係數
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安裝，無法使用神經網絡模型")
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.l2_reg = l2_reg
        
        # 設置隨機種子
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # 創建網絡模型
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=l2_reg)
        self.loss_fn = nn.MSELoss()
        
        # 訓練記錄
        self.train_losses = []
        self.val_losses = []
        self.is_fitted = False
    
    def _build_model(self) -> nn.Module:
        """
        構建神經網絡模型
        
        返回:
            torch.nn.Module: PyTorch神經網絡模型
        """
        # 為小樣本資料集設計的簡單神經網絡
        class MLPModel(nn.Module):
            def __init__(self, input_size, hidden_sizes, output_size):
                super(MLPModel, self).__init__()
                
                self.layers = nn.ModuleList()
                layer_sizes = [input_size] + hidden_sizes
                
                # 添加隱藏層
                for i in range(len(layer_sizes) - 1):
                    self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                    self.layers.append(nn.ReLU())
                    self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                    self.layers.append(nn.Dropout(0.2))  # 添加Dropout以減少過擬合
                
                # 輸出層
                self.output_layer = nn.Linear(layer_sizes[-1], output_size)
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.output_layer(x)
        
        return MLPModel(self.input_size, self.hidden_sizes, self.output_size)
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> 'MLPRegressor':
        """
        訓練神經網絡模型
        
        參數:
            X (numpy.ndarray): 特徵數據
            y (numpy.ndarray): 目標變數
            validation_split (float): 驗證集比例
            
        返回:
            MLPRegressor: 訓練後的模型實例
        """
        # 輸入數據檢查和預處理
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # 對於小樣本資料集，適當減少batch_size和early_stopping_patience
        if X.shape[0] < 50:
            batch_size = min(8, X.shape[0])
            early_stopping_patience = 30
        else:
            batch_size = self.batch_size
            early_stopping_patience = self.early_stopping_patience
        
        # 調整輸入大小
        if X.shape[1] != self.input_size:
            logger.warning(f"輸入特徵數量 {X.shape[1]} 與模型輸入大小 {self.input_size} 不符，調整模型結構")
            self.input_size = X.shape[1]
            self.model = self._build_model()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        
        # 確保y是二維數組
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # 劃分訓練集和驗證集
        # 為小樣本資料集特別處理：如果樣本數小於10，不使用驗證集
        if X.shape[0] < 10:
            X_train, y_train = X, y
            X_val, y_val = X, y  # 使用訓練集作為驗證集
            logger.warning("樣本數量過少，使用全部數據訓練，並以訓練集作為驗證集")
        else:
            # 隨機劃分訓練集和驗證集
            indices = np.random.permutation(X.shape[0])
            val_size = int(X.shape[0] * validation_split)
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
        
        # 轉換為PyTorch張量
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        
        # 創建數據加載器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 訓練模型
        best_val_loss = float('inf')
        patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(self.max_epochs):
            # 訓練階段
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # 驗證階段
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.loss_fn(val_outputs, y_val_tensor).item()
                self.val_losses.append(val_loss)
            
            # 早停檢查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"早停觸發於第 {epoch+1} 輪，最佳驗證損失: {best_val_loss:.6f}")
                    break
            
            # 每20輪輸出一次訓練進度
            if (epoch + 1) % 20 == 0:
                logger.info(f"輪次 {epoch+1}/{self.max_epochs}, 訓練損失: {train_loss:.6f}, 驗證損失: {val_loss:.6f}")
        
        # 恢復最佳模型
        if 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
        
        # 設置為已訓練狀態
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用訓練好的模型進行預測
        
        參數:
            X (numpy.ndarray): 特徵數據
            
        返回:
            numpy.ndarray: 預測結果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用fit方法")
        
        # 輸入數據檢查和預處理
        X = X.astype(np.float32)
        
        # 確保輸入特徵數量與模型匹配
        if X.shape[1] != self.input_size:
            raise ValueError(f"輸入特徵數量 {X.shape[1]} 與模型輸入大小 {self.input_size} 不符")
        
        # 轉換為PyTorch張量
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # 預測
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()
        
        return predictions


class PhysicsInformedNN:
    """
    物理信息神經網絡(PINN)，結合物理知識的神經網絡模型
    用於銲錫接點疲勞壽命預測的物理信息神經網絡模型
    
    基於Coffin-Manson關係: N_f = C * (Δε_p)^(-n)
    C和n為材料參數，結合神經網絡學習
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 random_state: int = 42, learning_rate: float = 0.001, 
                 max_epochs: int = 1000, batch_size: int = 16, 
                 early_stopping_patience: int = 50, l2_reg: float = 0.0001,
                 physics_weight: float = 0.5):
        """
        初始化物理信息神經網絡
        
        參數:
            input_size (int): 輸入特徵數量
            hidden_sizes (List[int]): 隱藏層神經元數量列表
            output_size (int): 輸出維度
            random_state (int): 隨機種子
            learning_rate (float): 學習率
            max_epochs (int): 最大訓練周期數
            batch_size (int): 批次大小
            early_stopping_patience (int): 早停耐心值
            l2_reg (float): L2正則化係數
            physics_weight (float): 物理損失權重
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安裝，無法使用神經網絡模型")
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.l2_reg = l2_reg
        self.physics_weight = physics_weight
        
        # 設置隨機種子
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # 創建網絡模型
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=l2_reg)
        self.mse_loss = nn.MSELoss()
        
        # 物理參數 (Coffin-Manson關係的C和n)
        # 會在訓練過程中學習這些參數
        self.C = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        self.n = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        
        # 訓練記錄
        self.train_losses = []
        self.val_losses = []
        self.physics_losses = []
        self.is_fitted = False
    
    def _build_model(self) -> nn.Module:
        """
        構建神經網絡模型
        
        返回:
            torch.nn.Module: PyTorch神經網絡模型
        """
        # 為物理信息神經網絡設計的模型
        class PINNModel(nn.Module):
            def __init__(self, input_size, hidden_sizes, output_size):
                super(PINNModel, self).__init__()
                
                self.layers = nn.ModuleList()
                layer_sizes = [input_size] + hidden_sizes
                
                # 添加隱藏層
                for i in range(len(layer_sizes) - 1):
                    self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                    self.layers.append(nn.Tanh())  # 使用Tanh激活函數以更好地擬合物理行為
                
                # 輸出層
                self.output_layer = nn.Linear(layer_sizes[-1], output_size)
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.output_layer(x)
        
        return PINNModel(self.input_size, self.hidden_sizes, self.output_size)
    
    def _physics_loss(self, X: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        計算物理損失
        
        參數:
            X (torch.Tensor): 輸入特徵
            y_pred (torch.Tensor): 預測的疲勞壽命
            
        返回:
            torch.Tensor: 物理損失
        """
        # 假設X中包含應變相關的特徵
        # 嘗試從輸入特徵中提取應變相關特徵 (例如等效塑性應變或塑性功)
        strain_idx = None
        # 檢查是否有應變相關的特徵索引
        if hasattr(self, 'strain_feature_idx'):
            strain_idx = self.strain_feature_idx
        else:
            # 假設最後一個特徵是與應變相關的
            strain_idx = -1
            
        # 獲取應變相關特徵
        strain_feature = X[:, strain_idx].reshape(-1, 1)
        
        # 使用Coffin-Manson關係計算理論疲勞壽命
        # N_f = C * (Δε_p)^(-n)
        theory_cycles = self.C * torch.pow(strain_feature, -self.n)
        
        # 計算與神經網絡預測的疲勞壽命之間的差異
        physics_loss = self.mse_loss(y_pred, theory_cycles)
        
        return physics_loss
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2,
           strain_feature_idx: int = None) -> 'PhysicsInformedNN':
        """
        訓練物理信息神經網絡模型
        
        參數:
            X (numpy.ndarray): 特徵數據
            y (numpy.ndarray): 目標變數
            validation_split (float): 驗證集比例
            strain_feature_idx (int, optional): 應變相關特徵的索引，默認為None
            
        返回:
            PhysicsInformedNN: 訓練後的模型實例
        """
        # 設置應變特徵索引
        if strain_feature_idx is not None:
            self.strain_feature_idx = strain_feature_idx
        else:
            # 嘗試自動檢測應變相關特徵
            # 假設與"Strain"或"NLPLWK"相關的特徵
            strain_keywords = ["strain", "nlplwk"]
            if hasattr(self, 'feature_names') and self.feature_names is not None:
                for i, feature in enumerate(self.feature_names):
                    if any(keyword in feature.lower() for keyword in strain_keywords):
                        self.strain_feature_idx = i
                        break
            
            # 如果未找到，使用最後一個特徵
            if not hasattr(self, 'strain_feature_idx'):
                self.strain_feature_idx = X.shape[1] - 1
        
        # 輸入數據檢查和預處理
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # 對於小樣本資料集，適當減少batch_size和early_stopping_patience
        if X.shape[0] < 50:
            batch_size = min(8, X.shape[0])
            early_stopping_patience = 30
        else:
            batch_size = self.batch_size
            early_stopping_patience = self.early_stopping_patience
        
        # 調整輸入大小
        if X.shape[1] != self.input_size:
            logger.warning(f"輸入特徵數量 {X.shape[1]} 與模型輸入大小 {self.input_size} 不符，調整模型結構")
            self.input_size = X.shape[1]
            self.model = self._build_model()
            self.optimizer = optim.Adam(
                list(self.model.parameters()) + [self.C, self.n], 
                lr=self.learning_rate, 
                weight_decay=self.l2_reg
            )
        
        # 確保y是二維數組
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # 劃分訓練集和驗證集
        # 為小樣本資料集特別處理：如果樣本數小於10，不使用驗證集
        if X.shape[0] < 10:
            X_train, y_train = X, y
            X_val, y_val = X, y  # 使用訓練集作為驗證集
            logger.warning("樣本數量過少，使用全部數據訓練，並以訓練集作為驗證集")
        else:
            # 隨機劃分訓練集和驗證集
            indices = np.random.permutation(X.shape[0])
            val_size = int(X.shape[0] * validation_split)
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
        
        # 轉換為PyTorch張量
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        
        # 創建數據加載器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 訓練模型
        best_val_loss = float('inf')
        patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.physics_losses = []
        
        for epoch in range(self.max_epochs):
            # 訓練階段
            self.model.train()
            train_loss = 0.0
            physics_loss_sum = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # 前向傳播
                outputs = self.model(batch_X)
                
                # 計算數據損失
                data_loss = self.mse_loss(outputs, batch_y)
                
                # 計算物理損失
                physics_loss = self._physics_loss(batch_X, outputs)
                physics_loss_sum += physics_loss.item()
                
                # 總損失 = 數據損失 + 物理損失權重 * 物理損失
                total_loss = data_loss + self.physics_weight * physics_loss
                
                # 反向傳播
                total_loss.backward()
                self.optimizer.step()
                
                train_loss += total_loss.item()
            
            train_loss /= len(train_loader)
            physics_loss_avg = physics_loss_sum / len(train_loader)
            
            self.train_losses.append(train_loss)
            self.physics_losses.append(physics_loss_avg)
            
            # 驗證階段
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_data_loss = self.mse_loss(val_outputs, y_val_tensor).item()
                val_physics_loss = self._physics_loss(X_val_tensor, val_outputs).item()
                val_loss = val_data_loss + self.physics_weight * val_physics_loss
                self.val_losses.append(val_loss)
            
            # 早停檢查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                best_C = self.C.detach().clone()
                best_n = self.n.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"早停觸發於第 {epoch+1} 輪，最佳驗證損失: {best_val_loss:.6f}")
                    break
            
            # 每20輪輸出一次訓練進度
            if (epoch + 1) % 20 == 0:
                logger.info(f"輪次 {epoch+1}/{self.max_epochs}, 訓練損失: {train_loss:.6f}, "
                          f"物理損失: {physics_loss_avg:.6f}, 驗證損失: {val_loss:.6f}, "
                          f"C={self.C.item():.4f}, n={self.n.item():.4f}")
        
        # 恢復最佳模型
        if 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
            self.C = nn.Parameter(best_C)
            self.n = nn.Parameter(best_n)
        
        # 設置為已訓練狀態
        self.is_fitted = True
        
        logger.info(f"PINN模型訓練完成，C={self.C.item():.4f}, n={self.n.item():.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用訓練好的模型進行預測
        
        參數:
            X (numpy.ndarray): 特徵數據
            
        返回:
            numpy.ndarray: 預測結果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用fit方法")
        
        # 輸入數據檢查和預處理
        X = X.astype(np.float32)
        
        # 確保輸入特徵數量與模型匹配
        if X.shape[1] != self.input_size:
            raise ValueError(f"輸入特徵數量 {X.shape[1]} 與模型輸入大小 {self.input_size} 不符")
        
        # 轉換為PyTorch張量
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # 預測
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()
        
        return predictions
    
    def get_physics_params(self) -> Dict[str, float]:
        """
        獲取學習到的物理參數
        
        返回:
            Dict[str, float]: 包含物理參數的字典
        """
        return {
            'C': self.C.item(),
            'n': self.n.item()
        }


class HybridPINNLSTM:
    """
    混合物理信息神經網絡(PINN)和長短期記憶網絡(LSTM)模型
    用於結合靜態特徵和時間序列特徵預測銲錫接點疲勞壽命
    
    PINN分支處理靜態特徵並確保與物理規律一致
    LSTM分支處理時間序列特徵以捕捉動態行為
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 lstm_hidden_size: int = 32, random_state: int = 42, 
                 learning_rate: float = 0.001, max_epochs: int = 1000, 
                 batch_size: int = 16, early_stopping_patience: int = 50, 
                 l2_reg: float = 0.0001, physics_weight: float = 0.3):
        """
        初始化混合PINN-LSTM模型
        
        參數:
            input_size (int): 輸入特徵數量
            hidden_sizes (List[int]): PINN分支隱藏層神經元數量列表
            output_size (int): 輸出維度
            lstm_hidden_size (int): LSTM隱藏狀態大小
            random_state (int): 隨機種子
            learning_rate (float): 學習率
            max_epochs (int): 最大訓練周期數
            batch_size (int): 批次大小
            early_stopping_patience (int): 早停耐心值
            l2_reg (float): L2正則化係數
            physics_weight (float): 物理損失權重
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安裝，無法使用混合神經網絡模型")
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.output_size = output_size
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.l2_reg = l2_reg
        self.physics_weight = physics_weight
        
        # 設置隨機種子
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # 創建網絡模型
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=l2_reg)
        self.mse_loss = nn.MSELoss()
        
        # 物理參數 (Coffin-Manson關係的C和n)
        self.C = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        self.n = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        
        # 設置特徵分組索引
        self.static_feature_indices = None  # 靜態特徵索引
        self.time_feature_indices = None    # 時間序列特徵索引
        self.strain_feature_idx = None      # 應變相關特徵索引
        
        # 訓練記錄
        self.train_losses = []
        self.val_losses = []
        self.physics_losses = []
        self.is_fitted = False
    
    def _build_model(self) -> nn.Module:
        """
        構建混合神經網絡模型
        
        返回:
            torch.nn.Module: PyTorch混合神經網絡模型
        """
        # 為混合PINN-LSTM模型設計的網絡架構
        class HybridModel(nn.Module):
            def __init__(self, input_size, hidden_sizes, lstm_hidden_size, output_size):
                super(HybridModel, self).__init__()
                
                # PINN分支 (用於靜態特徵)
                self.pinn_layers = nn.ModuleList()
                pinn_layer_sizes = [input_size] + hidden_sizes
                
                for i in range(len(pinn_layer_sizes) - 1):
                    self.pinn_layers.append(nn.Linear(pinn_layer_sizes[i], pinn_layer_sizes[i+1]))
                    self.pinn_layers.append(nn.Tanh())
                
                # LSTM分支 (用於時間序列特徵)
                self.lstm = nn.LSTM(
                    input_size=input_size,  # 實際使用時會根據時間特徵數量調整
                    hidden_size=lstm_hidden_size,
                    batch_first=True
                )
                
                # 合併層
                self.merge_layer = nn.Linear(pinn_layer_sizes[-1] + lstm_hidden_size, pinn_layer_sizes[-1])
                self.merge_activation = nn.ReLU()
                
                # 輸出層
                self.output_layer = nn.Linear(pinn_layer_sizes[-1], output_size)
            
            def forward(self, x_static, x_time=None):
                # 處理靜態特徵 (PINN分支)
                pinn_out = x_static
                for layer in self.pinn_layers:
                    pinn_out = layer(pinn_out)
                
                # 如果有時間序列特徵，處理LSTM分支
                if x_time is not None:
                    lstm_out, _ = self.lstm(x_time)
                    lstm_out = lstm_out[:, -1, :]  # 只使用最後一個時間步的輸出
                    
                    # 合併兩個分支的輸出
                    combined = torch.cat([pinn_out, lstm_out], dim=1)
                    combined = self.merge_layer(combined)
                    combined = self.merge_activation(combined)
                else:
                    combined = pinn_out
                
                # 輸出層
                return self.output_layer(combined)
        
        return HybridModel(self.input_size, self.hidden_sizes, self.lstm_hidden_size, self.output_size)
    
    def _physics_loss(self, X: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        計算物理損失
        
        參數:
            X (torch.Tensor): 輸入特徵
            y_pred (torch.Tensor): 預測的疲勞壽命
            
        返回:
            torch.Tensor: 物理損失
        """
        # 獲取應變相關特徵
        if self.strain_feature_idx is not None:
            strain_feature = X[:, self.strain_feature_idx].reshape(-1, 1)
        else:
            # 默認使用最後一個特徵
            strain_feature = X[:, -1].reshape(-1, 1)
        
        # 使用Coffin-Manson關係計算理論疲勞壽命
        # N_f = C * (Δε_p)^(-n)
        theory_cycles = self.C * torch.pow(strain_feature, -self.n)
        
        # 計算與神經網絡預測的疲勞壽命之間的差異
        physics_loss = self.mse_loss(y_pred, theory_cycles)
        
        return physics_loss
    
    def _detect_feature_types(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """
        檢測特徵類型，將特徵分為靜態特徵和時間序列特徵
        
        參數:
            X (numpy.ndarray): 特徵數據
            feature_names (List[str], optional): 特徵名稱列表
        """
        # 默認所有特徵都是靜態特徵
        self.static_feature_indices = list(range(X.shape[1]))
        self.time_feature_indices = []
        
        # 如果有特徵名稱，嘗試檢測時間序列特徵
        if feature_names is not None:
            # 檢測時間序列特徵 (包含時間點的特徵，如_3600, _7200等)
            time_keywords = ['_3600', '_7200', '_10800', '_14400']
            
            # 收集時間序列特徵的基礎名稱 (不包含時間點)
            time_series_base_names = set()
            for i, name in enumerate(feature_names):
                for keyword in time_keywords:
                    if keyword in name:
                        # 提取基礎名稱 (去掉時間點)
                        base_name = name.split('_up_')[0] if '_up_' in name else name.split('_down_')[0]
                        time_series_base_names.add(base_name)
                        break
            
            # 對每個時間序列基礎名稱，收集其所有時間點的特徵索引
            time_feature_groups = {}
            for base_name in time_series_base_names:
                indices = []
                for i, name in enumerate(feature_names):
                    if name.startswith(base_name + '_up_') or name.startswith(base_name + '_down_'):
                        indices.append(i)
                time_feature_groups[base_name] = indices
            
            # 將所有時間序列特徵索引合併
            all_time_indices = []
            for indices in time_feature_groups.values():
                all_time_indices.extend(indices)
            
            # 去重並排序
            self.time_feature_indices = sorted(list(set(all_time_indices)))
            
            # 靜態特徵是除了時間序列特徵之外的所有特徵
            self.static_feature_indices = [i for i in range(X.shape[1]) if i not in self.time_feature_indices]
            
            # 檢測應變相關特徵
            strain_keywords = ["strain", "equiv", "plastic", "acc"]
            for i, name in enumerate(feature_names):
                if any(keyword in name.lower() for keyword in strain_keywords):
                    self.strain_feature_idx = i
                    break
            
            if self.strain_feature_idx is None and 'Acc_Equi_Strain_max' in feature_names:
                self.strain_feature_idx = feature_names.index('Acc_Equi_Strain_max')
        
        logger.info(f"檢測到 {len(self.static_feature_indices)} 個靜態特徵和 {len(self.time_feature_indices)} 個時間序列特徵")
        if self.strain_feature_idx is not None:
            logger.info(f"檢測到應變相關特徵索引: {self.strain_feature_idx}")
    
    def _prepare_time_series_data(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        準備時間序列數據
        
        參數:
            X (numpy.ndarray): 特徵數據
            
        返回:
            numpy.ndarray or None: 重組的時間序列數據，如果沒有時間序列特徵則返回None
        """
        if not self.time_feature_indices:
            return None
        
        # 從原始數據中提取時間序列特徵
        time_features = X[:, self.time_feature_indices]
        
        # 對於小樣本資料集，我們簡化處理：
        # 將時間序列特徵按順序排列，形成一個序列，每個樣本一個序列
        # 這種處理方式適用於樣本數量較少的情況
        time_series_data = time_features.reshape(X.shape[0], -1, 1)
        
        return time_series_data
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None,
           validation_split: float = 0.2) -> 'HybridPINNLSTM':
        """
        訓練混合PINN-LSTM模型
        
        參數:
            X (numpy.ndarray): 特徵數據
            y (numpy.ndarray): 目標變數
            feature_names (List[str], optional): 特徵名稱列表，用於識別時間序列特徵
            validation_split (float): 驗證集比例
            
        返回:
            HybridPINNLSTM: 訓練後的模型實例
        """
        # 檢測特徵類型
        self._detect_feature_types(X, feature_names)
        
        # 輸入數據檢查和預處理
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # 準備時間序列數據
        X_time = self._prepare_time_series_data(X)
        
        # 提取靜態特徵
        X_static = X[:, self.static_feature_indices]
        
        # 調整模型輸入大小
        if X_static.shape[1] != self.input_size:
            logger.warning(f"靜態特徵數量 {X_static.shape[1]} 與模型輸入大小 {self.input_size} 不符，調整模型結構")
            self.input_size = X_static.shape[1]
            self.model = self._build_model()
            self.optimizer = optim.Adam(
                list(self.model.parameters()) + [self.C, self.n], 
                lr=self.learning_rate, 
                weight_decay=self.l2_reg
            )
        
        # 確保y是二維數組
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # 針對小樣本資料集調整參數
        if X.shape[0] < 50:
            batch_size = min(8, X.shape[0])
            early_stopping_patience = 30
        else:
            batch_size = self.batch_size
            early_stopping_patience = self.early_stopping_patience
        
        # 劃分訓練集和驗證集
        if X.shape[0] < 10:
            X_static_train, y_train = X_static, y
            X_static_val, y_val = X_static, y
            X_time_train = X_time
            X_time_val = X_time
            logger.warning("樣本數量過少，使用全部數據訓練，並以訓練集作為驗證集")
        else:
            indices = np.random.permutation(X.shape[0])
            val_size = int(X.shape[0] * validation_split)
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            X_static_train, y_train = X_static[train_indices], y[train_indices]
            X_static_val, y_val = X_static[val_indices], y[val_indices]
            
            if X_time is not None:
                X_time_train = X_time[train_indices]
                X_time_val = X_time[val_indices]
            else:
                X_time_train = None
                X_time_val = None
        
        # 轉換為PyTorch張量
        X_static_train_tensor = torch.tensor(X_static_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_static_val_tensor = torch.tensor(X_static_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        
        if X_time_train is not None:
            X_time_train_tensor = torch.tensor(X_time_train, dtype=torch.float32)
            X_time_val_tensor = torch.tensor(X_time_val, dtype=torch.float32)
        else:
            X_time_train_tensor = None
            X_time_val_tensor = None
        
        # 創建數據加載器
        if X_time_train_tensor is not None:
            train_dataset = TensorDataset(X_static_train_tensor, X_time_train_tensor, y_train_tensor)
        else:
            train_dataset = TensorDataset(X_static_train_tensor, y_train_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 訓練模型
        best_val_loss = float('inf')
        patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.physics_losses = []
        
        for epoch in range(self.max_epochs):
            # 訓練階段
            self.model.train()
            train_loss = 0.0
            physics_loss_sum = 0.0
            
            for batch_data in train_loader:
                self.optimizer.zero_grad()
                
                # 處理批次數據
                if X_time_train_tensor is not None:
                    X_static_batch, X_time_batch, y_batch = batch_data
                    outputs = self.model(X_static_batch, X_time_batch)
                else:
                    X_static_batch, y_batch = batch_data
                    outputs = self.model(X_static_batch)
                
                # 計算數據損失
                data_loss = self.mse_loss(outputs, y_batch)
                
                # 計算物理損失
                physics_loss = self._physics_loss(X_static_batch, outputs)
                physics_loss_sum += physics_loss.item()
                
                # 總損失 = 數據損失 + 物理損失權重 * 物理損失
                total_loss = data_loss + self.physics_weight * physics_loss
                
                # 反向傳播
                total_loss.backward()
                self.optimizer.step()
                
                train_loss += total_loss.item()
            
            train_loss /= len(train_loader)
            physics_loss_avg = physics_loss_sum / len(train_loader)
            
            self.train_losses.append(train_loss)
            self.physics_losses.append(physics_loss_avg)
            
            # 驗證階段
            self.model.eval()
            with torch.no_grad():
                if X_time_val_tensor is not None:
                    val_outputs = self.model(X_static_val_tensor, X_time_val_tensor)
                else:
                    val_outputs = self.model(X_static_val_tensor)
                
                val_data_loss = self.mse_loss(val_outputs, y_val_tensor).item()
                val_physics_loss = self._physics_loss(X_static_val_tensor, val_outputs).item()
                val_loss = val_data_loss + self.physics_weight * val_physics_loss
                self.val_losses.append(val_loss)
            
            # 早停檢查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                best_C = self.C.detach().clone()
                best_n = self.n.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"早停觸發於第 {epoch+1} 輪，最佳驗證損失: {best_val_loss:.6f}")
                    break
            
            # 每20輪輸出一次訓練進度
            if (epoch + 1) % 20 == 0:
                logger.info(f"輪次 {epoch+1}/{self.max_epochs}, 訓練損失: {train_loss:.6f}, "
                          f"物理損失: {physics_loss_avg:.6f}, 驗證損失: {val_loss:.6f}, "
                          f"C={self.C.item():.4f}, n={self.n.item():.4f}")
        
        # 恢復最佳模型
        if 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
            self.C = nn.Parameter(best_C)
            self.n = nn.Parameter(best_n)
        
        # 設置為已訓練狀態
        self.is_fitted = True
        
        logger.info(f"混合PINN-LSTM模型訓練完成，C={self.C.item():.4f}, n={self.n.item():.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用訓練好的模型進行預測
        
        參數:
            X (numpy.ndarray): 特徵數據
            
        返回:
            numpy.ndarray: 預測結果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用fit方法")
        
        # 輸入數據檢查和預處理
        X = X.astype(np.float32)
        
        # 提取靜態特徵
        X_static = X[:, self.static_feature_indices]
        
        # 確保輸入特徵數量與模型匹配
        if X_static.shape[1] != self.input_size:
            raise ValueError(f"靜態特徵數量 {X_static.shape[1]} 與模型輸入大小 {self.input_size} 不符")
        
        # 準備時間序列數據
        X_time = self._prepare_time_series_data(X)
        
        # 轉換為PyTorch張量
        X_static_tensor = torch.tensor(X_static, dtype=torch.float32)
        if X_time is not None:
            X_time_tensor = torch.tensor(X_time, dtype=torch.float32)
        else:
            X_time_tensor = None
        
        # 預測
        self.model.eval()
        with torch.no_grad():
            if X_time_tensor is not None:
                predictions = self.model(X_static_tensor, X_time_tensor).numpy()
            else:
                predictions = self.model(X_static_tensor).numpy()
        
        return predictions
    
    def get_physics_params(self) -> Dict[str, float]:
        """
        獲取學習到的物理參數
        
        返回:
            Dict[str, float]: 包含物理參數的字典
        """
        return {
            'C': self.C.item(),
            'n': self.n.item()
        }


class PINNLSTMTrainer:
    """
    PINN-LSTM訓練器，用於訓練、評估和管理混合PINN-LSTM模型
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32],
                lstm_hidden_size: int = 32, random_state: int = 42):
        """
        初始化PINN-LSTM訓練器
        
        參數:
            input_size (int): 輸入特徵數量
            hidden_sizes (List[int]): PINN分支隱藏層神經元數量列表
            lstm_hidden_size (int): LSTM隱藏狀態大小
            random_state (int): 隨機種子
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.scaler_X = None
        self.scaler_y = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2,
             learning_rate: float = 0.001, max_epochs: int = 1000,
             batch_size: int = 16, early_stopping_patience: int = 50,
             l2_reg: float = 0.0001, physics_weight: float = 0.3) -> Dict[str, List[float]]:
        """
        訓練混合PINN-LSTM模型
        
        參數:
            X (pandas.DataFrame): 特徵數據框
            y (pandas.Series): 目標變數
            validation_split (float): 驗證集比例
            learning_rate (float): 學習率
            max_epochs (int): 最大訓練周期數
            batch_size (int): 批次大小
            early_stopping_patience (int): 早停耐心值
            l2_reg (float): L2正則化係數
            physics_weight (float): 物理損失權重
            
        返回:
            Dict[str, List[float]]: 訓練過程中的損失記錄
        """
        # 保存特徵名稱
        self.feature_names = X.columns.tolist()
        
        # 創建模型
        self.model = HybridPINNLSTM(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=1,
            lstm_hidden_size=self.lstm_hidden_size,
            random_state=self.random_state,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            l2_reg=l2_reg,
            physics_weight=physics_weight
        )
        
        # 轉換為numpy數組
        X_np = X.values
        y_np = y.values
        if len(y_np.shape) == 1:
            y_np = y_np.reshape(-1, 1)
        
        # 訓練模型
        self.model.fit(X_np, y_np, feature_names=self.feature_names, validation_split=validation_split)
        
        # 返回訓練過程中的損失記錄
        return {
            'train_loss': self.model.train_losses,
            'val_loss': self.model.val_losses,
            'physics_loss': self.model.physics_losses
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用訓練好的模型進行預測
        
        參數:
            X (pandas.DataFrame): 特徵數據框
            
        返回:
            numpy.ndarray: 預測結果
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先調用train方法")
        
        # 檢查特徵名稱
        if not all(feature in X.columns for feature in self.feature_names):
            missing_features = [feature for feature in self.feature_names if feature not in X.columns]
            raise ValueError(f"輸入數據缺少以下特徵: {missing_features}")
        
        # 確保特徵順序一致
        X = X[self.feature_names]
        
        # 預測
        return self.model.predict(X.values)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        評估模型性能
        
        參數:
            X (pandas.DataFrame): 特徵數據框
            y (pandas.Series): 目標變數
            
        返回:
            Dict[str, float]: 評估指標
        """
        # 預測
        y_pred = self.predict(X)
        
        # 轉換為numpy數組
        y_np = y.values
        if len(y_np.shape) == 1:
            y_np = y_np.reshape(-1, 1)
        
        # 計算評估指標
        mse = mean_squared_error(y_np, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_np, y_pred)
        r2 = r2_score(y_np, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def get_physics_params(self) -> Dict[str, float]:
        """
        獲取學習到的物理參數
        
        返回:
            Dict[str, float]: 包含物理參數的字典
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先調用train方法")
        
        return self.model.get_physics_params()
    
    def save(self, filepath: str) -> None:
        """
        保存模型到檔案
        
        參數:
            filepath (str): 檔案路徑
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先調用train方法")
        
        # 確保目錄存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # 將模型和相關資訊保存到檔案
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'input_size': self.input_size,
                'hidden_sizes': self.hidden_sizes,
                'lstm_hidden_size': self.lstm_hidden_size,
                'random_state': self.random_state,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y
            }, f)
            
        logger.info(f"PINN-LSTM模型已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PINNLSTMTrainer':
        """
        從檔案載入模型
        
        參數:
            filepath (str): 檔案路徑
            
        返回:
            PINNLSTMTrainer: 載入的訓練器實例
        """
        # 檢查檔案是否存在
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"檔案 {filepath} 不存在")
        
        # 從檔案載入模型和相關資訊
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # 創建新的訓練器實例
        trainer = cls(
            input_size=data['input_size'],
            hidden_sizes=data['hidden_sizes'],
            lstm_hidden_size=data['lstm_hidden_size'],
            random_state=data['random_state']
        )
        
        trainer.model = data['model']
        trainer.feature_names = data['feature_names']
        trainer.scaler_X = data['scaler_X']
        trainer.scaler_y = data['scaler_y']
        
        logger.info(f"從 {filepath} 載入PINN-LSTM模型")
        
        return trainer


# 用法示例
if __name__ == "__main__":
    try:
        import os
        import pandas as pd
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
            
            # 分割訓練集和測試集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"訓練集: {X_train.shape}, 測試集: {X_test.shape}")
            
            # 初始化預測器
            predictor = FatigueLifePredictor(random_state=42)
            
            # 訓練多個模型
            metrics = predictor.train_models(X_train, y_train, cv=5, tune_hyperparams=True)
            
            # 輸出各模型在訓練集上的性能
            print("\n各模型在訓練集上的性能:")
            for name, metric in metrics.items():
                print(f"{name}: RMSE = {metric['rmse']:.4f}±{metric['rmse_std']:.4f}, "
                      f"R² = {metric['r2']:.4f}±{metric['r2_std']:.4f}")
            
            # 評估最佳模型在測試集上的性能
            best_metrics = predictor.evaluate_model(X_test, y_test)
            print(f"\n最佳模型 ({predictor.best_model_name}) 在測試集上的性能:")
            print(f"RMSE = {best_metrics['rmse']:.4f}, MAE = {best_metrics['mae']:.4f}, R² = {best_metrics['r2']:.4f}")
            
            # 如果最佳模型支援特徵重要性，則輸出重要特徵
            try:
                feature_importance = predictor.get_feature_importance()
                print("\n重要特徵 (前10個):")
                for i, (feature, importance) in enumerate(zip(feature_importance['feature'][:10], 
                                                            feature_importance['importance'][:10])):
                    print(f"{i+1}. {feature}: {importance:.4f}")
            except Exception as e:
                print(f"獲取特徵重要性時出錯: {str(e)}")
            
            # 嘗試訓練混合PINN-LSTM模型
            if TORCH_AVAILABLE:
                try:
                    print("\n開始訓練混合PINN-LSTM模型...")
                    pinn_lstm_trainer = PINNLSTMTrainer(
                        input_size=X_train.shape[1],
                        random_state=42
                    )
                    
                    # 訓練模型
                    loss_history = pinn_lstm_trainer.train(
                        X_train, y_train,
                        validation_split=0.2,
                        max_epochs=500,  # 簡化為500輪
                        batch_size=8,
                        physics_weight=0.3
                    )
                    
                    # 評估模型
                    pinn_metrics = pinn_lstm_trainer.evaluate(X_test, y_test)
                    print("\n混合PINN-LSTM模型在測試集上的性能:")
                    print(f"RMSE = {pinn_metrics['rmse']:.4f}, MAE = {pinn_metrics['mae']:.4f}, R² = {pinn_metrics['r2']:.4f}")
                    
                    # 獲取物理參數
                    physics_params = pinn_lstm_trainer.get_physics_params()
                    print(f"\n學習到的物理參數: C = {physics_params['C']:.4f}, n = {physics_params['n']:.4f}")
                    
                    # 保存模型
                    model_dir = os.path.join('models')
                    os.makedirs(model_dir, exist_ok=True)
                    pinn_lstm_trainer.save(os.path.join(model_dir, 'pinn_lstm_model.pkl'))
                    
                except Exception as e:
                    print(f"訓練混合PINN-LSTM模型時出錯: {str(e)}")
            
            # 保存最佳模型
            model_dir = os.path.join('models')
            os.makedirs(model_dir, exist_ok=True)
            predictor.save(os.path.join(model_dir, 'best_model.pkl'))
            print(f"\n最佳模型已保存到: {os.path.join(model_dir, 'best_model.pkl')}")
            
        else:
            print(f"示例數據文件 {data_path} 不存在，請提供正確的數據文件路徑。")
            print("您可以按照以下方式使用本模組:")
            print("""
# 初始化預測器
predictor = FatigueLifePredictor(random_state=42)

# 訓練多個模型
metrics = predictor.train_models(X_train, y_train, cv=5)

# 使用最佳模型進行預測
predictions = predictor.predict(X_test)

# 評估模型性能
eval_metrics = predictor.evaluate_model(X_test, y_test)

# 保存模型
predictor.save('models/best_model.pkl')

# 載入已訓練的模型
loaded_predictor = FatigueLifePredictor.load('models/best_model.pkl')
            """)
    
    except Exception as e:
        print(f"運行示例時出錯: {str(e)}")
        print("請確保所需的依賴包已正確安裝，並提供有效的數據文件。")