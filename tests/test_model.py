#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
test_model.py - 銲錫接點疲勞壽命預測系統模型測試
本模組包含對src/model.py中FatigueLifePredictor類和相關模型的單元測試，
測試內容包括模型初始化、訓練、預測、評估以及模型保存與加載功能。
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# 將父目錄加入sys.path以便正確導入src模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入被測試的模組
from src.model import FatigueLifePredictor, MLPRegressor, PhysicsInformedNN, HybridPINNLSTM

# 檢查PyTorch是否可用
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestFatigueLifePredictor(unittest.TestCase):
    """測試FatigueLifePredictor類"""
    
    def setUp(self):
        """設置測試環境"""
        # 創建合成數據用於測試
        n_samples = 100
        n_features = 10
        X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                              noise=0.1, random_state=42)
        
        # 分割訓練集和測試集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 創建特徵名稱
        self.feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # 轉換為DataFrame和Series
        self.X_train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        self.X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        self.y_train_series = pd.Series(self.y_train)
        self.y_test_series = pd.Series(self.y_test)
        
        # 初始化預測器
        self.predictor = FatigueLifePredictor(random_state=42)
        
        # 創建測試模型保存路徑
        self.test_model_path = os.path.join('tests', 'temp_model.pkl')
    
    def tearDown(self):
        """清理測試環境"""
        # 刪除測試過程中創建的文件
        if os.path.exists(self.test_model_path):
            os.remove(self.test_model_path)
    
    def test_init(self):
        """測試初始化"""
        self.assertEqual(self.predictor.random_state, 42)
        self.assertEqual(len(self.predictor.models), 0)
        self.assertIsNone(self.predictor.best_model)
        self.assertIsNone(self.predictor.best_model_name)
    
    def test_create_models(self):
        """測試創建模型"""
        models = self.predictor._create_models()
        
        # 檢查是否創建了所有預期的模型
        expected_models = ['linear', 'ridge', 'lasso', 'elastic_net', 
                          'random_forest', 'gradient_boosting', 'svr', 
                          'knn', 'gpr']
                          
        for model_name in expected_models:
            self.assertIn(model_name, models)
        
        # 檢查是否根據PyTorch可用性添加了神經網絡模型
        if TORCH_AVAILABLE:
            self.assertIn('neural_network', models)
            self.assertIn('pinn', models)
    
    def test_train_models(self):
        """測試訓練模型"""
        # 使用部分模型進行快速測試
        test_models = {
            'linear': self.predictor._create_models()['linear'],
            'random_forest': self.predictor._create_models()['random_forest']
        }
        
        # 僅保留測試模型
        original_create_models = self.predictor._create_models
        self.predictor._create_models = lambda: test_models
        
        # 訓練模型
        metrics = self.predictor.train_models(self.X_train_df, self.y_train_series, 
                                           cv=2, tune_hyperparams=False)
        
        # 恢復原始方法
        self.predictor._create_models = original_create_models
        
        # 檢查是否訓練了所有測試模型
        for model_name in test_models.keys():
            self.assertIn(model_name, self.predictor.models)
            self.assertIn(model_name, metrics)
        
        # 檢查是否選擇了最佳模型
        self.assertIsNotNone(self.predictor.best_model_name)
        self.assertIsNotNone(self.predictor.best_model)
        
        # 檢查評估指標是否包含預期的鍵
        for model_metrics in metrics.values():
            self.assertIn('rmse', model_metrics)
            self.assertIn('r2', model_metrics)
    
    def test_predict(self):
        """測試預測功能"""
        # 先訓練模型
        test_models = {
            'linear': self.predictor._create_models()['linear']
        }
        
        # 僅保留測試模型
        original_create_models = self.predictor._create_models
        self.predictor._create_models = lambda: test_models
        
        # 訓練模型
        self.predictor.train_models(self.X_train_df, self.y_train_series, 
                                   cv=2, tune_hyperparams=False)
        
        # 恢復原始方法
        self.predictor._create_models = original_create_models
        
        # 使用DataFrame進行預測
        y_pred_df = self.predictor.predict(self.X_test_df)
        self.assertEqual(len(y_pred_df), len(self.X_test_df))
        
        # 使用numpy數組進行預測
        y_pred_np = self.predictor.predict(self.X_test_df.values)
        self.assertEqual(len(y_pred_np), len(self.X_test_df))
        
        # 檢查預測結果是否相同
        np.testing.assert_array_almost_equal(y_pred_df, y_pred_np)
        
        # 測試指定模型名稱進行預測
        y_pred_named = self.predictor.predict(self.X_test_df, model_name='linear')
        np.testing.assert_array_almost_equal(y_pred_df, y_pred_named)
    
    def test_evaluate_model(self):
        """測試模型評估"""
        # 先訓練模型
        test_models = {
            'linear': self.predictor._create_models()['linear']
        }
        
        # 僅保留測試模型
        original_create_models = self.predictor._create_models
        self.predictor._create_models = lambda: test_models
        
        # 訓練模型
        self.predictor.train_models(self.X_train_df, self.y_train_series, 
                                   cv=2, tune_hyperparams=False)
        
        # 恢復原始方法
        self.predictor._create_models = original_create_models
        
        # 評估模型
        metrics = self.predictor.evaluate_model(self.X_test_df, self.y_test_series)
        
        # 檢查評估指標
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        
        # 檢查指標值是否合理
        self.assertGreaterEqual(metrics['r2'], -1.0)  # R2可以為負值，但應大於-1
        self.assertLessEqual(metrics['r2'], 1.0)      # R2最大為1
        self.assertGreaterEqual(metrics['rmse'], 0.0) # RMSE應非負
        self.assertGreaterEqual(metrics['mae'], 0.0)  # MAE應非負
    
    def test_get_feature_importance(self):
        """測試獲取特徵重要性"""
        # 先訓練支持特徵重要性的模型
        test_models = {
            'random_forest': self.predictor._create_models()['random_forest']
        }
        
        # 僅保留測試模型
        original_create_models = self.predictor._create_models
        self.predictor._create_models = lambda: test_models
        
        # 訓練模型
        self.predictor.train_models(self.X_train_df, self.y_train_series, 
                                   cv=2, tune_hyperparams=False)
        
        # 恢復原始方法
        self.predictor._create_models = original_create_models
        
        # 獲取特徵重要性
        feature_importance = self.predictor.get_feature_importance()
        
        # 檢查特徵重要性數據框
        self.assertIsInstance(feature_importance, pd.DataFrame)
        self.assertEqual(len(feature_importance), len(self.feature_names))
        self.assertIn('feature', feature_importance.columns)
        self.assertIn('importance', feature_importance.columns)
        
        # 檢查重要性值的總和是否接近1.0
        self.assertAlmostEqual(feature_importance['importance'].sum(), 1.0, delta=0.01)
    
    def test_save_load(self):
        """測試模型保存和加載"""
        # 先訓練模型
        test_models = {
            'linear': self.predictor._create_models()['linear'],
            'random_forest': self.predictor._create_models()['random_forest']
        }
        
        # 僅保留測試模型
        original_create_models = self.predictor._create_models
        self.predictor._create_models = lambda: test_models
        
        # 訓練模型
        self.predictor.train_models(self.X_train_df, self.y_train_series, 
                                   cv=2, tune_hyperparams=False)
        
        # 恢復原始方法
        self.predictor._create_models = original_create_models
        
        # 保存模型
        self.predictor.save(self.test_model_path)
        
        # 檢查文件是否存在
        self.assertTrue(os.path.exists(self.test_model_path))
        
        # 加載模型
        loaded_predictor = FatigueLifePredictor.load(self.test_model_path)
        
        # 檢查加載的模型
        self.assertEqual(loaded_predictor.best_model_name, self.predictor.best_model_name)
        self.assertEqual(loaded_predictor.feature_names, self.predictor.feature_names)
        
        # 使用加載的模型進行預測
        y_pred_original = self.predictor.predict(self.X_test_df)
        y_pred_loaded = loaded_predictor.predict(self.X_test_df)
        
        # 檢查預測結果是否相同
        np.testing.assert_array_almost_equal(y_pred_original, y_pred_loaded)


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch不可用，跳過神經網絡模型測試")
class TestNeuralNetworkModels(unittest.TestCase):
    """測試神經網絡相關模型（只有在PyTorch可用時才運行）"""
    
    def setUp(self):
        """設置測試環境"""
        # 創建小型合成數據用於測試
        n_samples = 20
        n_features = 5
        X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                              noise=0.1, random_state=42)
        
        # 確保y是二維數組
        y = y.reshape(-1, 1)
        
        # 轉換為float32類型
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    
    def test_mlp_regressor(self):
        """測試多層感知器回歸器"""
        # 初始化模型
        mlp = MLPRegressor(
            input_size=self.X.shape[1],
            hidden_sizes=[10, 5],
            output_size=1,
            max_epochs=10  # 減少周期數以加快測試
        )
        
        # 訓練模型
        mlp.fit(self.X, self.y, validation_split=0.2)
        
        # 檢查是否已訓練
        self.assertTrue(mlp.is_fitted)
        
        # 進行預測
        y_pred = mlp.predict(self.X)
        
        # 檢查預測結果形狀
        self.assertEqual(y_pred.shape, self.y.shape)
    
    def test_physics_informed_nn(self):
        """測試物理信息神經網絡"""
        # 初始化模型
        pinn = PhysicsInformedNN(
            input_size=self.X.shape[1],
            hidden_sizes=[10, 5],
            output_size=1,
            max_epochs=10  # 減少周期數以加快測試
        )
        
        # 設置應變特徵索引
        strain_feature_idx = 0
        
        # 訓練模型
        pinn.fit(self.X, self.y, strain_feature_idx=strain_feature_idx, validation_split=0.2)
        
        # 檢查是否已訓練
        self.assertTrue(pinn.is_fitted)
        
        # 進行預測
        y_pred = pinn.predict(self.X)
        
        # 檢查預測結果形狀
        self.assertEqual(y_pred.shape, self.y.shape)
        
        # 獲取物理參數
        physics_params = pinn.get_physics_params()
        
        # 檢查物理參數
        self.assertIn('C', physics_params)
        self.assertIn('n', physics_params)
    
    def test_hybrid_pinn_lstm(self):
        """測試混合PINN-LSTM模型"""
        # 初始化模型
        hybrid = HybridPINNLSTM(
            input_size=self.X.shape[1],
            hidden_sizes=[10, 5],
            output_size=1,
            lstm_hidden_size=8,
            max_epochs=10  # 減少周期數以加快測試
        )
        
        # 設置特徵類型和應變特徵索引
        hybrid.static_feature_indices = list(range(self.X.shape[1]))
        hybrid.time_feature_indices = []
        hybrid.strain_feature_idx = 0
        
        # 準備時間序列數據
        X_time = hybrid._prepare_time_series_data(self.X)
        
        # 這裡X_time應該為None，因為我們將所有特徵都設為靜態特徵
        self.assertIsNone(X_time)
        
        # 訓練模型
        hybrid.fit(self.X, self.y, validation_split=0.2)
        
        # 檢查是否已訓練
        self.assertTrue(hybrid.is_fitted)
        
        # 進行預測
        y_pred = hybrid.predict(self.X)
        
        # 檢查預測結果形狀
        self.assertEqual(y_pred.shape, self.y.shape)
        
        # 獲取物理參數
        physics_params = hybrid.get_physics_params()
        
        # 檢查物理參數
        self.assertIn('C', physics_params)
        self.assertIn('n', physics_params)


if __name__ == '__main__':
    unittest.main()