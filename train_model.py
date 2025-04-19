#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
train_model.py - 銲錫接點疲勞壽命預測模型訓練腳本
用於命令行訓練模型的腳本，支持各種參數設置

使用方法：
python train_model.py --data_path DATA_PATH --model_type MODEL_TYPE --output_path OUTPUT_PATH
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# 添加src目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入所需模組
from src.model import FatigueLifePredictor
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.evaluation import ModelEvaluator
from src.utils import ensure_dir, setup_logger

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='銲錫接點疲勞壽命預測模型訓練腳本')
    
    # 必要參數
    parser.add_argument('--data_path', type=str, required=True,
                      help='數據文件路徑 (CSV 或 Excel)')
    
    # 可選參數
    parser.add_argument('--target_column', type=str, default='Nf_pred (cycles)',
                      help='目標變數列名 (默認: "Nf_pred (cycles)")')
    parser.add_argument('--model_type', type=str, default='random_forest',
                      choices=['random_forest', 'gradient_boosting', 'svr', 'multiple'],
                      help='模型類型 (默認: "random_forest")')
    parser.add_argument('--output_path', type=str, default='models/trained_model.pkl',
                      help='模型輸出路徑 (默認: "models/trained_model.pkl")')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='測試集比例 (默認: 0.2)')
    parser.add_argument('--tune_hyperparams', action='store_true',
                      help='是否調整超參數 (默認: False)')
    parser.add_argument('--feature_engineering', action='store_true',
                      help='是否啟用特徵工程 (默認: False)')
    parser.add_argument('--log_file', type=str, default='logs/training.log',
                      help='日誌文件路徑 (默認: "logs/training.log")')
    parser.add_argument('--figures_dir', type=str, default='figures',
                      help='圖像輸出目錄 (默認: "figures")')
    parser.add_argument('--cv', type=int, default=5,
                      help='交叉驗證折數 (默認: 5)')
    parser.add_argument('--random_state', type=int, default=42,
                      help='隨機種子 (默認: 42)')
    
    return parser.parse_args()

def main():
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 確保輸出目錄存在
    output_dir = os.path.dirname(args.output_path)
    ensure_dir(output_dir)
    
    # 確保日誌目錄存在
    log_dir = os.path.dirname(args.log_file)
    ensure_dir(log_dir)
    
    # 確保圖像目錄存在
    ensure_dir(args.figures_dir)
    
    # 設置日誌
    logger = setup_logger(log_file=args.log_file)
    logger.info(f"開始訓練模型，參數: {args}")
    
    try:
        # 載入數據
        logger.info(f"正在從 {args.data_path} 載入數據...")
        processor = DataProcessor(random_state=args.random_state)
        data = processor.load_data(args.data_path)
        logger.info(f"數據載入成功，形狀: {data.shape}")
        
        # 數據處理
        if args.feature_engineering:
            logger.info("開始進行數據處理與特徵工程...")
            X_train, X_test, y_train, y_test, feature_names = processor.prepare_data_pipeline(
                data, 
                target_col=args.target_column,
                test_size=args.test_size,
                clean=True,
                feature_engineering=True,
                feature_selection=False,
                normalization='standard'
            )
        else:
            logger.info("僅進行基本數據處理...")
            # 基本數據清理
            clean_data = processor.clean_data(data)
            
            # 分離特徵和目標變數
            if args.target_column not in clean_data.columns:
                raise ValueError(f"目標變數列 '{args.target_column}' 不在數據中")
            
            y = clean_data[args.target_column]
            X = clean_data.drop(args.target_column, axis=1)
            
            # 數據分割
            X_train, X_test, y_train, y_test = processor.split_data(
                X, y, test_size=args.test_size
            )
            feature_names = X.columns.tolist()
        
        logger.info(f"數據處理完成: 訓練集 {X_train.shape}, 測試集 {X_test.shape}")
        
        # 初始化預測器
        predictor = FatigueLifePredictor(random_state=args.random_state)
        predictor.feature_names = feature_names
        
        # 創建適當的模型字典
        logger.info(f"選擇模型類型: {args.model_type}")
        if args.model_type == 'random_forest':
            models = {
                'random_forest': predictor._create_models()['random_forest']
            }
        elif args.model_type == 'gradient_boosting':
            models = {
                'gradient_boosting': predictor._create_models()['gradient_boosting']
            }
        elif args.model_type == 'svr':
            models = {
                'svr': predictor._create_models()['svr']
            }
        elif args.model_type == 'multiple':
            # 訓練多個模型
            models = {
                'random_forest': predictor._create_models()['random_forest'],
                'gradient_boosting': predictor._create_models()['gradient_boosting'],
                'svr': predictor._create_models()['svr'],
                'linear': predictor._create_models()['linear']
            }
        
        # 訓練模型
        logger.info("開始訓練模型...")
        start_time = datetime.now()
        
        # 對每個模型進行訓練和評估
        metrics = {}
        for name, model in models.items():
            logger.info(f"訓練模型: {name}")
            
            if args.tune_hyperparams:
                model = predictor._tune_hyperparameters(name, model, X_train, y_train, cv=args.cv)
            
            # 在整個訓練集上訓練最終模型
            model.fit(X_train, y_train)
            predictor.models[name] = model
            
            # 進行預測並評估
            y_pred = model.predict(X_test)
            
            # 評估指標
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics[name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            logger.info(f"模型 {name} 評估: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        # 找出最佳模型
        best_model_name = min(metrics.items(), key=lambda x: x[1]['rmse'])[0]
        predictor.best_model_name = best_model_name
        predictor.best_model = predictor.models[best_model_name]
        
        # 計算訓練時間
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"模型訓練完成，耗時: {training_time:.2f} 秒")
        logger.info(f"最佳模型: {best_model_name}, RMSE: {metrics[best_model_name]['rmse']:.4f}")
        
        # 保存訓練好的模型
        predictor.save(args.output_path)
        logger.info(f"模型已保存至: {args.output_path}")
        
        # 生成評估圖表
        logger.info("生成評估圖表...")
        try:
            # 初始化評估器
            evaluator = ModelEvaluator(output_dir=args.figures_dir)
            
            # 實際值vs預測值圖
            evaluator.plot_actual_vs_predicted(
                y_test, 
                predictor.predict(X_test),
                model_name=best_model_name,
                save_fig=True,
                show_fig=False
            )
            
            # 殘差分析圖
            #evaluator.plot_residuals(
            #    y_test, 
            #    predictor.predict(X_test),
            #    model_name=best_model_name,
            #    save_fig=True,
            #    show_fig=False
            #)
            
            # 嘗試繪製特徵重要性圖
            if hasattr(predictor.best_model, 'feature_importances_'):
                evaluator.plot_feature_importance(
                    predictor.best_model,
                    feature_names,
                    model_name=best_model_name,
                    save_fig=True,
                    show_fig=False
                )
                logger.info(f"特徵重要性圖已保存至: {args.figures_dir}")
            
            logger.info(f"評估圖表已保存至: {args.figures_dir}")
                
        except Exception as e:
            logger.error(f"生成評估圖表時出錯: {str(e)}")
            
        logger.info("全部流程完成!")
        
    except Exception as e:
        logger.error(f"訓練過程中出錯: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()