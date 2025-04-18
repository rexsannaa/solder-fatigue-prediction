#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
predict.py - 銲錫接點疲勞壽命預測腳本
用於命令行進行批量預測的腳本，支持各種參數設置

使用方法：
python predict.py --data_path DATA_PATH --model_path MODEL_PATH --output_path OUTPUT_PATH
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
from src.utils import ensure_dir, setup_logger

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='銲錫接點疲勞壽命預測腳本')
    
    # 必要參數
    parser.add_argument('--data_path', type=str, required=True,
                      help='待預測數據文件路徑 (CSV 或 Excel)')
    parser.add_argument('--model_path', type=str, required=True,
                      help='訓練好的模型路徑 (.pkl)')
    
    # 可選參數
    parser.add_argument('--output_path', type=str, default='results/predictions.csv',
                      help='預測結果輸出路徑 (默認: "results/predictions.csv")')
    parser.add_argument('--log_file', type=str, default='logs/prediction.log',
                      help='日誌文件路徑 (默認: "logs/prediction.log")')
    parser.add_argument('--has_target', action='store_true',
                      help='數據是否包含目標變數 (默認: False)')
    parser.add_argument('--target_column', type=str, default='Nf_pred (cycles)',
                      help='目標變數列名，僅當 has_target=True 時有效 (默認: "Nf_pred (cycles)")')
    
    return parser.parse_args()

def engineer_features(data, target_column=None):
    """
    為預測數據進行特徵工程處理
    
    參數:
        data (pandas.DataFrame): 原始數據
        target_column (str, optional): 目標變數列名，如果有的話
        
    返回:
        pandas.DataFrame: 特徵工程處理後的數據
    """
    # 從原始特徵創建所需的派生特徵
    df = data.copy()
    
    # 如果有目標列，暫時移除它
    target_data = None
    if target_column and target_column in df.columns:
        target_data = df[target_column].copy()
        df = df.drop(target_column, axis=1)
    
    # 1. 創建幾何比例特徵
    if all(col in df.columns for col in ['Die', 'PCB']):
        df['die_pcb_ratio'] = df['Die'] / df['PCB']
    
    if all(col in df.columns for col in ['stud', 'mold']):
        df['stud_mold_ratio'] = df['stud'] / df['mold']
    
    # 2. 添加其他必要的比例特徵
    if all(col in df.columns for col in ['Die', 'mold']):
        df['die_mold_ratio'] = df['Die'] / df['mold']
    
    # 3. 添加翹曲相關特徵
    if all(col in df.columns for col in ['Total_warpage', 'Unit_warpage']):
        df['warpage_ratio'] = df['Total_warpage'] / df['Unit_warpage']
    
    # 4. 添加應力應變相關特徵
    stress_cols = [col for col in df.columns if 'NLPLWK' in col]
    strain_cols = [col for col in df.columns if 'Strain' in col]
    
    if stress_cols and strain_cols:
        # 添加應力應變比
        for stress_col in stress_cols[:1]:  # 只使用第一個應力列
            for strain_col in strain_cols[:1]:  # 只使用第一個應變列
                df['stress_strain_ratio'] = df[stress_col] / df[strain_col].replace(0, np.nan)
    
    # 5. 添加應力平均值特徵
    if stress_cols:
        df['avg_stress'] = df[stress_cols].mean(axis=1)
    
    # 6. 添加應變相關特徵
    if 'Acc_Equi_Strain_max' in df.columns and 'Die' in df.columns:
        df['damage_factor'] = df['Acc_Equi_Strain_max'] * df['Die']
    
    # 如果有目標列，添加回去
    if target_data is not None:
        df[target_column] = target_data
    
    return df

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
    
    # 設置日誌
    logger = setup_logger(log_file=args.log_file)
    logger.info(f"開始預測，參數: {args}")
    
    try:
        # 檢查模型文件是否存在
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"找不到模型文件: {args.model_path}")
        
        # 載入模型
        logger.info(f"正在載入模型: {args.model_path}")
        predictor = FatigueLifePredictor.load(args.model_path)
        logger.info(f"模型載入成功，最佳模型: {predictor.best_model_name}")
        
        # 載入數據
        logger.info(f"正在從 {args.data_path} 載入數據...")
        processor = DataProcessor()
        data = processor.load_data(args.data_path)
        logger.info(f"數據載入成功，形狀: {data.shape}")
        
        # 進行特徵工程處理
        logger.info("進行特徵工程處理...")
        try:
            # 使用自定義函數進行特徵工程
            data = engineer_features(data, args.target_column if args.has_target else None)
            logger.info(f"特徵工程完成，處理後特徵數量: {data.shape[1]}")
            
            # 如果上述方法不夠，可以嘗試使用FeatureEngineer
            if not all(feature in data.columns for feature in predictor.feature_names):
                logger.info("基本特徵工程不足，嘗試使用FeatureEngineer...")
                engineer = FeatureEngineer()
                
                # 如果數據中包含目標變數，暫時保留它
                if args.has_target and args.target_column in data.columns:
                    target_data = data[args.target_column].copy()
                    data_no_target = data.drop(args.target_column, axis=1)
                    
                    # 應用特徵工程
                    engineered_data = engineer.create_material_features(data_no_target)
                    
                    # 恢復目標變數
                    engineered_data[args.target_column] = target_data
                else:
                    # 如果沒有目標變數，直接進行特徵工程
                    engineered_data = engineer.create_material_features(data)
                
                data = engineered_data
                logger.info(f"FeatureEngineer處理完成，處理後特徵數量: {data.shape[1]}")
        except Exception as e:
            logger.warning(f"進行特徵工程時遇到問題: {str(e)}")
            logger.warning("將嘗試使用原始數據進行預測")
        
        # 檢查特徵是否完整
        required_features = predictor.feature_names
        if required_features is not None:
            missing_features = [f for f in required_features if f not in data.columns]
            if missing_features:
                raise ValueError(f"數據缺少以下必要特徵: {', '.join(missing_features)}")
            
            # 僅保留必要的特徵，並按照模型訓練時的順序
            prediction_data = data[required_features]
            logger.info(f"保留 {len(required_features)} 個特徵用於預測")
        else:
            # 如果模型沒有指定特徵名稱，使用全部特徵
            prediction_data = data.copy()
            # 如果有目標變數，則排除
            if args.has_target and args.target_column in prediction_data.columns:
                prediction_data = prediction_data.drop(args.target_column, axis=1)
            logger.info(f"使用全部 {prediction_data.shape[1]} 個特徵進行預測")
        
        # 進行預測
        logger.info("開始進行預測...")
        start_time = datetime.now()
        predictions = predictor.predict(prediction_data)
        prediction_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"預測完成，耗時: {prediction_time:.2f} 秒，預測了 {len(predictions)} 個樣本")
        
        # 準備結果數據框
        result_data = data.copy()
        result_data['預測疲勞壽命'] = predictions
        
        # 如果有目標變數，計算評估指標
        if args.has_target and args.target_column in data.columns:
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            y_true = data[args.target_column]
            y_pred = predictions
            
            # 計算評估指標
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            logger.info(f"評估指標: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
            
            # 添加誤差列
            result_data['實際值'] = y_true
            result_data['殘差'] = y_true - predictions
            result_data['百分比誤差(%)'] = 100 * (y_true - predictions) / y_true
        
        # 保存結果
        result_data.to_csv(args.output_path, index=False)
        logger.info(f"預測結果已保存至: {args.output_path}")
        
        # 輸出樣本結果
        logger.info("預測結果樣本 (前5行):")
        for i, row in result_data.head(5).iterrows():
            if args.has_target and args.target_column in data.columns:
                logger.info(f"樣本 {i+1}: 預測值 = {row['預測疲勞壽命']:.2f}, 實際值 = {row['實際值']:.2f}")
            else:
                logger.info(f"樣本 {i+1}: 預測值 = {row['預測疲勞壽命']:.2f}")
        
        logger.info("預測流程完成!")
        
    except Exception as e:
        logger.error(f"預測過程中出錯: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()