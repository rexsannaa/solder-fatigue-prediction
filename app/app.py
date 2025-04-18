#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
app.py - 銲錫接點疲勞壽命預測系統應用程式
本模組提供銲錫接點疲勞壽命預測系統的Web應用介面，用戶可以通過此介面：
1. 上傳數據並進行模型預測
2. 查看預測結果及可視化圖表
3. 比較不同模型的預測性能
4. 進行敏感性分析和特徵重要性評估
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
import logging
import torch.nn as nn


# 添加父目錄到路徑，以便導入其他模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入自己的模組
from src.model import FatigueLifePredictor
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.evaluation import ModelEvaluator
from src.utils import setup_logger, ensure_dir, plot_actual_vs_predicted, plot_feature_importance

# 配置日誌
logger = setup_logger(log_file='logs/app.log')

# 初始化 Flask 應用
app = Flask(__name__)
app.secret_key = 'solder_fatigue_prediction_app_secret_key'

# 配置上傳文件夾和允許的文件類型
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上傳文件大小為16MB

# 確保上傳和結果文件夾存在
ensure_dir(UPLOAD_FOLDER)
ensure_dir(RESULT_FOLDER)

# 加載預訓練的模型
MODEL_PATH = 'models/best_model.pkl'
MODELS_DIR = 'models'

# 全局變量，用於存儲當前會話的數據和模型
session_data = {
    'uploaded_file': None,
    'processed_data': None,
    'predictor': None,
    'feature_names': None,
    'target_column': None,
    'prediction_results': None,
    'available_models': []
}

def allowed_file(filename):
    """檢查文件類型是否允許上傳"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_available_models():
    """加載可用的模型列表"""
    models = []
    if os.path.exists(MODELS_DIR):
        for file in os.listdir(MODELS_DIR):
            if file.endswith('.pkl'):
                model_path = os.path.join(MODELS_DIR, file)
                model_name = file.replace('.pkl', '')
                models.append({
                    'name': model_name,
                    'path': model_path,
                    'size': f"{os.path.getsize(model_path) / (1024 * 1024):.2f} MB",
                    'modified': datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
                })
    return models

@app.route('/')
def index():
    """首頁"""
    # 加載可用的模型
    session_data['available_models'] = load_available_models()
    
    return render_template('index.html', 
                         models=session_data['available_models'],
                         has_data=session_data['processed_data'] is not None,
                         has_prediction=session_data['prediction_results'] is not None)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """處理文件上傳"""
    if request.method == 'POST':
        # 檢查是否有文件
        if 'file' not in request.files:
            flash('沒有選擇文件')
            return redirect(request.url)
        
        file = request.files['file']
        
        # 檢查文件名是否為空
        if file.filename == '':
            flash('沒有選擇文件')
            return redirect(request.url)
        
        # 檢查文件類型是否允許
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # 保存文件路徑到會話
            session_data['uploaded_file'] = file_path
            
            # 嘗試讀取數據
            try:
                processor = DataProcessor()
                data = processor.load_data(file_path)
                
                # 獲取列名顯示給用戶選擇目標變數
                columns = data.columns.tolist()
                
                # 渲染選擇目標變數的頁面
                return render_template('select_target.html', 
                                     columns=columns,
                                     filename=filename)
            
            except Exception as e:
                logger.error(f"處理上傳文件時出錯: {str(e)}")
                flash(f'處理文件時出錯: {str(e)}')
                return redirect(url_for('index'))
        
        else:
            flash(f'不支援的文件類型。允許的類型: {", ".join(ALLOWED_EXTENSIONS)}')
            return redirect(request.url)
    
    # GET 請求顯示上傳表單
    return render_template('upload.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    """處理數據，包括特徵工程和數據預處理"""
    if session_data['uploaded_file'] is None:
        flash('請先上傳文件')
        return redirect(url_for('index'))
    
    try:
        # 獲取選擇的目標變數
        target_column = request.form.get('target_column')
        if not target_column:
            flash('請選擇目標變數')
            return redirect(url_for('upload_file'))
        
        session_data['target_column'] = target_column
        
        # 初始化數據處理器
        processor = DataProcessor()
        
        # 載入數據
        data = processor.load_data(session_data['uploaded_file'])
        
        # 基本數據清理
        clean_data = processor.clean_data(data)
        
        # 分離特徵和目標變數
        if target_column not in clean_data.columns:
            flash(f'選擇的目標變數 {target_column} 不在數據中')
            return redirect(url_for('upload_file'))
        
        y = clean_data[target_column]
        X = clean_data.drop(target_column, axis=1)
        
        # 特徵工程
        engineer = FeatureEngineer()
        X_engineered = engineer.apply_feature_engineering_pipeline(
            X, y,
            material_features=True,
            interaction_features=False,
            time_series_features=True,
            feature_selection=False,
            scaling=True
        )
        
        # 保存處理後的數據和特徵名稱
        session_data['processed_data'] = {
            'X': X_engineered,
            'y': y,
            'original_data': data
        }
        session_data['feature_names'] = X_engineered.columns.tolist()
        
        # 顯示數據概況
        data_info = {
            'rows': len(data),
            'columns': len(data.columns),
            'target_column': target_column,
            'features_count': len(X_engineered.columns),
            'missing_values': data.isnull().sum().sum(),
            'sample_data': X_engineered.head(5).to_html(classes='table table-striped table-bordered')
        }
        
        # 生成一些基本的數據分析圖
        try:
            # 保存目標變數分佈圖
            plt.figure(figsize=(10, 6))
            sns.histplot(y, kde=True)
            plt.title(f'{target_column} 分佈')
            plt.xlabel(target_column)
            plt.ylabel('頻率')
            plt.grid(True, linestyle='--', alpha=0.7)
            target_hist_path = os.path.join(app.config['RESULT_FOLDER'], 'target_distribution.png')
            plt.savefig(target_hist_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 生成相關性熱圖
            plt.figure(figsize=(12, 10))
            corr_matrix = X_engineered.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, 
                       square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
            plt.title('特徵相關性熱圖')
            plt.tight_layout()
            corr_path = os.path.join(app.config['RESULT_FOLDER'], 'correlation_heatmap.png')
            plt.savefig(corr_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 添加圖表路徑到數據信息
            data_info['target_hist'] = os.path.basename(target_hist_path)
            data_info['correlation_map'] = os.path.basename(corr_path)
            
        except Exception as e:
            logger.error(f"生成數據分析圖時出錯: {str(e)}")
            flash(f'生成數據分析圖時出錯，但數據處理成功')
        
        flash('數據處理成功！')
        return render_template('data_preview.html', 
                             data_info=data_info,
                             models=session_data['available_models'])
    
    except Exception as e:
        logger.error(f"處理數據時出錯: {str(e)}")
        flash(f'處理數據時出錯: {str(e)}')
        return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    """使用選擇的模型進行預測"""
    if session_data['processed_data'] is None:
        flash('請先上傳並處理數據')
        return redirect(url_for('index'))
    
    try:
        # 獲取選擇的模型
        model_path = request.form.get('model_path')
        
        if not model_path or not os.path.exists(model_path):
            flash('請選擇有效的模型')
            return redirect(url_for('index'))
        
        # 載入模型
        predictor = FatigueLifePredictor.load(model_path)
        session_data['predictor'] = predictor
        
        # 獲取處理後的數據
        X = session_data['processed_data']['X']
        y_true = session_data['processed_data']['y']
        
        # 進行預測
        y_pred = predictor.predict(X)
        
        # 初始化評估器
        evaluator = ModelEvaluator(output_dir=app.config['RESULT_FOLDER'])
        
        # 計算評估指標
        metrics = evaluator.calculate_metrics(y_true, y_pred, predictor.best_model_name)
        
        # 生成結果圖
        actual_vs_pred_path = os.path.join(app.config['RESULT_FOLDER'], 'actual_vs_predicted.png')
        plot_actual_vs_predicted(y_true.values, y_pred, 
                               title=f"{predictor.best_model_name} 預測結果",
                               filepath=actual_vs_pred_path)
        
        # 嘗試生成特徵重要性圖
        importance_path = None
        try:
            if hasattr(predictor.best_model, 'feature_importances_'):
                importance_path = os.path.join(app.config['RESULT_FOLDER'], 'feature_importance.png')
                plot_feature_importance(
                    session_data['feature_names'], 
                    predictor.best_model.feature_importances_,
                    title=f"{predictor.best_model_name} 特徵重要性",
                    filepath=importance_path
                )
        except Exception as e:
            logger.error(f"生成特徵重要性圖時出錯: {str(e)}")
        
        # 生成殘差分析圖
        residuals_path = os.path.join(app.config['RESULT_FOLDER'], 'residuals.png')
        evaluator.plot_residuals(y_true, y_pred, predictor.best_model_name, save_fig=True, show_fig=False)
        
        # 保存預測結果
        result_df = pd.DataFrame({
            '實際值': y_true.values,
            '預測值': y_pred.flatten(),
            '殘差': y_true.values - y_pred.flatten(),
            '百分比誤差(%)': 100 * (y_true.values - y_pred.flatten()) / y_true.values
        })
        result_csv_path = os.path.join(app.config['RESULT_FOLDER'], 'prediction_results.csv')
        result_df.to_csv(result_csv_path, index=False)
        
        # 保存結果到會話
        session_data['prediction_results'] = {
            'metrics': metrics,
            'actual_vs_pred_path': os.path.basename(actual_vs_pred_path),
            'importance_path': os.path.basename(importance_path) if importance_path else None,
            'residuals_path': os.path.basename(residuals_path),
            'result_csv_path': os.path.basename(result_csv_path),
            'sample_results': result_df.head(20).to_html(classes='table table-striped table-bordered'),
            'model_name': predictor.best_model_name
        }
        
        # 顯示預測結果頁面
        return render_template('prediction_results.html', 
                             results=session_data['prediction_results'])
    
    except Exception as e:
        logger.error(f"預測時出錯: {str(e)}")
        flash(f'預測時出錯: {str(e)}')
        return redirect(url_for('index'))

@app.route('/sensitivity_analysis', methods=['GET', 'POST'])
def sensitivity_analysis():
    """特徵敏感性分析頁面"""
    if session_data['predictor'] is None or session_data['processed_data'] is None:
        flash('請先上傳數據並進行預測')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            # 獲取選擇的特徵
            feature_name = request.form.get('feature_name')
            
            if not feature_name or feature_name not in session_data['feature_names']:
                flash('請選擇有效的特徵')
                return redirect(url_for('sensitivity_analysis'))
            
            # 獲取特徵索引
            feature_idx = session_data['feature_names'].index(feature_name)
            
            # 獲取數據和模型
            X = session_data['processed_data']['X'].values
            model = session_data['predictor'].best_model
            
            # 初始化評估器
            evaluator = ModelEvaluator(output_dir=app.config['RESULT_FOLDER'])
            
            # 進行敏感性分析
            sensitivity_path = os.path.join(app.config['RESULT_FOLDER'], f'sensitivity_{feature_name.replace(" ", "_")}.png')
            
            evaluator.perform_sensitivity_analysis(
                model, X, session_data['feature_names'], 
                feature_idx, model_name=session_data['predictor'].best_model_name,
                save_fig=True, show_fig=False
            )
            
            # 選擇第二個特徵進行交互分析
            feature_name2 = request.form.get('feature_name2')
            interaction_path = None
            
            if feature_name2 and feature_name2 in session_data['feature_names'] and feature_name2 != feature_name:
                feature_idx2 = session_data['feature_names'].index(feature_name2)
                
                interaction_path = os.path.join(app.config['RESULT_FOLDER'], 
                                        f'interaction_{feature_name.replace(" ", "_")}_{feature_name2.replace(" ", "_")}.png')
                
                evaluator.plot_two_feature_interaction(
                    model, X, session_data['feature_names'],
                    feature_idx, feature_idx2,
                    model_name=session_data['predictor'].best_model_name,
                    save_fig=True, show_fig=False
                )
                
                interaction_path = os.path.basename(interaction_path)
            
            # 顯示敏感性分析結果
            return render_template('sensitivity_results.html',
                                 feature_name=feature_name,
                                 feature_name2=feature_name2,
                                 sensitivity_path=os.path.basename(sensitivity_path),
                                 interaction_path=interaction_path,
                                 all_features=session_data['feature_names'])
            
        except Exception as e:
            logger.error(f"進行敏感性分析時出錯: {str(e)}")
            flash(f'敏感性分析時出錯: {str(e)}')
            return redirect(url_for('index'))
    
    # 顯示特徵選擇頁面
    return render_template('sensitivity_analysis.html', 
                         features=session_data['feature_names'])

@app.route('/download/<filename>')
def download_file(filename):
    """下載結果文件"""
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), as_attachment=True)

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """清除當前會話數據"""
    global session_data
    session_data = {
        'uploaded_file': None,
        'processed_data': None,
        'predictor': None,
        'feature_names': None,
        'target_column': None,
        'prediction_results': None,
        'available_models': load_available_models()
    }
    flash('會話數據已清除')
    return redirect(url_for('index'))

@app.route('/train_new_model', methods=['GET', 'POST'])
def train_new_model():
    """訓練新模型頁面"""
    if session_data['processed_data'] is None:
        flash('請先上傳並處理數據')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            # 獲取表單參數
            model_type = request.form.get('model_type', 'random_forest')
            tune_hyperparams = 'tune_hyperparams' in request.form
            model_name = request.form.get('model_name', datetime.now().strftime('%Y%m%d_%H%M%S'))
            
            # 獲取數據
            X = session_data['processed_data']['X']
            y = session_data['processed_data']['y']
            
            # 初始化預測器
            predictor = FatigueLifePredictor()
            
            # 創建適當的模型字典
            if model_type == 'random_forest':
                models = {
                    'random_forest': predictor._create_models()['random_forest']
                }
            elif model_type == 'gradient_boosting':
                models = {
                    'gradient_boosting': predictor._create_models()['gradient_boosting']
                }
            elif model_type == 'svr':
                models = {
                    'svr': predictor._create_models()['svr']
                }
            elif model_type == 'multiple':
                # 訓練多個模型
                models = {
                    'random_forest': predictor._create_models()['random_forest'],
                    'gradient_boosting': predictor._create_models()['gradient_boosting'],
                    'svr': predictor._create_models()['svr'],
                    'linear': predictor._create_models()['linear']
                }
            else:
                flash('不支援的模型類型')
                return redirect(url_for('train_new_model'))
            
            # 訓練模型
            flash('模型訓練已開始，請稍候...')
            
            # 對每個模型進行訓練和評估
            metrics = {}
            for name, model in models.items():
                logger.info(f"訓練模型: {name}")
                
                if tune_hyperparams:
                    model = predictor._tune_hyperparameters(name, model, X, y, cv=5)
                
                # 在整個數據集上訓練最終模型
                model.fit(X, y)
                predictor.models[name] = model
                
                # 進行預測並評估
                y_pred = model.predict(X)
                
                # 簡單的評估指標
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(y, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y, y_pred)
                
                metrics[name] = {
                    'rmse': rmse,
                    'r2': r2
                }
                
                logger.info(f"模型 {name} 評估: RMSE={rmse:.4f}, R²={r2:.4f}")
            
            # 找出最佳模型
            best_model_name = min(metrics.items(), key=lambda x: x[1]['rmse'])[0]
            predictor.best_model_name = best_model_name
            predictor.best_model = predictor.models[best_model_name]
            
            # 保存訓練好的模型
            model_filename = f"{model_name}.pkl"
            model_path = os.path.join(MODELS_DIR, model_filename)
            predictor.save(model_path)
            
            # 刷新可用模型列表
            session_data['available_models'] = load_available_models()
            
            flash(f'模型 {model_name} 訓練成功！最佳模型: {best_model_name}, RMSE: {metrics[best_model_name]["rmse"]:.4f}')
            return redirect(url_for('index'))
            
        except Exception as e:
            logger.error(f"訓練模型時出錯: {str(e)}")
            flash(f'訓練模型時出錯: {str(e)}')
            return redirect(url_for('train_new_model'))
    
    # 顯示訓練新模型表單
    return render_template('train_model.html')

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    """批量預測頁面"""
    if request.method == 'POST':
        # 檢查是否有文件
        if 'file' not in request.files:
            flash('沒有選擇文件')
            return redirect(request.url)
        
        file = request.files['file']
        
        # 檢查文件名是否為空
        if file.filename == '':
            flash('沒有選擇文件')
            return redirect(request.url)
        
        # 檢查文件類型是否允許
        if file and allowed_file(file.filename):
            try:
                # 獲取選擇的模型
                model_path = request.form.get('model_path')
                
                if not model_path or not os.path.exists(model_path):
                    flash('請選擇有效的模型')
                    return redirect(url_for('batch_predict'))
                
                # 保存上傳的文件
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # 載入模型
                predictor = FatigueLifePredictor.load(model_path)
                
                # 載入數據
                processor = DataProcessor()
                data = processor.load_data(file_path)
                
                # 檢查數據是否包含所有必要的特徵
                required_features = predictor.feature_names
                missing_features = [f for f in required_features if f not in data.columns]
                
                if missing_features:
                    flash(f'數據缺少以下必要特徵: {", ".join(missing_features)}')
                    return redirect(url_for('batch_predict'))
                
                # 僅保留必要的特徵
                prediction_data = data[required_features]
                
                # 進行預測
                predictions = predictor.predict(prediction_data)
                
                # 將預測結果添加到原始數據
                result_data = data.copy()
                result_data['預測疲勞壽命'] = predictions
                
                # 保存結果
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_filename = f"batch_prediction_{timestamp}.csv"
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                result_data.to_csv(result_path, index=False)
                
                # 顯示預測結果概述
                return render_template('batch_results.html',
                                     model_name=predictor.best_model_name,
                                     file_name=filename,
                                     sample_count=len(result_data),
                                     result_filename=result_filename,
                                     sample_results=result_data.head(10).to_html(classes='table table-striped table-bordered'))
                
            except Exception as e:
                logger.error(f"批量預測時出錯: {str(e)}")
                flash(f'批量預測時出錯: {str(e)}')
                return redirect(url_for('batch_predict'))
        
        else:
            flash(f'不支援的文件類型。允許的類型: {", ".join(ALLOWED_EXTENSIONS)}')
            return redirect(request.url)
    
    # 顯示批量預測表單
    return render_template('batch_predict.html', models=session_data['available_models'])

@app.errorhandler(404)
def page_not_found(e):
    """處理404錯誤"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """處理500錯誤"""
    logger.error(f"服務器內部錯誤: {str(e)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    try:
        # 加載可用的模型
        session_data['available_models'] = load_available_models()
        
        # 啟動應用
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.critical(f"啟動應用時出錯: {str(e)}")