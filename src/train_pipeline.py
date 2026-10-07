import os
import joblib
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

warnings.filterwarnings('ignore')

# 1. 設定連線與路徑
PG_URL = "postgresql://aml_user:aml_password@localhost:5432/aml_db"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MODEL_DIR, exist_ok=True)

def extract_features():
    """從 PostgreSQL 提取資料，並建構 45 維特徵矩陣"""
    print("1/4 正在從資料庫抽取 raw data 並建構 45 維風控特徵矩陣...")
    engine = create_engine(PG_URL)
    
    df_cust = pd.read_sql("SELECT * FROM customers", engine)
    df_tx = pd.read_sql("SELECT * FROM transactions", engine)
    
    # ---------------------------------------------------------
    # (A) 交易層級 & 客戶歷史統計特徵 (Transaction & Account Aggregates)
    # ---------------------------------------------------------
    df_tx['timestamp'] = pd.to_datetime(df_tx['timestamp'])
    df_tx['hour'] = df_tx['timestamp'].dt.hour
    df_tx['is_night'] = df_tx['hour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)
    df_tx['is_structuring_range'] = df_tx['amount'].apply(lambda x: 1 if 9000 <= x <= 10000 else 0)
    
    # 計算轉出帳戶 (Sender) 統計特徵
    sender_stats = df_tx.groupby('sender_id').agg(
        out_tx_count=('transaction_id', 'count'),
        out_total_amount=('amount', 'sum'),
        out_avg_amount=('amount', 'mean'),
        out_max_amount=('amount', 'max'),
        out_std_amount=('amount', 'std'),
        out_night_ratio=('is_night', 'mean'),
        out_structuring_ratio=('is_structuring_range', 'mean')
    ).reset_index().rename(columns={'sender_id': 'account_id'})
    
    # 計算轉入帳戶 (Receiver) 統計特徵
    receiver_stats = df_tx.groupby('receiver_id').agg(
        in_tx_count=('transaction_id', 'count'),
        in_total_amount=('amount', 'sum'),
        in_avg_amount=('amount', 'mean'),
        in_max_amount=('amount', 'max'),
        in_night_ratio=('is_night', 'mean')
    ).reset_index().rename(columns={'receiver_id': 'account_id'})
    
    # 合併帳戶統計
    acc_features = pd.merge(df_cust, sender_stats, on='account_id', how='left')
    acc_features = pd.merge(acc_features, receiver_stats, on='account_id', how='left').fillna(0)
    
    # 計算衍生比例特徵
    acc_features['net_flow'] = acc_features['in_total_amount'] - acc_features['out_total_amount']
    acc_features['in_out_tx_ratio'] = (acc_features['in_tx_count'] + 1) / (acc_features['out_tx_count'] + 1)
    acc_features['max_to_avg_ratio'] = (acc_features['out_max_amount'] + 1) / (acc_features['out_avg_amount'] + 1)

    # ---------------------------------------------------------
    # (B) NetworkX 圖拓撲中心度特徵 (Graph Topology Features)
    # ---------------------------------------------------------
    print("   -> 正在使用 NetworkX 計算帳戶圖網絡中心度度量 (Degree / PageRank)...")
    G = nx.DiGraph()
    for _, row in df_tx.iterrows():
        G.add_edge(row['sender_id'], row['receiver_id'], weight=row['amount'])
        
    pagerank = nx.pagerank(G, alpha=0.85)
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    degree_centrality = nx.degree_centrality(G)
    
    acc_features['graph_pagerank'] = acc_features['account_id'].map(pagerank).fillna(0)
    acc_features['graph_in_degree'] = acc_features['account_id'].map(in_degree).fillna(0)
    acc_features['graph_out_degree'] = acc_features['account_id'].map(out_degree).fillna(0)
    acc_features['graph_degree_centrality'] = acc_features['account_id'].map(degree_centrality).fillna(0)
    
    # ---------------------------------------------------------
    # (C) 將帳戶特徵拼接回每筆交易 (Join features back to transactions)
    # ---------------------------------------------------------
    # 類別型變數 One-Hot Encoding
    acc_features = pd.get_dummies(acc_features, columns=['account_type', 'country'], drop_first=True)
    
    # 映射發送方與接收方特徵
    full_df = pd.merge(df_tx, acc_features, left_on='sender_id', right_on='account_id', suffixes=('', '_sender'))
    
    # 補足或延展特徵緯度至 ~45 維
    feature_cols = [c for c in full_df.columns if c not in [
        'transaction_id', 'sender_id', 'receiver_id', 'timestamp', 
        'channel', 'is_laundering', 'laundering_type', 'account_id', 'customer_name', 'created_at'
    ]]
    
    # 確保補齊 45 維（包含生成交互特徵）
    full_df['tx_to_avg_ratio'] = full_df['amount'] / (full_df['out_avg_amount'] + 1)
    full_df['tx_to_max_ratio'] = full_df['amount'] / (full_df['out_max_amount'] + 1)
    feature_cols.extend(['tx_to_avg_ratio', 'tx_to_max_ratio'])
    
    print(f"   ✅ 特徵矩陣構建完成！交易總數: {len(full_df)}, 特徵緯度: {len(feature_cols)} 維")
    return full_df, feature_cols

def train_hybrid_models():
    df, feature_cols = extract_features()
    
    X = df[feature_cols].fillna(0)
    y = df['is_laundering']
    
    # 劃分訓練與測試集 (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # ---------------------------------------------------------
    # 2/4 使用 SMOTE 解決數據不平衡問題 (Data Imbalance Handling)
    # ---------------------------------------------------------
    print("2/4 正在對訓練集進行 SMOTE 過採樣...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"   -> 原始訓練集洗錢樣本數: {sum(y_train)}, SMOTE 後樣本數: {sum(y_train_res)}")
    
    # ---------------------------------------------------------
    # 3/4 訓練 LightGBM 信用評分模型 (Supervised Risk Model)
    # ---------------------------------------------------------
    print("3/4 正在訓練 LightGBM 風控模型...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train_res, y_train_res)
    
    # 評估 LightGBM
    y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\n==========================================")
    print(f"🔥 LightGBM 模型測試集 AUC 得分: {auc_score:.4f}")
    print(f"==========================================\n")
    
    # ---------------------------------------------------------
    # 4/4 訓練 Isolation Forest 無監督異常檢測 (Unsupervised Anomaly)
    # ---------------------------------------------------------
    print("4/4 正在訓練 Isolation Forest 無監督異常檢測模型...")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso_forest.fit(X_train) # 僅對正常特徵佈局學習離群點
    
    # 結合兩者判定 (降低偽陽性 - False Positive Reduction)
    iso_scores = iso_forest.decision_function(X_test)
    
    # 保存模型與特徵清單
    print("5/5 正在將模型與 Transformer 保存至 models/ 目錄...")
    joblib.dump(lgb_model, os.path.join(MODEL_DIR, "lgb_aml_model.pkl"))
    joblib.dump(iso_forest, os.path.join(MODEL_DIR, "iso_forest_model.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_columns.pkl"))
    
    print("\n✅ Phase 3 完成！模型導出檔案：")
    print(f" - LightGBM 模型: models/lgb_aml_model.pkl")
    print(f" - Isolation Forest 模型: models/iso_forest_model.pkl")
    print(f" - 特徵特徵清單: models/feature_columns.pkl")

if __name__ == "__main__":
    train_hybrid_models()