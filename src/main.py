import os
import time
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. 建立 FastAPI 實例
app = FastAPI(
    title="金融風控與反洗錢 (AML) 實時推理 API",
    description="結合 LightGBM 評分卡與 Isolation Forest 異常檢測的高併發風控引擎",
    version="1.0.0"
)

# 2. 載入模型與特徵對齊清單
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")

try:
    lgb_model = joblib.load(os.path.join(MODEL_DIR, "lgb_aml_model.pkl"))
    iso_forest = joblib.load(os.path.join(MODEL_DIR, "iso_forest_model.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    print("✅ 模型與特徵對齊矩陣載入成功！")
except Exception as e:
    print(f"⚠️ 模型載入失敗，請確認 Phase 3 已成功執行: {e}")

# 3. 定義 API 請求數據結構 (Pydantic Schema)
class TransactionPayload(BaseModel):
    transaction_id: str = Field(..., example="TX_TEST_99999")
    sender_id: str = Field(..., example="ACC_100001")
    receiver_id: str = Field(..., example="ACC_100002")
    amount: float = Field(..., example=9850.00)
    channel: str = Field(..., example="WIRE")
    
    # 即時計算或上游傳入的統計特徵 (預設值供快速測試)
    out_tx_count: int = Field(default=25)
    out_avg_amount: float = Field(default=2000.0)
    out_max_amount: float = Field(default=15000.0)
    graph_pagerank: float = Field(default=0.0025)
    graph_degree_centrality: float = Field(default=0.015)

# 4. 健康檢查端點
@app.get("/health")
def health_check():
    return {"status": "HEALTHY", "engine": "FastAPI + LightGBM AML Engine"}

# 5. 實時風控推理端點
@app.post("/api/v1/predict")
def predict_transaction_risk(payload: TransactionPayload):
    start_time = time.time()
    
    # 將輸入轉為 DataFrame 並進行特徵對齊
    input_dict = payload.dict()
    
    # 派生與對齊 45 維特徵 (缺乏的特徵自動補 0)
    input_dict['tx_to_avg_ratio'] = input_dict['amount'] / (input_dict['out_avg_amount'] + 1)
    input_dict['tx_to_max_ratio'] = input_dict['amount'] / (input_dict['out_max_amount'] + 1)
    
    df_input = pd.DataFrame([input_dict])
    
    # 構建全量特徵向量 (與 Phase 3 訓練時的欄位順序一致)
    X_infer = pd.DataFrame(0, index=[0], columns=feature_cols)
    for col in feature_cols:
        if col in df_input.columns:
            X_infer[col] = df_input[col]

    # 模型推理 (Inference)
    lgb_risk_score = float(lgb_model.predict_proba(X_infer)[:, 1][0])
    iso_anomaly_score = float(iso_forest.decision_function(X_infer)[0]) # 分數越低越異常
    
    # 混合決策樹機制
    is_high_risk = False
    decision = "APPROVE"
    
    if lgb_risk_score > 0.7 or iso_anomaly_score < -0.15:
        is_high_risk = True
        decision = "BLOCK_AND_ALERT"
    elif lgb_risk_score > 0.4:
        decision = "MANUAL_REVIEW"
        
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    return {
        "transaction_id": payload.transaction_id,
        "risk_score": round(lgb_risk_score, 4),
        "anomaly_score": round(iso_anomaly_score, 4),
        "is_high_risk": is_high_risk,
        "decision": decision,
        "latency_ms": f"{latency_ms}ms"
    }

from fastapi.responses import RedirectResponse

# 點擊首頁網址時自動重定向到 /docs 頁面
@app.get("/")
def main_root():
    return RedirectResponse(url="/docs")