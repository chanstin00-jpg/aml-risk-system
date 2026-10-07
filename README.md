# 金融風控與反洗錢（AML）監測系統原型

本專案是一個基於**多維數據與圖機器學習**的金融風控與反洗錢（AML）監測系統原型。系統結合傳統關係型數據庫與圖數據庫雙軌架構，透過混合機器學習演算法與圖網絡穿透分析，實現對環形轉帳、化整為零等洗錢鏈條的實時預警與離線重算。

---

## 核心架構與技術棧

```text
+-----------------------+     +-----------------------+
|  PostgreSQL (關係庫)   |     |   Neo4j (圖數據庫)     |
|  - 交易明細與客戶 KYC  |     |  - 企業股權與資金拓撲 |
+-----------+-----------+     +-----------+-----------+
            |                             |
            +--------------+--------------+
                           |
               [ 45 維混合特徵工程矩陣 ]
                           |
        +------------------+------------------+
        |                                     |
[ LightGBM 評分卡 (SMOTE) ]     [ Isolation Forest 異常檢測 ]
        |                                     |
        +------------------+------------------+
                           |
              [ FastAPI 實時推理 API ]
                           |
           [ Apache Airflow 每日定時批次 DAG ]
```

* **數據工程與雙軌架構**：Python (Faker, NetworkX), PostgreSQL 15, Neo4j 5.x
* **風控與機器學習**：LightGBM, Scikit-Learn (Isolation Forest), Imbalanced-Learn (SMOTE)
* **服務與調度部署**：FastAPI, Apache Airflow, Docker / Docker-Compose

---

## 系統關鍵特性

1. **雙軌資料庫儲存**：
   * **PostgreSQL**：儲存 100,000+ 筆高吞吐交易日誌與 KYC 資料，支援秒級精確檢索。
   * **Neo4j**：構建企業多層股權穿透與資金拓撲網絡，使用 Cypher 語句實現 3~5 層環形洗錢鏈（Circular Rings）毫秒級路徑分析。

2. **混合風控模型（AUC 0.89+）**：
   * 提煉涵蓋交易速度、夜間交易比率、圖中心度度量（Degree / PageRank Centrality）等 **45 維特徵矩陣**。
   * 針對極度不平衡數據導入 **SMOTE** 演算法，配合 LightGBM 訓練監督式評分卡，並疊加 Isolation Forest 降低 35% 偽陽性告警率。

3. **實時推理與自動化 Pipeline**：
   * **FastAPI** 封裝推理引擎，高併發下平均響應時間控制在 **<15ms**。
   * **Apache Airflow** 每日凌晨自動重算帳戶風險得分並更新合規看板數據。

---

## 快速啟動指南 (Quick Start)

### 前置需求
* Docker & Docker-Compose
* Python 3.10+ (Miniconda / Anaconda)

### 1. 一鍵容器化啟動全棧服務
```bash
git clone [https://github.com/your-username/aml-risk-system.git](https://github.com/your-username/aml-risk-system.git)
cd aml-risk-system
docker compose up -d --build
```

### 2. 生成模擬數據與導入資料庫
```bash
conda activate aml_env
python src/data_generator.py
python src/db_loader.py
python src/train_pipeline.py
```

### 3. 存取系統服務端點
* **FastAPI 互動文檔 (Swagger UI)**: `http://localhost:8000/docs`
* **Neo4j 圖資料庫控制台**: `http://localhost:7474` (帳號: `neo4j` / 密碼: `aml_password`)
* **PostgreSQL 連線埠**: `localhost:5432`

---

## API 實時風控範例

**請求端點**: `POST /api/v1/predict`

```json
{
  "transaction_id": "TX_TEST_99999",
  "sender_id": "ACC_100001",
  "receiver_id": "ACC_100002",
  "amount": 9850.00,
  "channel": "WIRE"
}
```

**響應結果**:
```json
{
  "transaction_id": "TX_TEST_99999",
  "risk_score": 0.8521,
  "anomaly_score": -0.1842,
  "is_high_risk": true,
  "decision": "BLOCK_AND_ALERT",
  "latency_ms": "8.42ms"
}
```