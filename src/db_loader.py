import os
import pandas as pd
from sqlalchemy import create_engine
from neo4j import GraphDatabase

# 資料路徑
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/raw")

# 1. 資料庫連線設定
PG_URL = "postgresql://aml_user:aml_password@localhost:5432/aml_db"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "aml_password")


def load_to_postgres():
    """將 CSV 資料匯入 PostgreSQL ( relational tables )"""
    print("1/2 正在將交易與 KYC 資料寫入 PostgreSQL...")
    engine = create_engine(PG_URL)

    df_cust = pd.read_csv(f"{DATA_DIR}/customers.csv")
    df_tx = pd.read_csv(f"{DATA_DIR}/transactions.csv")

    # 寫入 PostgreSQL
    df_cust.to_sql("customers", engine, if_exists="replace", index=False)
    df_tx.to_sql("transactions", engine, if_exists="replace", index=False)
    print("   ✅ PostgreSQL 匯入完成！ (customers, transactions 表建置完畢)")


def load_to_neo4j():
    """將股權關係與洗錢交易鏈拓撲寫入 Neo4j Graph DB"""
    print("2/2 正在將股權關係與交易圖譜寫入 Neo4j...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    
    df_cust = pd.read_csv(f"{DATA_DIR}/customers.csv")
    df_owner = pd.read_csv(f"{DATA_DIR}/ownership.csv")
    df_tx = pd.read_csv(f"{DATA_DIR}/transactions.csv")
    
    # 篩選出標記為洗錢或大額交易的記錄導入 Neo4j (節省圖檢索記憶體)
    df_tx_graph = df_tx[(df_tx['is_laundering'] == 1) | (df_tx['amount'] > 50000)]

    with driver.session() as session:
        # 清空舊數據
        session.run("MATCH (n) DETACH DELETE n")

        # 建置帳戶/實體節點 (Account Nodes)
        print("   -> 建立客戶與企業節點...")
        cust_records = df_cust.to_dict('records')
        session.run("""
            UNWIND $records AS rec
            CREATE (a:Account {
                id: rec.account_id,
                name: rec.customer_name,
                type: rec.account_type,
                country: rec.country,
                riskScore: rec.risk_score
            })
        """, records=cust_records)

        # 建置股權/控制權邊 (OWNERSHIP Edges)
        print("   -> 建立企業多層股權穿透邊 (OWNERSHIP)...")
        owner_records = df_owner.to_dict('records')
        session.run("""
            UNWIND $records AS rec
            MATCH (parent:Account {id: rec.parent_id})
            MATCH (child:Account {id: rec.child_id})
            CREATE (parent)-[:OWNS {shareholding: rec.shareholding_pct, type: rec.relation_type}]->(child)
        """, records=owner_records)

        # 建置資金轉帳邊 (TRANSFER Edges)
        print("   -> 建立資金交易拓撲邊 (TRANSFER)...")
        tx_records = df_tx_graph.to_dict('records')
        session.run("""
            UNWIND $records AS rec
            MATCH (s:Account {id: rec.sender_id})
            MATCH (r:Account {id: rec.receiver_id})
            CREATE (s)-[:TRANSFERRED {
                tx_id: rec.transaction_id,
                amount: rec.amount,
                timestamp: rec.timestamp,
                is_laundering: rec.is_laundering,
                laundering_type: rec.laundering_type
            }]->(r)
        """, records=tx_records)

    driver.close()
    print("   ✅ Neo4j 圖資料庫匯入完成！")


if __name__ == "__main__":
    load_to_postgres()
    load_to_neo4j()