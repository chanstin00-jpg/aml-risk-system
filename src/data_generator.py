import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker

# 設定隨機種子以確保模擬資料的可重複性
np.random.seed(42)
random.seed(42)
fake = Faker('en_US')

# 檔案輸出路徑
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/raw")
os.makedirs(DATA_DIR, exist_ok=True)

# 系統規模參數
NUM_CUSTOMERS = 5000       # 帳戶總數
NUM_TRANSACTIONS = 100000  # 交易日誌總條數


def generate_kyc_and_ownership(num_customers):
    """生成 KYC 客戶檔案與 Neo4j 股權穿透圖資"""
    print(f"1/3 正在生成 {num_customers} 個帳戶的 KYC 與企業股權拓撲關係...")
    
    account_ids = [f"ACC_{100000 + i}" for i in range(num_customers)]
    countries = ['US', 'HK', 'SG', 'GB', 'KY', 'VG', 'CN', 'TW'] # 包含開曼(KY)、維京(VG)等避稅天堂
    account_types = ['INDIVIDUAL', 'CORPORATE', 'SHELL_COMPANY']
    
    customers = []
    for acc_id in account_ids:
        acc_type = np.random.choice(account_types, p=[0.75, 0.20, 0.05])
        country = np.random.choice(countries, p=[0.4, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05])
        
        # 避稅天堂或空殼公司的基礎風險值偏高
        base_risk = random.uniform(0.7, 0.95) if acc_type == 'SHELL_COMPANY' or country in ['KY', 'VG'] else random.uniform(0.05, 0.4)
        
        customers.append({
            "account_id": acc_id,
            "customer_name": fake.company() if acc_type != 'INDIVIDUAL' else fake.name(),
            "account_type": acc_type,
            "country": country,
            "risk_score": round(base_risk, 3),
            "created_at": fake.date_between(start_date='-3y', end_date='-1y').strftime('%Y-%m-%d')
        })
    
    df_customers = pd.DataFrame(customers)
    
    # 生成 Neo4j 企業多層股權/受益人拓撲網絡 (Ownership Graph)
    ownerships = []
    corp_ids = df_customers[df_customers['account_type'].isin(['CORPORATE', 'SHELL_COMPANY'])]['account_id'].tolist()
    
    for i, child in enumerate(corp_ids):
        # 每個公司隨機 1-3 個母公司或持股受益人
        parents = random.sample(account_ids, k=random.randint(1, 3))
        for parent in parents:
            if parent != child:
                ownerships.append({
                    "parent_id": parent,
                    "child_id": child,
                    "shareholding_pct": round(random.uniform(0.15, 0.85), 2),
                    "relation_type": "BENEFICIAL_OWNER" if "ACC_1" in parent else "PARENT_COMPANY"
                })
                
    df_ownership = pd.DataFrame(ownerships)
    return df_customers, df_ownership


def generate_transactions(df_customers, total_txs):
    """生成 10 萬條交易明細，並注入環形洗錢與高頻分流洗錢鏈"""
    print(f"2/3 正在生成 {total_txs} 條交易明細並注入洗錢異常模式...")
    
    account_list = df_customers['account_id'].tolist()
    start_date = datetime(2026, 1, 1)
    
    transactions = []
    
    # -------------------------------------------------------------
    # 注入洗錢樣態 1：多層環形轉帳 (Circular Money Laundering Rings)
    # A -> B -> C -> D -> A (短時間內資金歸原主，帶微量損耗)
    # -------------------------------------------------------------
    num_rings = 15
    ring_tx_count = 0
    for r in range(num_rings):
        ring_members = random.sample(account_list, k=4) # 4節點環
        base_amount = random.uniform(50000, 200000)
        ring_start_time = start_date + timedelta(days=random.randint(0, 180), hours=random.randint(0, 20))
        
        for i in range(len(ring_members)):
            sender = ring_members[i]
            receiver = ring_members[(i + 1) % len(ring_members)]
            # 資金每轉過一手稍微扣除 1%-2% 手續費損耗
            amount = round(base_amount * ((0.98) ** i), 2)
            tx_time = ring_start_time + timedelta(minutes=i * 15 + random.randint(1, 5))
            
            transactions.append({
                "transaction_id": f"TX_RING_{r}_{i}",
                "sender_id": sender,
                "receiver_id": receiver,
                "amount": amount,
                "timestamp": tx_time.strftime('%Y-%m-%d %H:%M:%S'),
                "channel": "WIRE",
                "is_laundering": 1,
                "laundering_type": "CIRCULAR_RING"
            })
            ring_tx_count += 1

    # -------------------------------------------------------------
    # 注入洗錢樣態 2：高頻化整為零分流 (Structuring / Smurfing)
    # 大額資金先分散發送至 20 個小帳戶，再快速匯聚至目標帳戶
    # -------------------------------------------------------------
    num_smurf_clusters = 20
    smurf_tx_count = 0
    for s in range(num_smurf_clusters):
        source_acc = random.choice(account_list)
        target_acc = random.choice([a for a in account_list if a != source_acc])
        mule_accs = random.sample([a for a in account_list if a not in [source_acc, target_acc]], k=15)
        
        cluster_start = start_date + timedelta(days=random.randint(0, 180), hours=random.randint(0, 20))
        
        for mule in mule_accs:
            # 規避 10,000 美金申報門檻（故意發 9,500 ~ 9,900）
            sub_amount = round(random.uniform(9500, 9900), 2)
            
            # 階段 1：Source -> Mule (分流)
            t1 = cluster_start + timedelta(minutes=random.randint(1, 30))
            transactions.append({
                "transaction_id": f"TX_SMURF_OUT_{s}_{mule}",
                "sender_id": source_acc,
                "receiver_id": mule,
                "amount": sub_amount,
                "timestamp": t1.strftime('%Y-%m-%d %H:%M:%S'),
                "channel": "MOBILE",
                "is_laundering": 1,
                "laundering_type": "STRUCTURING_SMURF"
            })
            
            # 階段 2：Mule -> Target (歸集)
            t2 = t1 + timedelta(minutes=random.randint(10, 60))
            transactions.append({
                "transaction_id": f"TX_SMURF_IN_{s}_{mule}",
                "sender_id": mule,
                "receiver_id": target_acc,
                "amount": round(sub_amount * 0.99, 2),
                "timestamp": t2.strftime('%Y-%m-%d %H:%M:%S'),
                "channel": "WIRE",
                "is_laundering": 1,
                "laundering_type": "STRUCTURING_SMURF"
            })
            smurf_tx_count += 2

    # -------------------------------------------------------------
    # 填補剩餘的正常金融交易 (Normal Transactions)
    # -------------------------------------------------------------
    remaining_txs = total_txs - (ring_tx_count + smurf_tx_count)
    channels = ['SWIFT', 'WIRE', 'ACH', 'MOBILE', 'ATM']
    
    senders = np.random.choice(account_list, size=remaining_txs)
    receivers = np.random.choice(account_list, size=remaining_txs)
    
    # 金融金額對數正態分佈（符合真實小額多、大額少的常態）
    amounts = np.random.lognormal(mean=7.0, sigma=1.2, size=remaining_txs).round(2)
    amounts = np.clip(amounts, 10.0, 500000.0)
    
    # 隨機時間戳分佈
    random_seconds = np.random.randint(0, 180 * 86400, size=remaining_txs)
    
    for i in range(remaining_txs):
        s_id = senders[i]
        r_id = receivers[i] if receivers[i] != s_id else random.choice(account_list)
        t_time = start_date + timedelta(seconds=int(random_seconds[i]))
        
        transactions.append({
            "transaction_id": f"TX_NORM_{i:06d}",
            "sender_id": s_id,
            "receiver_id": r_id,
            "amount": amounts[i],
            "timestamp": t_time.strftime('%Y-%m-%d %H:%M:%S'),
            "channel": np.random.choice(channels, p=[0.2, 0.3, 0.25, 0.2, 0.05]),
            "is_laundering": 0,
            "laundering_type": "NORMAL"
        })

    df_tx = pd.DataFrame(transactions)
    # 按時間戳排序，模擬真實串流時間序
    df_tx = df_tx.sort_values(by="timestamp").reset_index(drop=True)
    return df_tx


def main():
    df_customers, df_ownership = generate_kyc_and_ownership(NUM_CUSTOMERS)
    df_tx = generate_transactions(df_customers, NUM_TRANSACTIONS)
    
    print("3/3 正在將資料寫入 data/raw/ 目錄...")
    
    cust_path = os.path.join(DATA_DIR, "customers.csv")
    owner_path = os.path.join(DATA_DIR, "ownership.csv")
    tx_path = os.path.join(DATA_DIR, "transactions.csv")
    
    df_customers.to_csv(cust_path, index=False)
    df_ownership.to_csv(owner_path, index=False)
    df_tx.to_csv(tx_path, index=False)
    
    print("\n✅ 資料生成成功！產出檔案如下：")
    print(f" - 客戶 KYC 檔案: {cust_path} ({len(df_customers)} 筆)")
    print(f" - 股權穿透圖資: {owner_path} ({len(df_ownership)} 筆)")
    print(f" - 交易日誌資料: {tx_path} ({len(df_tx)} 筆, 洗錢標籤筆數: {df_tx['is_laundering'].sum()} 筆)")


if __name__ == "__main__":
    main()