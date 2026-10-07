from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
from airflow import DAG
from airflow.operators.python import PythonOperator

# 資料庫連線
PG_URL = "postgresql://aml_user:aml_password@localhost:5432/aml_db"

default_args = {
    'owner': 'aml_risk_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'aml_daily_risk_recalculation',
    default_args=default_args,
    description='每日定時重算全量帳戶風險得分與異常交易預警流',
    schedule_interval='0 2 * * *',  # 每日凌晨 02:00 定時觸發
    catchup=False
)

def task_extract_and_recalculate(**kwargs):
    """ Task 1: 提取昨日全量交易，進行批次風險重算 """
    print("Airflow Task 1: 開始執行每日全量交易風險重算...")
    engine = create_engine(PG_URL)
    df_tx = pd.read_sql("SELECT * FROM transactions WHERE amount > 5000", engine)
    
    # 標記高風險大額交易寫入預警暫存表
    df_high_risk = df_tx[df_tx['amount'] > 100000]
    df_high_risk.to_sql("daily_high_risk_alerts", engine, if_exists="replace", index=False)
    print(f"✅ 發現 {len(df_high_risk)} 筆高風險大額交易，已存入 daily_high_risk_alerts 表。")

def task_update_compliance_dashboard(**kwargs):
    """ Task 2: 更新合規看板與風險統計數據 """
    print("Airflow Task 2: 正在更新每日合規風控 KPI 看板...")
    engine = create_engine(PG_URL)
    summary_sql = """
    CREATE TABLE IF NOT EXISTS daily_risk_summary AS
    SELECT 
        CURRENT_DATE as report_date,
        COUNT(*) as total_tx_count,
        SUM(amount) as total_volume,
        SUM(CASE WHEN amount > 100000 THEN 1 ELSE 0 END) as alert_count
    FROM transactions;
    """
    with engine.connect() as conn:
        conn.execute(summary_sql)
    print("✅ 合規看板統計數據更新完成！")

# 定義 Task 節點與相依關係
t1 = PythonOperator(
    task_id='recalculate_daily_risk',
    python_callable=task_extract_and_recalculate,
    dag=dag,
)

t2 = PythonOperator(
    task_id='update_dashboard_metrics',
    python_callable=task_update_compliance_dashboard,
    dag=dag,
)

# 執行順序: Task 1 成功後才執行 Task 2
t1 >> t2