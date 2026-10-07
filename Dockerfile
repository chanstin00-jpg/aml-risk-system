FROM python:3.10-slim

WORKDIR /app

# 安裝 C 語言庫與 OpenMP (LightGBM 所需依賴)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製代碼與模型檔
COPY src/ /app/src/
COPY models/ /app/models/

EXPOSE 8000

# 啟動 Uvicorn 服務
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]