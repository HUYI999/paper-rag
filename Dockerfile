FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（markitdown 解析 PDF 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# 先安装依赖，利用 Docker 缓存层
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建运行时目录
RUN mkdir -p data/db/chroma data/db/bm25 data/images logs

EXPOSE 8501

# 启动 Streamlit UI
CMD ["streamlit", "run", "src/observability/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
