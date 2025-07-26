# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量
ENV FLASK_APP=rest_api.py
ENV FLASK_ENV=production

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["flask", "run", "--host=0.0.0.0"]