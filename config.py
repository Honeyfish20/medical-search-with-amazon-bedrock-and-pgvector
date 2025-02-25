import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# AWS配置
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Cohere配置
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "cohere.rerank-v3-5:0")

# 模型配置
NOVA_MODEL_ID = os.getenv("NOVA_MODEL_ID")
TITAN_MODEL_ID = os.getenv("TITAN_MODEL_ID")
DEEPSEEK_MODEL_ID = os.getenv("DEEPSEEK_MODEL_ID")

# 数据库配置
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")
