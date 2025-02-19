# 医学知识查询系统

基于AWS Bedrock的医学知识查询系统，通过多模型组合实现智能医学知识问答。本方案特别适用于数据和应用部署在AWS中国区，而需要调用AWS Global区域的Bedrock服务的混合架构场景。

## 架构说明

本系统采用混合云架构：
- 数据存储：使用AWS中国区的Amazon Aurora PostgreSQL
- 应用部署：部署在AWS中国区的EC2实例上
- AI模型服务：调用AWS Global区域（如us-west-2）的Amazon Bedrock服务
- 网络连接：通过配置跨区域访问实现中国区应用访问Global区域的Bedrock服务

此架构设计可以：
- 满足数据本地化要求
- 充分利用AWS Global区域的AI能力
- 实现低延迟的用户访问体验
- 控制跨区域数据传输成本

## 项目结构

```
medical_chatbot/
├── app.py                # Gradio Web应用主程序
├── config.py            # 配置文件
├── requirements.txt     # 项目依赖
├── words_embedding.py   # 文档向量生成程序
└── README_ZH.md           # 说明文档
```

## 功能特点

- 基于关键词的预过滤
- 多模型组合查询（Nova、Titan、Cohere、Deepseek）
- 向量相似度匹配
- 实时问答交互
- PostgreSQL向量数据库支持

## 环境要求

- Python 3.9+
- AWS账号（具有Bedrock访问权限）
- PostgreSQL数据库（AWS RDS）
- EC2实例（建议至少2GB RAM）

## 数据库配置指南

### 1. 部署 Aurora PostgreSQL 数据库

- 选择支持pgvector的Aurora版本(15.3、14.8、13.11、12.15及更高版本)
- 本项目使用Aurora PostgreSQL 13.12
- 具体部署步骤参考AWS文档：[创建Aurora PostgreSQL数据库集群](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/Aurora.CreateInstance.html)

### 2. Aurora数据库配置调整

1. 修改Aurora DB实例自定义参数组:
```sql
maintenance_work_mem = 10240000 (10GB)
```

2. 重启数据库集群使配置生效

3. 验证参数设置:
```sql
test=> show maintenance_work_mem;
 maintenance_work_mem
----------------------
 10000MB
```

### 3. 启用pgvector插件

1. 连接到数据库执行:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

2. 验证插件安装:
```sql
\dx vector
```

### 4. 建表并导入医疗数据

1. 下载测试数据:
```bash
git clone https://github.com/zhangsheng93/cMedQA2.git
unzip answer.zip
```

2. 创建数据表:
```sql
CREATE TABLE text_embedding_cos (
    id int,
    doc_type int,
    doc text,
    embedding_doc vector(1536) null,
    keywords text null
);
ALTER TABLE text_embedding_cos ADD PRIMARY KEY(id);
```

3. 导入数据:
```sql
\copy text_embedding_cos(id, doc_type, doc) from '/home/centos/answer.csv' WITH DELIMITER ',' CSV HEADER;
```

### 5. 生成文档Embedding

1. 安装依赖:
```bash
python -m pip install -r requirements.txt
```

2. 执行embedding生成:
```bash
python words_embedding.py -m embedding -r 240000
```

3. 创建向量索引:
```sql
CREATE INDEX ON text_embedding_cos 
USING ivfflat (embedding_doc vector_cosine_ops) 
WITH (lists = 10000);
```

4. 创建全文检索索引:
```sql
CREATE INDEX idx_cos_gin_keywords 
ON text_embedding_cos 
USING GIN(to_tsvector('simple', keywords));
```

5 创建欧距索引
-- 测试数据大概20W行，桶选择10000个，按照Lists=rows / 10 for up to 1M rows andsqrt(rows) for over 1M rows;
```sql
CREATE INDEX ON text_embedding USING ivfflat (embedding_doc vector_l2_ops) WITH(lists = 10000);
```

注意事项:
- embedding生成过程较耗时(约2小时/15万文档)
- 建议先用小数据集测试
- 确保数据库有足够存储空间
- 定期备份向量数据
- 监控索引创建过程

## 快速部署指南

### 1. 环境准备
```bash
# 更新系统并安装Python
sudo yum update -y
sudo yum install python3 python3-pip -y

# 创建项目目录
mkdir medical_chatbot
cd medical_chatbot

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate
```

### 2. 安装依赖
```bash
# 卸载可能存在的旧版本
pip uninstall boto3 botocore -y

# 安装依赖
pip install -r requirements.txt --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple


###安装依赖解决方案
pydantic.errors.PydanticSchemaGenerationError: Unable to generate pydantic-core schema for <class 'starlette.requests.Request'>

rm -rf venv
python3 -m venv venv
source venv/bin/activate

# 2. 升级pip
pip install --upgrade pip

# 3. 按顺序安装关键依赖
pip install pydantic==1.10.13 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install fastapi==0.95.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install gradio==3.50.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 安装其他依赖
pip install -r requirements.txt --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 配置文件
确保 config.py 包含正确的配置：
```python
# AWS配置
AWS_REGION = "<your-aws-region>"              # AWS区域，如：us-west-2
AWS_ACCESS_KEY_ID = "<your-access-key>"       # AWS访问密钥ID
AWS_SECRET_ACCESS_KEY = "<your-secret-key>"   # AWS访问密钥

# Cohere配置
COHERE_API_KEY = "cohere.rerank-v3-5:0"

# 模型配置
NOVA_MODEL_ID = "<your-nova-model-arn>"       # Nova模型ARN
TITAN_MODEL_ID = "<your-titan-model-id>"      # Titan模型ID
DEEPSEEK_MODEL_ID = "<your-deepseek-model-arn>" # Deepseek模型ARN

# 数据库配置
DB_HOST = "<your-db-endpoint>"                # 数据库终端节点
DB_NAME = "<your-db-name>"                    # 数据库名称，默认：postgres
DB_USER = "<your-db-username>"                # 数据库用户名
DB_PASSWORD = "<your-db-password>"            # 数据库密码
DB_PORT = <your-db-port>                      # 数据库端口，默认：5432
```

### 4. 启动应用
```bash
# 方式1：直接启动
python app.py

# 方式2：指定端口启动
export GRADIO_SERVER_PORT=7861
python app.py

# 方式3：后台运行
nohup python app.py > nohup.out 2>&1 &
```

### 5. 访问应用
- 浏览器访问：`http://您的EC2公网IP:7860`
- 确保EC2安全组允许7860端口的入站流量


## 验证部署

1. 检查应用是否正常启动：
   - 看到 "Running on local URL: http://0.0.0.0:7860" 提示
   - gradio 版本提示可以忽略

2. 测试查询功能：
   - 输入关键词（如"发烧"）
   - 输入具体问题
   - 选择查询方法
   - 验证返回结果

## 故障排查

1. 如果遇到依赖问题：
```bash
pip install --upgrade pip
pip uninstall boto3 botocore -y
pip install -r requirements.txt --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. 如果遇到数据库连接问题：
   - 检查数据库连接信息
   - 确认安全组设置
   - 验证数据库用户权限

3. 如果遇到AWS服务问题：
   - 确认AWS凭证配置
   - 检查Bedrock服务权限
   - 验证模型ARN是否正确

4. 如果遇到 "Internal Server Error" 或 JSON 解析错误：
   - 检查 AWS Bedrock 服务状态
   - 验证模型 ARN 是否正确
   - 确认 AWS 凭证权限
   - 检查网络连接
   - 查看详细错误日志：
   ```bash
   tail -f nohup.out
   ```

5. 如果遇到端口占用问题：
   ```bash
   # 查看端口占用
   sudo lsof -i :7860
   
   # 关闭占用进程
   sudo kill -9 $(sudo lsof -t -i:7860)
   
   # 或者使用其他端口
   export GRADIO_SERVER_PORT=7861
   python app.py
   ```

6. 如果遇到数据库连接错误：
   ```bash
   # 检查数据库连接
   psql -h ${DB_HOST} -U ${DB_USER} -d ${DB_NAME}
   
   # 检查网络连接
   ping ${DB_HOST}
   ```

7. 数据库连接优化建议：
   - 使用连接池管理连接
   - 实现自动重连机制
   - 设置合适的超时时间
   - 定期检查连接状态

## 注意事项

1. 保持虚拟环境激活：
```bash
source venv/bin/activate
```

2. 定期检查日志：
```bash
tail -f nohup.out  # 如果使用nohup运行
```

3. 后台运行（可选）：
```bash
nohup python app.py > nohup.out 2>&1 &
```

## 维护建议

1. 定期更新依赖：
```bash
pip install --upgrade -r requirements.txt
```

2. 监控系统资源：
```bash
top
df -h
free -m
```

3. 备份配置文件：
```bash
cp config.py config.py.backup
```

## 使用说明

### 查询流程
1. 输入关键词（如"发烧"）
2. 等待系统返回相关记录数量
3. 输入具体问题（如"发烧超过39度该怎么处理？"）
4. 选择查询方法
5. 获取系统回答

### 查询方法说明

1. **Nova + Cohere方法**
   - 适用场景：需要高精度答案
   - 实现文件：nova-cohere-new.py
   - 特点：结合Nova的生成能力和Cohere的重排序

2. **Titan + Nova方法**
   - 适用场景：快速问答
   - 实现文件：nova-Titan-new.py
   - 特点：使用Titan进行向量检索

3. **Deepseek + Cohere方法**
   - 适用场景：复杂医学问题
   - 特点：结合Deepseek的专业知识和Cohere的优化

### 示例查询

```python
# 示例1：发烧相关
关键词：发烧
问题：发烧超过39度该怎么处理？
推荐方法：Nova + Cohere

# 示例2：感冒相关
关键词：感冒
问题：感冒初期有什么症状？
推荐方法：Titan + Nova

# 示例3：腹泻相关
关键词：腹泻
问题：急性腹泻的治疗方法是什么？
推荐方法：Deepseek + Cohere
```

## 维护指南

### 日常维护
- 定期备份数据库
- 监控系统资源
- 更新依赖包
- 检查日志文件

### 性能优化
- 调整查询限制（当前1000条）
- 优化向量检索参数
- 监控响应时间
- 优化数据库查询

### 安全维护
- 更新安全补丁
- 轮换访问密钥
- 监控访问日志
- 定期安全审计

## 依赖说明

requirements.txt 内容：
```
# Web应用框架
gradio==3.50.2
fastapi==0.95.2
pydantic==1.10.13

# AWS服务
boto3>=1.34.0
botocore>=1.34.0

# 数据库
psycopg2-binary>=2.9.9
DBUtils==1.2

# 向量计算和机器学习
numpy>=2.0.2
faiss-cpu>=1.7.4

# NLP工具
jieba>=0.42.1
spacy_pkuseg>=1.0.0

# 工具库
catalogue>=2.0.10
srsly>=2.4.8
portalocker>=2.8.2
pytz>=2022.6
typing_extensions>=4.12.0

# 日志和监控
logging
```

依赖说明：
1. Web框架
   - gradio: 用于构建Web界面
   - fastapi: 提供API服务
   - pydantic: 数据验证

2. AWS服务
   - boto3/botocore: AWS SDK，用于调用Bedrock等服务

3. 数据库
   - psycopg2-binary: PostgreSQL数据库驱动
   - DBUtils: 数据库连接池管理

4. 向量计算
   - numpy: 数值计算库
   - faiss-cpu: 向量检索库

5. NLP工具
   - jieba: 中文分词
   - spacy_pkuseg: 分词工具

6. 工具库
   - catalogue: 配置管理
   - srsly: 序列化工具
   - portalocker: 文件锁
   - pytz: 时区处理
   - typing_extensions: 类型注解

7. 日志
   - logging: 标准日志库

版本说明：
- 使用 >= 表示最低版本要求
- 特定版本(==)表示必须使用该版本
- 建议使用虚拟环境管理依赖