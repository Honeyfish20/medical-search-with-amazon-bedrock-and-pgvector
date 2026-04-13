# Medical Knowledge Query System

A medical knowledge query system based on AWS Bedrock, implementing intelligent medical knowledge Q&A through multi-model combination. This solution is specifically designed for scenarios where data and applications are deployed in AWS China Regions while leveraging Bedrock services from AWS Global Regions.

## Architecture Overview

The system employs a hybrid cloud architecture:
- Data Storage: Amazon Aurora PostgreSQL in AWS China Regions
- Application Deployment: EC2 instances in AWS China Regions
- AI Model Services: Amazon Bedrock services in AWS Global Regions (e.g., us-west-2)
- Network Connectivity: Cross-region access configuration for China Region applications to access Global Region Bedrock services

This architecture design enables:
- Compliance with data localization requirements
- Utilization of AWS Global Region AI capabilities
- Low-latency user access experience
- Control of cross-region data transfer costs

## Project Structure

```
medical_chatbot/
├── app.py                # Gradio Web Application
├── config.py            # Configuration File
├── requirements.txt     # Project Dependencies
├── words_embedding.py   # Document Vector Generation Program
└── README_EN.md           # Documentation
```

## Features

- Keyword-based pre-filtering
- Multi-model combination query (Nova, Titan, Cohere, Deepseek)
- Vector similarity matching
- Real-time Q&A interaction
- PostgreSQL vector database support

## Requirements

- Python 3.9+
- AWS Account (with Bedrock access)
- PostgreSQL Database (AWS RDS)
- EC2 Instance (minimum 2GB RAM recommended)

## Database Configuration Guide

### 1. Deploy Aurora PostgreSQL Database

- Choose Aurora version supporting pgvector (15.3, 14.8, 13.11, 12.15 or higher)
- This project uses Aurora PostgreSQL 13.12
- Follow AWS documentation: [Create Aurora PostgreSQL Database Cluster](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/Aurora.CreateInstance.html)

### 2. Aurora Database Configuration

1. Modify Aurora DB instance parameter group:
```sql
maintenance_work_mem = 10240000 (10GB)
```

2. Restart database cluster to apply changes

3. Verify parameter settings:
```sql
test=> show maintenance_work_mem;
 maintenance_work_mem
----------------------
 10000MB
```

### 3. Enable pgvector Extension

1. Connect to database and execute:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

2. Verify installation:
```sql
\dx vector
```

### 4. Create Tables and Import Medical Data

1. Download test data:
```bash
git clone https://github.com/zhangsheng93/cMedQA2.git
unzip answer.zip
```

2. Create table:
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

3. Import data:
```sql
\copy text_embedding_cos(id, doc_type, doc) from '/home/centos/answer.csv' WITH DELIMITER ',' CSV HEADER;
```

### 5. Generate Document Embeddings

1. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

2. Generate embeddings:
```bash
python words_embedding.py -m embedding -r 240000
```

3. Create vector index:
```sql
CREATE INDEX ON text_embedding_cos 
USING ivfflat (embedding_doc vector_cosine_ops) 
WITH (lists = 10000);
```

4. Create full-text search index:
```sql
CREATE INDEX idx_cos_gin_keywords 
ON text_embedding_cos 
USING GIN(to_tsvector('simple', keywords));
```

5. Create Euclidean distance index:
```sql
CREATE INDEX ON text_embedding USING ivfflat (embedding_doc vector_l2_ops) WITH(lists = 10000);
```

Notes:
- Embedding generation is time-consuming (about 2 hours/150k documents)
- Test with small dataset first
- Ensure sufficient database storage
- Regular vector data backup
- Monitor index creation process

## Quick Deployment Guide

### 1. Environment Setup
```bash
# Update system and install Python
sudo yum update -y
sudo yum install python3 python3-pip -y

# Create project directory
mkdir medical_chatbot
cd medical_chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
# Uninstall old versions if exist
pip uninstall boto3 botocore -y

# Install dependencies
pip install -r requirements.txt --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple

### Dependency Installation Solution
pydantic.errors.PydanticSchemaGenerationError: Unable to generate pydantic-core schema for <class 'starlette.requests.Request'>

rm -rf venv
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install key dependencies in order
pip install pydantic==1.10.13 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install fastapi==0.95.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install gradio==3.50.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. Install other dependencies
pip install -r requirements.txt --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. Configuration File
Ensure config.py contains correct settings:
```python
# AWS Configuration
AWS_REGION = "<your-aws-region>"              # AWS Region, e.g., us-west-2
AWS_ACCESS_KEY_ID = "<your-access-key>"       # AWS Access Key ID
AWS_SECRET_ACCESS_KEY = "<your-secret-key>"   # AWS Secret Access Key

# Cohere Configuration
COHERE_API_KEY = "cohere.rerank-v3-5:0"

# Model Configuration
NOVA_MODEL_ID = "<your-nova-model-arn>"       # Nova Model ARN
TITAN_MODEL_ID = "<your-titan-model-id>"      # Titan Model ID
DEEPSEEK_MODEL_ID = "<your-deepseek-model-arn>" # Deepseek Model ARN

# Database Configuration
DB_HOST = "<your-db-endpoint>"                # Database Endpoint
DB_NAME = "<your-db-name>"                    # Database Name, default: postgres
DB_USER = "<your-db-username>"                # Database Username
DB_PASSWORD = "<your-db-password>"            # Database Password
DB_PORT = <your-db-port>                      # Database Port, default: 5432
```

### 4. Launch Application
```bash
# Method 1: Direct launch
python app.py

# Method 2: Launch with specific port
export GRADIO_SERVER_PORT=7861
python app.py

# Method 3: Run in background
nohup python app.py > nohup.out 2>&1 &
```

### 5. Access Application
- Browser access: `http://your-EC2-public-IP:7860`
- Ensure EC2 security group allows inbound traffic on port 7860

## Deployment Verification

1. Check if application starts normally:
   - Look for "Running on local URL: http://0.0.0.0:7860" message
   - gradio version notice can be ignored

2. Test query functionality:
   - Enter keyword (e.g., "fever")
   - Enter specific question
   - Select query method
   - Verify returned results

## Troubleshooting

1. For dependency issues:
```bash
pip install --upgrade pip
pip uninstall boto3 botocore -y
pip install -r requirements.txt --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. For database connection issues:
   - Check database connection information
   - Confirm security group settings
   - Verify database user permissions

3. For AWS service issues:
   - Confirm AWS credential configuration
   - Check Bedrock service permissions
   - Verify model ARN correctness

4. For "Internal Server Error" or JSON parsing errors:
   - Check AWS Bedrock service status
   - Verify model ARN correctness
   - Confirm AWS credential permissions
   - Check network connection
   - View detailed error logs:
   ```bash
   tail -f nohup.out
   ```

5. For port occupation issues:
   ```bash
   # Check port occupation
   sudo lsof -i :7860
   
   # Kill occupying process
   sudo kill -9 $(sudo lsof -t -i:7860)
   
   # Or use different port
   export GRADIO_SERVER_PORT=7861
   python app.py
   ```

6. For database connection errors:
   ```bash
   # Check database connection
   psql -h ${DB_HOST} -U ${DB_USER} -d ${DB_NAME}
   
   # Check network connection
   ping ${DB_HOST}
   ```

7. Database connection optimization suggestions:
   - Use connection pool management
   - Implement auto-reconnection mechanism
   - Set appropriate timeout values
   - Regular connection status checks

## Notes

1. Keep virtual environment activated:
```bash
source venv/bin/activate
```

2. Regular log checks:
```bash
tail -f nohup.out  # if using nohup
```

3. Background running (optional):
```bash
nohup python app.py > nohup.out 2>&1 &
```

## Maintenance Suggestions

1. Regular dependency updates:
```bash
pip install --upgrade -r requirements.txt
```

2. Monitor system resources:
```bash
top
df -h
free -m
```

3. Backup configuration files:
```bash
cp config.py config.py.backup
```

## Usage Guide

### Query Process
1. Enter keyword (e.g., "fever")
2. Wait for system to return number of relevant records
3. Enter specific question (e.g., "How to handle fever above 39 degrees?")
4. Select query method
5. Get system response

### Query Methods Explanation

1. **Nova + Cohere Method**
   - Use case: Need high-precision answers
   - Implementation file: nova-cohere-new.py
   - Features: Combines Nova's generation capability with Cohere's reranking

2. **Titan + Nova Method**
   - Use case: Quick Q&A
   - Implementation file: nova-Titan-new.py
   - Features: Uses Titan for vector retrieval

3. **Deepseek + Cohere Method**
   - Use case: Complex medical questions
   - Features: Combines Deepseek's professional knowledge with Cohere's optimization

### Example Queries

```python
# Example 1: Fever related
Keyword: fever
Question: How to handle fever above 39 degrees?
Recommended method: Nova + Cohere

# Example 2: Cold related
Keyword: cold
Question: What are the early symptoms of a cold?
Recommended method: Titan + Nova

# Example 3: Diarrhea related
Keyword: diarrhea
Question: What are the treatment methods for acute diarrhea?
Recommended method: Deepseek + Cohere
```

## Maintenance Guide

### Daily Maintenance
- Regular database backup
- Monitor system resources
- Update dependencies
- Check log files

### Performance Optimization
- Adjust query limits (currently 1000)
- Optimize vector retrieval parameters
- Monitor response time
- Optimize database queries

### Security Maintenance
- Update security patches
- Rotate access keys
- Monitor access logs
- Regular security audits

## Dependencies

requirements.txt contents:
```
# Web Framework
gradio==3.50.2
fastapi==0.95.2
pydantic==1.10.13

# AWS Services
boto3>=1.34.0
botocore>=1.34.0

# Database
psycopg2-binary>=2.9.9
DBUtils==1.2

# Vector Computation and Machine Learning
numpy>=2.0.2
faiss-cpu>=1.7.4

# NLP Tools
jieba>=0.42.1
spacy_pkuseg>=1.0.0

# Utility Libraries
catalogue>=2.0.10
srsly>=2.4.8
portalocker>=2.8.2
pytz>=2022.6
typing_extensions>=4.12.0

# Logging
logging
```

Dependencies Explanation:
1. Web Framework
   - gradio: For building web interface
   - fastapi: For API service
   - pydantic: For data validation

2. AWS Services
   - boto3/botocore: AWS SDK for calling Bedrock services

3. Database
   - psycopg2-binary: PostgreSQL database driver
   - DBUtils: Database connection pool management

4. Vector Computation
   - numpy: Numerical computation library
   - faiss-cpu: Vector retrieval library

5. NLP Tools
   - jieba: Chinese word segmentation
   - spacy_pkuseg: Segmentation tool

6. Utility Libraries
   - catalogue: Configuration management
   - srsly: Serialization tool
   - portalocker: File lock
   - pytz: Timezone handling
   - typing_extensions: Type annotations

7. Logging
   - logging: Standard logging library

Version Notes:
- >= indicates minimum version requirement
- == indicates exact version requirement
- Virtual environment recommended for dependency management 