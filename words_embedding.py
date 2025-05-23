# -*- coding: utf-8 -*- 
'''
# Generate and print an embedding with Amazon Titan Text Embeddings V2.
# load data: COPY temp_doc (id, doc_type, doc) FROM 'answer.csv' DELIMITER ',' CSV HEADER;
# RDS:  \copy temp_doc(id, doc_type, doc) from '/home/centos/zhcn_search/answer.csv' WITH DELIMITER ',' CSV HEADER;
# download test data: git clone https://github.com/zhangsheng93/cMedQA2.git
'''

import sys
import boto3
import json
import psycopg2.extras
from DBUtils.PooledDB import PooledDB
import threading
from typing import Dict, List
import argparse
import datetime
import pytz
import math
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

tz = pytz.timezone('Asia/Shanghai')

host = os.getenv("DB_HOST")
port = int(os.getenv("DB_PORT", "5432"))
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
dbname = os.getenv("DB_NAME")

def args_parse():
    parser = argparse.ArgumentParser(description='search test by vector')
    parser.add_argument('--mode', '-m', help='embedding: update embedding, search: search a keyword, mandatory', required=True, default='search')
    parser.add_argument('--probes', '-p', help='probes for vectors search, optional', required=False, default=10)
    parser.add_argument('--topk', '-t', help='topk', required=False, default=2)
    parser.add_argument('--input', '-i', help='word to search', required=False)
    parser.add_argument('--maxId', '-r', help='rows to embedding, default 226272, which is same with test data', required=False)
    args = parser.parse_args()
    return args

class PsycopgConn:

    _instance_lock = threading.Lock()

    def __init__(self):
        self.init_pool()

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            with PsycopgConn._instance_lock:
                if not hasattr(cls, '_instance'):
                    PsycopgConn._instance = object.__new__(cls)
        return PsycopgConn._instance
		
    def get_pool_conn(self):
        """
        get conn from pool
        :return: 
        """
        if not self._pool:
            self.init_pool()
        return self._pool.connection()

    def init_pool(self):
        """
        init pool
        :return: 
        """
        try:
            pool = PooledDB(
                creator=psycopg2,
                maxconnections=50,
                mincached=2,
                maxcached=20,
                blocking=True,
                maxusage=None,
                setsession=[],
                host=host,
                port=port,
                user=user,
                password=password,
                database=dbname)
            self._pool = pool
        except:
            print ('connect postgresql error when init pool')
            self.close_pool()

    def close_pool(self):
        """
        close pool
        :return: 
        """
        if self._pool != None:
            self._pool.close()


    def SelectSql(self,sql,vars=None):
        """
        query
        :param sql: SQL查询
        :param vars: 查询参数
        :return: 
        """
        try:
            conn = self.get_pool_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            if vars:
                cursor.execute(sql, vars)
            else:
                cursor.execute(sql)
            result = cursor.fetchall()
        except Exception as e:
            print('execute sql {0} is error'.format(sql))
            sys.exit('ERROR: query data from database error caused {0}'.format(str(e)))
        finally:
            cursor.close()
            conn.commit()
            conn.close()
        return result
    
    def SelectSqlWithInitSql(self,sql, vars=None, initSQL=None, init_vars=None):
        """
        query
        :param sql: 
        :param vars: 查询参数
        :param initSQL: 初始化SQL
        :param init_vars: 初始化SQL的参数
        :return: 
        """
        try:
            conn = self.get_pool_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if init_vars:
                cursor.execute(initSQL, init_vars)
            else:
                cursor.execute(initSQL)
                
            if vars:
                cursor.execute(sql, vars)
            else:
                cursor.execute(sql)
                
            result = cursor.fetchall()
        except Exception as e:
            print('execute sql {0} is error'.format(sql))
            sys.exit('ERROR: query data with init sql from database error caused {0}'.format(str(e)))
        finally:
            cursor.close()
            conn.commit()
            conn.close()
        return result
    
    def InsertSql(self,sql):
        """
        insert data
        :param sql: 
        :return: 
        """
        try:
            conn = self.get_pool_conn()
            cursor = conn.cursor()
            cursor.execute(sql)
            result = True
        except Exception as e:
            print('ERROR: execute  {0} causes error'.format(sql))
            sys.exit('ERROR: insert data from database error caused {0}'.format(str(e)))
        finally:
            cursor.close()
            conn.commit()
            conn.close()
        return result

    def UpdateSql(self,sql,vars=None):
        """
        update
        :param sql: 
        :return: 
        """
        try:
            conn = self.get_pool_conn()
            cursor = conn.cursor()
            cursor.execute(sql, vars)
            result = True
        except Exception as e:
            print('ERROR: execute  {0} causes error'.format(sql))
            sys.exit('ERROR: update data from database error caused {0}'.format(str(e)))
        finally:
            cursor.close()
            conn.commit()
            conn.close()
        return result

def now_time():
    for_now = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    return for_now

def make_print_to_file(path='./', prefix='log_'):
    '''
    path, it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            if not os.path.exists(path):
                os.makedirs(path)
            self.filename = filename
            self.path = path
            self.log_path = os.path.join(path, filename)
 
        def write(self, message):
            self.terminal.write(message)
            # 每次写入时打开并关闭文件，确保资源正确释放
            with open(self.log_path, "a", encoding='utf8') as f:
                f.write(message)
 
        def flush(self):
            pass  # 无需刷新，因为每次写入后文件已关闭
                
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass  # 文件在每次写入后都已关闭，无需额外处理
    
    fileName = prefix + datetime.datetime.now(tz).strftime("%Y%m%d%H%M")
    logger = Logger(fileName + '.log', path=path)
    sys.stdout = logger
    print(fileName.center(60,'*'))  # 添加回原先的日志文件标题行
    return logger

# query abstract
def queryAbstracts(pool, tableName, minId, maxId):
    sql = 'select id, doc, embedding_doc from {} where id between %s and %s'
    sql = sql.format(tableName)  # 表名无法参数化，但至少要验证
    return pool.SelectSql(sql, (minId, maxId))

def updateEmbeddingById(pool, id, embedding: List):
    sql = "update text_embedding set embedding_doc=%s::vector(1536) where id = %s;"
    return pool.UpdateSql(sql, (str(embedding), id))

def embedding_titan(input_text: str):
    # Create the request for the model.
    native_request = {"inputText": input_text}
    
    # Convert the native request to JSON.
    request = json.dumps(native_request)
    
    # Invoke the model with the request.
    response = client.invoke_model(modelId=model_id, body=request)
    
    # Decode the model's native response body.
    model_response = json.loads(response["body"].read())
    
    # Extract and print the generated embedding and the input text token count.
    embedding = model_response["embedding"]
    input_token_count = model_response["inputTextTokenCount"]
    return embedding, input_token_count

# batch update the embedding column in table
def batchUpdateEmbedding(pool, maxId: int):
    rows = queryAbstracts(pool, 'text_embedding', 1, maxId)
    for row in rows:
        doc = dict(row)
        doc_text = doc["doc"]
        id = doc["id"]
        embedding, input_token_count = embedding_titan(doc_text)
        updateEmbeddingById(pool, id, embedding)
        print("doc: %s, token_count: %d " % (doc["doc"], input_token_count))
    pool.close_pool()

# search records by pg vector l2 distance
def searchByWord(input_word: str, pool, probes: int, topk: int):
    word_embedding, count = embedding_titan(input_word)
    initSql = "SET ivfflat.probes = %s"
    sql = "select id, doc, embedding_doc <-> %s::vector(1536) as distance from text_embedding order by embedding_doc <-> %s::vector(1536) limit %s"
    return pool.SelectSqlWithInitSql(sql, (str(word_embedding), str(word_embedding), topk), initSql, (probes,))

def searchRc(input_word: str, pool, probes: int = 10, topk: int = 2):
    start_time = datetime.datetime.now(tz)
    rows = searchByWord(input_word, pool, probes, topk)
    end_time = datetime.datetime.now(tz)
    running_seconds = math.ceil((end_time - start_time).total_seconds())
    print("search result by keyword: %s , search time: %d sec\n" % (input_word, running_seconds))
    for row in rows:
        doc = dict(row)
        print(doc)
    return

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-west-2")
# Set the model ID, e.g., Titan Text Embeddings V2: amazon.titan-embed-text-v2:0
model_id = "amazon.titan-embed-text-v1"

def test_titan():
    # The text to convert to an embedding.
    input_text = "外周神经病变"
    
    # Create the request for the model.
    native_request = {"inputText": input_text}
    
    # Convert the native request to JSON.
    request = json.dumps(native_request)
    
    # Invoke the model with the request.
    response = client.invoke_model(modelId=model_id, body=request)
    
    # Decode the model's native response body.
    model_response = json.loads(response["body"].read())
    
    # Extract and print the generated embedding and the input text token count.
    embedding = model_response["embedding"]
    input_token_count = model_response["inputTextTokenCount"]
    
    print("\nYour input:")
    print(input_text)
    print("Number of input tokens: %d" % input_token_count)
    print("Size of the generated embedding: %d" % len(embedding))
    print("Embedding:")
    print(embedding)
    
if __name__ == "__main__":
    args = args_parse()
    mode = args.mode
    probes= int(args.probes) if args.probes is not None else None
    topk=int(args.topk) if args.topk is not None else None
    input_word = args.input
    maxId = int(args.maxId) if args.maxId is not None else None
    pool = PsycopgConn()
    if mode == "embedding":
        batchUpdateEmbedding(pool, maxId)
    elif mode == "search":
        searchRc(input_word, pool, probes, topk)
    pool.close_pool()
