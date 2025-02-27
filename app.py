import gradio as gr
import psycopg2
import boto3
import json
from config import *
import numpy as np
import time
import os

def create_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=DB_NAME,
            connect_timeout=10
        )
        print("数据库连接成功")
        return conn
    except Exception as e:
        print(f"数据库连接失败: {str(e)}")
        return None

def create_clients():
    try:
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        cohere_client = session.client("bedrock-runtime")
        nova_client = session.client("bedrock-runtime")
        titan_client = session.client("bedrock-runtime")
        deepseek_client = session.client("bedrock-runtime")
        print("AWS客户端创建成功")
        return cohere_client, nova_client, titan_client, deepseek_client
    except Exception as e:
        print(f"AWS客户端创建失败: {str(e)}")
        return None, None, None, None

def search_documents(keyword):
    """根据关键词搜索文档"""
    try:
        conn = create_db_connection()
        if not conn:
            return "数据库连接失败", None
            
        cur = conn.cursor()
        
        query = """
        SELECT id, doc, embedding_doc
        FROM text_embedding
        WHERE doc ILIKE %s
        LIMIT 1000;
        """
        cur.execute(query, [f'%{keyword}%'])
        results = cur.fetchall()
        
        cur.close()
        conn.close()
        
        if not results:
            return "未找到相关记录", None
            
        return f"找到 {len(results)} 条相关记录", results
    except Exception as e:
        return f"数据库查询出错: {str(e)}", None

def rerank_documents(cohere_client, query, documents):
    """使用Cohere重排序文档"""
    try:
        request = {
            "query": query,
            "documents": documents,
            "api_version": 2
        }
        
        response = cohere_client.invoke_model(
            modelId="cohere.rerank-v3-5:0",
            body=json.dumps(request)
        )
        return json.loads(response["body"].read())['results']
    except Exception as e:
        raise Exception(f"Cohere重排序出错: {str(e)}")

def generate_summary(nova_client, content):
    """使用Nova生成总结"""
    try:
        messages = [{
            "role": "user",
            "content": [{"text": f"请根据以下医学相关内容，给出专业的回答：\n\n{content}"}]
        }]
        
        response = nova_client.invoke_model(
            modelId=NOVA_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "inferenceConfig": {"max_new_tokens": 1000},
                "messages": messages
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        return response_body["output"]["message"]["content"][0]["text"]
    except Exception as e:
        raise Exception(f"Nova生成总结出错: {str(e)}")

def get_titan_embedding(titan_client, text):
    """使用Titan模型获取文本嵌入向量"""
    try:
        native_request = {
            "inputText": text
        }
        
        request = json.dumps(native_request)
        
        response = titan_client.invoke_model(
            modelId=TITAN_MODEL_ID,
            body=request
        )
        
        model_response = json.loads(response["body"].read())
        return model_response["embedding"]
    except Exception as e:
        raise Exception(f"Titan嵌入向量生成失败: {str(e)}")

def calculate_similarity(query_embedding, doc_embeddings):
    """计算欧几里得距离并返回排序后的索引"""
    try:
        query_embedding = np.array(query_embedding)
        
        # 过滤掉None值，同时保存原始索引
        valid_embeddings = []
        valid_indices = []
        for idx, emb in enumerate(doc_embeddings):
            if emb is not None:
                try:
                    # 转换字符串为列表
                    embedding = json.loads(emb.replace("'", "\""))
                    valid_embeddings.append(embedding)
                    valid_indices.append(idx)
                except ValueError as e:
                    logger.warning(f"处理索引时出错: {e}")
                    continue
        
        if not valid_embeddings:
            raise Exception("没有有效的嵌入向量可供比较")
            
        valid_embeddings = np.array(valid_embeddings)
        
        # 计算欧几里得距离
        distances = np.sqrt(np.sum((valid_embeddings - query_embedding) ** 2, axis=1))
        
        # 获取排序后的索引
        sorted_local_indices = np.argsort(distances)
        
        # 将局部索引转换回原始索引
        sorted_global_indices = [valid_indices[i] for i in sorted_local_indices]
        
        return sorted_global_indices
    except Exception as e:
        raise Exception(f"相似度计算失败: {str(e)}")

def generate_deepseek_response(deepseek_client, content, question, max_retries=5):
    """使用Deepseek生成结构化回答"""
    try:
        # 格式化输入内容
        content_list = content.split('\n\n')
        unique_content = []
        seen = set()
        for doc in content_list:
            doc = doc.strip()
            if doc and doc not in seen:
                seen.add(doc)
                unique_content.append(doc)
                if len(unique_content) >= 5:
                    break
                    
        formatted_content = "\n\n".join(unique_content)
        
        # 优化 prompt 结构
        structured_prompt = f"""请作为一位专业的医生，根据以下参考资料，对"{question}"进行专业的分析和总结。
请严格按照以下格式输出，每个部分必须详细回答：

### 1. 发病原因
[此处分析可能的致病原因]

### 2. 预防措施
[此处说明预防和避免方法]

### 3. 处理方法
[此处描述治疗和处理方案]

### 4. 医疗建议
[此处提供就医建议和注意事项]

### 5. 特别注意事项
[此处列出需要特别关注的问题]

参考资料：
{formatted_content}

请注意：
1. 必须按照上述格式输出，保留"### "标记
2. 每个部分都必须基于参考资料进行详细回答
3. 如果某方面信息不足，请说明"根据现有资料无法确定"
4. 避免重复内容
"""

        request_body = {
            "prompt": structured_prompt,
            "temperature": 0.3,
            "max_gen_len": 2000,
            "top_p": 0.9
        }
        
        attempt = 0
        while attempt < max_retries:
            try:
                response = deepseek_client.invoke_model(
                    modelId=DEEPSEEK_MODEL_ID,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(request_body, ensure_ascii=False).encode('utf-8')
                )
                
                response_body = json.loads(response['body'].read().decode('utf-8'))
                output_text = response_body.get('generation', '')
                
                if not output_text:
                    print(f"尝试 {attempt + 1}: 未获得有效响应")
                    continue
                
                # 改进输出处理逻辑
                sections = []
                current_section = []
                is_valid_section = False  # 添加标记来判断是否是有效的章节
                
                for line in output_text.split('\n'):
                    # 只处理以 "### 数字." 开头的标题和其内容
                    if line.startswith('### ') and any(f"{i}." in line for i in range(1, 6)):
                        if current_section:
                            sections.append('\n'.join(current_section))
                        current_section = [line]
                        is_valid_section = True
                    elif is_valid_section:
                        current_section.append(line)
                
                if current_section:
                    sections.append('\n'.join(current_section))
                
                # 验证输出完整性
                if len(sections) < 5:
                    print(f"尝试 {attempt + 1}: 输出不完整")
                    continue
                
                formatted_output = "分析结果：\n\n" + '\n\n'.join(sections)
                return formatted_output.strip()
                
            except Exception as e:
                print(f"尝试 {attempt + 1} 失败: {str(e)}")
            
            attempt += 1
            if attempt < max_retries:
                wait_time = min(30, 5 * (2 ** attempt))
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                
        return "生成回答失败，请稍后重试"
            
    except Exception as e:
        print(f"Deepseek处理失败: {str(e)}")
        return f"处理出错: {str(e)}"

def process_query(keyword, question, method="nova_cohere"):
    """处理用户查询"""
    if not keyword or not question:
        return "请输入关键词和问题"
        
    try:
        # 创建客户端
        cohere_client, nova_client, titan_client, deepseek_client = create_clients()
        if not all([cohere_client, nova_client, titan_client, deepseek_client]):
            return "错误: AWS服务连接失败，请检查AWS凭证配置"
        
        # 搜索相关文档
        status, results = search_documents(keyword)
        if not results:
            return f"错误: {status}"
        
        # 提取文档内容和嵌入向量
        documents = [result[1] for result in results]
        embeddings = [result[2] for result in results]
        
        try:
            search_results = "\n相关性最强的前5条记录：\n"
            
            if method == "nova_cohere":
                # Cohere重排序
                reranked_results = rerank_documents(cohere_client, question, documents)
                for result in reranked_results[:5]:
                    search_results += f"\n记录索引：{result['index']}, 相关性得分：{result['relevance_score']:.4f}\n"
                    search_results += f"文档内容：{documents[result['index']]}\n"
                    
                top_docs = "\n\n".join([documents[result['index']] for result in reranked_results[:5]])
                final_answer = generate_summary(nova_client, f"{question}\n\n{top_docs}")
                
            elif method == "nova_titan":
                # Titan嵌入和相似度计算
                query_embedding = get_titan_embedding(titan_client, question)
                sorted_indices = calculate_similarity(query_embedding, embeddings)
                
                for idx in sorted_indices[:5]:
                    distance = np.sqrt(np.sum((np.array(json.loads(embeddings[idx].replace("'", "\""))) - np.array(query_embedding)) ** 2))
                    search_results += f"\n记录索引：{idx}, 距离：{distance:.4f}\n"
                    search_results += f"文档内容：{documents[idx]}\n"
                    
                top_docs = "\n\n".join([documents[idx] for idx in sorted_indices[:5]])
                final_answer = generate_summary(nova_client, f"{question}\n\n{top_docs}")
                
            else:  # deepseek_cohere
                # Cohere重排序
                reranked_results = rerank_documents(cohere_client, question, documents)
                
                # 构建搜索结果，确保显示前5条
                search_results = "\n相关文档检索结果：\n"
                shown_docs = set()
                result_count = 0
                
                for result in reranked_results:
                    doc = documents[result['index']]
                    if doc not in shown_docs and result_count < 5:
                        result_count += 1
                        shown_docs.add(doc)
                        search_results += f"\n{result_count}. 相关性得分：{result['relevance_score']:.4f}\n"
                        search_results += f"   文档内容：{doc}\n"
                
                # 使用去重后的前5条文档
                top_docs = "\n\n".join([doc for doc in shown_docs])
                final_answer = generate_deepseek_response(deepseek_client, top_docs, question)
                
                if not final_answer:
                    raise Exception("未能获得有效的回答")
                
                return f"""检索到的相关文档：
{search_results}
----------------------------------------
{final_answer}"""
            
            # 其他方法保持原有的输出格式
            return f"{search_results}\n\n最终答案：\n{final_answer}"
            
        except Exception as e:
            return f"{method}处理失败: {str(e)}"
            
    except Exception as e:
        return f"处理查询时出错: {str(e)}"

def create_interface():
    with gr.Blocks(title="医学知识查询系统") as demo:
        gr.Markdown("# 医学知识查询系统")
        gr.Markdown("## 支持三种查询方法")
        
        with gr.Row():
            keyword = gr.Textbox(label="请输入关键词（如：发烧、感冒等）")
            question = gr.Textbox(label="请输入您的具体问题")
        
        method = gr.Radio(
            choices=["nova_cohere", "nova_titan", "deepseek_cohere"],
            value="nova_cohere",
            label="选择查询方法",
            info="Nova+Cohere: 更精确的重排序; Nova+Titan: 更快的向量相似度; Deepseek+Cohere: 结构化专业分析"
        )
        
        submit_btn = gr.Button("提交查询")
        output = gr.Textbox(label="查询结果", lines=20)  # 增加显示行数
        
        submit_btn.click(
            fn=process_query,
            inputs=[keyword, question, method],
            outputs=output
        )
    
    return demo

# 启动应用
if __name__ == "__main__":
    demo = create_interface()
    # 获取环境变量中配置的允许访问的IP
    allowed_ip = os.getenv("ALLOWED_HOST", "127.0.0.1")  # 默认只允许本地访问
    demo.launch(
        server_name=allowed_ip,
        server_port=7860,
        auth=(os.getenv("BASIC_AUTH_USER"), os.getenv("BASIC_AUTH_PASS"))  # 基本认证
    ) 