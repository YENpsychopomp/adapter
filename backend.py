import os
import json
import time
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from pydantic import BaseModel
from pathlib import Path
import uvicorn

# 匯入你原本的工具
from embedding import ChromaDBManager, build_azure_client_from_env, prepare_texts_with_splitter, run_embedding_pipeline

app = FastAPI()

# --- 初始化 ---
azure_client = build_azure_client_from_env()
db_manager = ChromaDBManager(collection_name="stock_news", persist_dir=Path("db/chroma_db"))

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    user_input = request.messages[-1]["content"]
    
    # 1. RAG 判斷與檢索
    need_rag = "?" in user_input or "？" in user_input or len(user_input) < 100
    context = ""
    if need_rag:
        print(f"[RAG] 檢索中: {user_input[:20]}...")
        emb_res = azure_client.embeddings.create(input=[user_input], model="text-embedding-ada-002")
        q_emb = emb_res.data[0].embedding
        search_res = db_manager.query_by_embedding(query_embedding=q_emb, top_k=3)
        if search_res['documents'] and search_res['documents'][0]:
            print(f"[RAG] 找到相關文件，一共 {len(search_res['documents'][0])} 筆。")
            context = "\n".join(search_res['documents'][0])

    # 2. 重新建構發送給 Azure 的 Messages
    system_instruction = """你是一個專業財經助手。
    1. 參考背景資料回答問題。
    2. 若輸入具備財經價值，請在結尾標註 [SAVE_START] 與 [SAVE_END] 夾帶 JSON 格式。
    3. 回應始終保持中文，除非使用者特別要求其他語言。
    JSON 格式：{"title": "...", "content": "...", "date_publish": "YYYY-MM-DD", "url": "..."}
    """
    
    azure_messages = [{"role": "system", "content": system_instruction}]
    for msg in request.messages[:-1]:
        azure_messages.append(msg)
        
    # 加入當前輸入與 Context
    azure_messages.append({
        "role": "user", 
        "content": f"【背景資料】：\n{context}\n\n【使用者輸入】：{user_input}"
    })

    # 3. 呼叫 Azure LLM
    response = azure_client.chat.completions.create(
        model="gpt-4o", # 這裡填你的部署名稱
        messages=azure_messages,
        temperature=request.temperature
    )
    
    ai_reply = response.choices[0].message.content

    # 4. 自動入庫判斷 (背景執行)
    if "[SAVE_START]" in ai_reply:
        process_auto_save(ai_reply)
        # 移除 JSON 標記後再回傳給前端
        ai_reply = ai_reply.split("[SAVE_START]")[0].strip()

    # 5. 回傳符合 OpenAI 格式的 Response
    return {
        "id": "chatcmpl-" + str(time.time()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": ai_reply},
            "finish_reason": "stop"
        }],
        "usage": response.usage.model_dump()
    }

@app.get("/v1/models")
async def list_models():
    # 回傳一個簡單的模型列表
    return {
        "object": "list",
        "data": [
            {
                "id": "financial-rag-assistant",  # 這是你會在選單看到的名稱
                "object": "model",
                "created": int(time.time()),
                "owned_by": "custom-proxy"
            }
        ]
    }

def process_auto_save(ai_reply: str):
    """提取 JSON 並存入 ChromaDB"""
    try:
        raw_json = ai_reply.split("[SAVE_START]")[1].split("[SAVE_END]")[0].strip()
        data = json.loads(raw_json)
        print(f"✨ [自動入庫] 標題: {data.get('title')}")
        
        # 這裡可以根據你之前 embedding.py 的邏輯存入
        # 為了簡化，你可以直接用 db_manager.collection.add (...)
        # 或調用你原本寫好的批次處理函數
    except Exception as e:
        print(f"⚠️ 入庫失敗: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9898)