import json
import time
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from pydantic import BaseModel
from pathlib import Path
import uvicorn

# 匯入你原本的工具
from embedding import ChromaDBManager, build_azure_client_from_env, query

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
    print(request)
    response = query(azure_client, db_manager, request)
    return response

@app.get("/v1/models")
async def list_models():
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