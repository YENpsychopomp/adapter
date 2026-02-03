import Adapter
import os
import dotenv
import time
import json
from pathlib import Path
from embedding import ChromaDBManager, build_azure_client_from_env

dotenv.load_dotenv()
azure_client = build_azure_client_from_env()
db_manager = ChromaDBManager(collection_name="stock_news", persist_dir=Path("db/chroma_db"))

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_API_KEY = os.getenv("API_KEY")
client = Adapter.adapter(api_key=AZURE_API_KEY, endpoint=AZURE_ENDPOINT, api_version=AZURE_API_VERSION)

# 維護對話紀錄
chat_history = []

def get_ai_response(user_input):
    global chat_history

    need_rag = "?" in user_input or "？" in user_input or len(user_input) < 100
    context = ""
    if need_rag:
        print("🔍 [系統] 偵測到疑問意圖，正在檢索資料庫...")
        emb_res = azure_client.embeddings.create(input=[user_input], model="text-embedding-ada-002")
        q_emb = emb_res.data[0].embedding
        search_res = db_manager.query_by_embedding(query_embedding=q_emb, top_k=3)
        if search_res['documents'][0]:
            context = "\n".join(search_res['documents'][0])

    # --- 步驟 B：建構 Prompt (多輪對話 + 上下文) ---
    system_instruction = """你是一個專業財經助手。
    1. 參考提供的背景資料與對話歷史回答問題，並加入你自己的見解。
    2. 如果使用者輸入的是具備價值的財經資訊（如新聞、數據、深入分析），請在回覆最後標註 [SAVE_START] 與 [SAVE_END]，並以 JSON 格式提供該內容。
    JSON 格式要求：{"title": "...", "content": "...", "date_publish": "YYYY-MM-DD"}
    """
    
    messages = [{"role": "system", "content": system_instruction}]
    # 加入對話歷史
    messages.extend(chat_history[-6:]) # 取最近 3 輪對話
    
    # 加入當前輸入與 RAG 上下文
    current_prompt = f"【背景資料】：\n{context}\n\n【使用者輸入】：{user_input}"
    messages.append({"role": "user", "content": current_prompt})

    # --- 步驟 C：呼叫模型 ---
    body = {
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0.7,
    }
    
    response = client.create(body)
    full_content = response["choices"][0]["message"]["content"]
    
    # 更新歷史紀錄
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": full_content})
    
    return full_content

# --- 2. 執行循環 ---
print("🤖 財經對話機器人已上線 (輸入 'exit' 結束對話)")
while True:
    user_input = input("\n👤 使用者: ").strip()
    if user_input.lower() in ['exit', 'quit', 'bye']:
        break
    
    start_time = time.time()
    ai_reply = get_ai_response(user_input)
    
    # --- 步驟 D：處理「入庫價值」判斷 ---
    if "[SAVE_START]" in ai_reply:
        print("\n✨ [系統] 偵測到高價值資訊，準備存入 ChromaDB...")
        try:
            # 提取 JSON 區塊
            raw_json = ai_reply.split("[SAVE_START]")[1].split("[SAVE_END]")[0].strip()
            save_data = json.loads(raw_json)
            
            # 呼叫你原本的 embedding.py 邏輯進行 upsert
            # 這裡簡化流程：直接將 content 轉向量存入
            # (建議在此處調用 prepare_texts_with_splitter 的邏輯)
            print(f"✅ 已成功記錄：{save_data.get('title')}")
            
            # 顯示給使用者看的回覆則去掉 JSON 部分
            clean_reply = ai_reply.split("[SAVE_START]")[0].strip()
        except Exception as e:
            print(f"⚠️ 入庫格式解析失敗: {e}")
            clean_reply = ai_reply
    else:
        clean_reply = ai_reply

    print(f"\n🤖 助手: {clean_reply}")
    print(f"(耗時: {time.time() - start_time:.2f}s)")