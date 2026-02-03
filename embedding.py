"""Utilities for preparing text and writing embeddings to ChromaDB."""
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import chromadb
from bs4 import BeautifulSoup
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
import dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
import tiktoken

dotenv.load_dotenv()

PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "articles"
DEFAULT_MODEL = "text-embedding-ada-002"
AZURE_BATCH_LIMIT = 16


# ---------------------------------------------------------------------------
# Chroma helpers
# ---------------------------------------------------------------------------
class ChromaDBManager:
    """Small convenience wrapper for Chroma client/collection lifecycle."""

    def __init__(self, persist_dir: Path = PERSIST_DIR, collection_name: str = COLLECTION_NAME):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.client = self._init_client()
        self.collection = self._init_collection()

    def _init_client(self) -> ClientAPI:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self.persist_dir))

    def _init_collection(self) -> Collection:
        return self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Store vectors and optional metadata/documents in the managed collection."""
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query_by_embedding(self, query_embedding: List[float], top_k: int = 5, where: Optional[Dict] = None) -> Dict[str, Any]:
        """Query the collection using a vector embedding."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where if where else None
        )

    def delete_all(self) -> None:
        """Remove every item in the managed collection."""
        self.collection.delete(where={})

    def count(self) -> int:
        """Return number of items currently stored."""
        return self.collection.count()

    def existing_ids(self, ids: List[str]) -> Set[str]:
        """Return ids that already exist in the collection."""
        if not ids:
            return set()
        result = self.collection.get(ids=ids, include=[])
        return set(result.get("ids", []))


# ---------------------------------------------------------------------------
# Budget utilities
# ---------------------------------------------------------------------------
class BudgetManager:
    """Simple budget guard for embedding batches."""

    def __init__(self, model_name: str = DEFAULT_MODEL, price_per_1k_tokens: float = 0.0001, budget_usd: float = 1.0):
        self.model_name = model_name
        self.price_rate = price_per_1k_tokens
        self.budget = budget_usd
        self.encoder = tiktoken.encoding_for_model(model_name)

    def calculate_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def check_budget(self, texts: Sequence[str]) -> tuple[int, float]:
        total_tokens = sum(self.calculate_tokens(t) for t in texts)
        estimated_cost = (total_tokens / 1000) * self.price_rate

        print("--- 預算檢查 ---")
        print(f"預計消耗 Tokens: {total_tokens}")
        print(f"預計花費: ${estimated_cost:.6f} USD")
        print(f"剩餘預算: ${self.budget:.6f} USD")

        if estimated_cost > self.budget:
            raise ValueError(f"預算不足！預計花費 ${estimated_cost:.6f} > 預算 ${self.budget:.6f}")

        return total_tokens, estimated_cost


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
def clean_html_to_markdown(raw_html: str) -> str:
    """Convert HTML into a Markdown-ish, embedding-friendly string."""
    decoded_html = html.unescape(raw_html)
    soup = BeautifulSoup(decoded_html, "html.parser")

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            rows.append("| " + " | ".join(cols) + " |")
        markdown_table = "\n" + "\n".join(rows) + "\n"
        table.replace_with(markdown_table)

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_text = "\n".join(lines)
    cleaned_text = re.split(r"原始連結|看更多|延伸閱讀", cleaned_text)[0]
    return cleaned_text

def _batch(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# Embedding pipeline
# ---------------------------------------------------------------------------
def run_embedding_pipeline(
    documents: Sequence[Dict[str, Any]],
    azure_client: AzureOpenAI,
    budget_manager: BudgetManager,
    model_name: str = DEFAULT_MODEL,
) -> Optional[List[Dict[str, Any]]]:
    processed_texts: List[str] = [doc["text"] for doc in documents]
    metadata_list: List[Dict[str, Any]] = [doc.get("metadata", {}) for doc in documents]
    ids: List[str] = [str(doc.get("id", idx)) for idx, doc in enumerate(documents)]

    print(">>> 2. 進行預算評估...")
    try:
        budget_manager.check_budget(processed_texts)
    except ValueError as exc:  # keep going gracefully
        print(exc)
        return None

    print(">>> 3. 呼叫 Azure API 進行 Embedding...")
    try:
        embeddings: List[List[float]] = []
        for batch in _batch(processed_texts, AZURE_BATCH_LIMIT):
            response = azure_client.embeddings.create(input=batch, model=model_name)
            embeddings.extend([data.embedding for data in response.data])

        print(f"成功生成 {len(embeddings)} 筆向量資料")

        return [
            {"id": ids[idx], "text": text, "metadata": metadata_list[idx], "vector": embeddings[idx]}
            for idx, text in enumerate(processed_texts)
        ]
    except Exception as exc:  # surface API errors for debugging
        print(f"API 呼叫失敗: {exc}")
        return None

def prepare_texts_with_splitter(
    json_file: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    with open(json_file, "r", encoding="utf-8") as fp:
        articles = json.load(fp)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
        or ["\n\n", "\n", "。", "！", "？", " ", ""],
    )

    final_documents: List[Dict[str, Any]] = []
    for art in articles:
        clean_body = clean_html_to_markdown(art.get("content", ""))
        publish_date = art.get("date_publish", "未知時間")
        combined_text = f"標題: {art['title']}\n內容: {clean_body}\n發布時間: {publish_date}"

        chunks = splitter.split_text(combined_text)
        for idx, chunk_text in enumerate(chunks):
            metadata = {
                "title": art["title"],
                "date": art.get("date_publish"),
                "url": art["url"],
                "chunk_index": idx,
            }
            if "keywords" in art and isinstance(art["keywords"], list):
                metadata["keywords"] = ", ".join(art["keywords"])
            final_documents.append(
                {
                    "id": f"{art['url']}_{idx}",
                    "text": chunk_text,
                    "metadata": metadata,
                }
            )

    return final_documents


def build_azure_client_from_env() -> AzureOpenAI:
    api_key = dotenv.get_key(dotenv.find_dotenv(), "API_KEY")
    endpoint = dotenv.get_key(dotenv.find_dotenv(), "AZURE_ENDPOINT")
    api_version = dotenv.get_key(dotenv.find_dotenv(), "AZURE_API_VERSION")
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

def query(Azure_client: AzureOpenAI, ChromaDB: ChromaDBManager, query_text: str, top_k: int = 5) -> Dict[str, Any]:
    """Helper to get embedding for a query text."""
    response = Azure_client.embeddings.create(input=[query_text], model=DEFAULT_MODEL)
    query_embedding = response.data[0].embedding
    return ChromaDB.query_by_embedding(query_embedding=query_embedding, top_k=top_k)


if __name__ == "__main__":
    action = int(input("請選擇操作: 1. 建立/更新資料庫 2. 查詢資料庫 (輸入數字1或2): ").strip())
    stockNewsChromaDB = ChromaDBManager(collection_name="stock_news", persist_dir=Path("db/chroma_db"))
    client = build_azure_client_from_env()
    if action == 1:
        json_path = "spyder/article.json"
        documents = prepare_texts_with_splitter(json_path)
        doc_ids = [doc["id"] for doc in documents]
        existing = stockNewsChromaDB.existing_ids(doc_ids)
        new_documents = [doc for doc in documents if doc["id"] not in existing]

        if existing:
            print(f"跳過 {len(existing)} 筆已存在的資料，未送往 embedding。")

        if not new_documents:
            print("沒有新資料需要寫入")
        else:
            results = run_embedding_pipeline(new_documents, client, BudgetManager())
            if results:
                ids = [item["id"] for item in results]
                vectors = [item["vector"] for item in results]
                texts = [item["text"] for item in results]
                metas = [item["metadata"] for item in results]
                print(f"正在批次寫入 {len(ids)} 筆新資料...")
                stockNewsChromaDB.upsert_embeddings(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
            else:
                print("Embedding 失敗或未產生結果。")
        print(f"資料庫目前有 {stockNewsChromaDB.count()} 筆資料。")
    elif action == 2:
        user_query = input("請輸入查詢內容: ").strip()
        print(f"正在查詢與 '{user_query}' 相關的文章...")
        query_result = query(client, stockNewsChromaDB, user_query, top_k=5)
        print("查詢結果:")
        print(query_result)