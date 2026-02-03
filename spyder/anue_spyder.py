import os
import time
import random
import requests
import json
import re
import warnings
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta

def fetch_anue_stock_news(Page: int = 1, limit: int = 30) -> List[Dict]:
    url = f"https://api.cnyes.com/media/api/v1/newslist/category/tw_stock?page={Page}&limit={limit}&isCategoryHeadline=0"
    response = requests.get(url)
    return response.json()

def load_articles(path: str) -> List[Dict]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            return []
    return []


def url_exists(articles: List[Dict], url: str) -> bool:
    return any(a.get("url") == url for a in articles)


if __name__ == "__main__":

    """
    example: page=1&limit=30&isCategoryHeadline=0&startAt=1769059882&endAt=1769923882
    every news url like: https://news.cnyes.com/news/id/{news_id}, news_id in the response data
    time format is Unix timestamp
    """
    articles = load_articles("spyder/article.json")
    count = 0
    isBreak = False
    limit = 30
    while not isBreak:
        count += 1
        result = fetch_anue_stock_news(Page=count, limit=limit)
        print(f"fetched page {count}, got {len(result['items']['data'])} items")
        if not result["items"]["data"] or len(result["items"]["data"]) < limit:
            isBreak = True
        for j in result["items"]["data"]:
            newsurl = f"https://news.cnyes.com/news/id/{j['newsId']}"
            if url_exists(articles, newsurl):
                print(f"skip existing url: {newsurl}")
                continue
            if j["categoryName"] not in ("台股新聞", "ETF", "台股"):
                continue
            published_at = datetime.fromtimestamp(j["publishAt"], tz=timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S%z")
            article = {
                "title": j.get("title", ""),
                "date_publish": published_at,
                "url": newsurl,
                "content": j.get("content", ""),
                "keywords": j.get("keyword", []),
                "stock": j.get("stock", []),
            }
            articles.append(article)

    with open("spyder/article.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    print(f"written {len(articles)} articles to spyder/article.json")