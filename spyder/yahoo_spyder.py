import os
import time
import random
import requests
from bs4 import BeautifulSoup
import json
import re
from typing import List, Dict, Optional, Set
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


def polite_get(url: str, retries: int = 3, base_delay: float = 1.0):
    """Polite HTTP GET with backoff and random jitter."""
    for attempt in range(retries):
        try:
            time.sleep(base_delay + random.uniform(0, 1))
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp
        except requests.RequestException:
            pass
        base_delay *= 1.6
    return None

def fetch_yahoo_stock_news(url):

    response = polite_get(url)
    if not response:
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Yahoo 股市新聞的內文通常在 .caas-body 這個 class 裡面
    article_body = soup.find('div', class_='caas-body')
    
    if not article_body:
        # 備用方案：尋找文章主要的容器
        article_body = soup.find('article')

    # 提取所有段落文字
    paragraphs = article_body.find_all('p') if article_body else []
    full_text = "\n".join([p.get_text() for p in paragraphs])

    # 提取標題
    title = soup.find('h1').get_text() if soup.find('h1') else ""

    return {
        "title": title,
        "url": url,
        "full_text": full_text,
        "date_publish": soup.find('time')['datetime'] if soup.find('time') else None
    }


def _detect_industry(content: str) -> str:
    industry_map = {
        "半導體": ["台積電", "晶圓", "IC設計", "封測", "聯發科"],
        "航運": ["SCFI", "貨櫃", "長榮", "陽明", "萬海", "航運"],
        "AI": ["散熱", "伺服器", "廣達", "緯創", "英業達", "人工智慧"],
        "金融": ["銀行", "金控", "除權息", "壽險"],
    }
    for ind, keywords in industry_map.items():
        if any(k in content for k in keywords):
            return ind
    return "綜合"

def clean_news_data(raw_data: Dict) -> Dict:
    text = raw_data.get("full_text", "")
    
    # 1. 去除末尾雜訊 (從 "原始連結" 或 "看更多" 切斷)
    noise_indicators = ["原始連結", "看更多", "文章"]
    for indicator in noise_indicators:
        if indicator in text:
            text = text.split(indicator)[0]
    
    # 2. 提取股票代碼與名稱 (格式: 名稱（代碼）)
    stock_pattern = r'([\u4e00-\u9fa5]{2,4})（(\d{4})）'
    stocks = re.findall(stock_pattern, text)
    stock_list = [{"name": s[0], "code": s[1]} for s in stocks]
    
    # 3. 提取關鍵數字 (SCFI 指數、跌幅)
    scfi_pattern = r'SCFI.*?([\d,]+\.\d+)'
    scfi_match = re.search(scfi_pattern, text)
    scfi_value = scfi_match.group(1) if scfi_match else None
    
    # 4. 提取百分比 (漲跌幅)
    percentage_pattern = r'跌幅([\d\.]+)％|周跌([\d\.]+)％'
    percentages = re.findall(percentage_pattern, text)
    # 扁平化並過濾空值
    percentages = [p[0] or p[1] for p in percentages]

    # 5. 格式化清洗後的內文 (去除多餘換行)
    clean_text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])

    # 構建新的結構化資料
    return {
        "title": raw_data.get("title"),
        "date_publish": raw_data.get("date_publish"),
        "url": raw_data.get("url"),
        "content": clean_text,
    }

def get_news_list(list_url: str) -> List[str]:
    """從列表頁抓取所有新聞連結"""
    resp = polite_get(list_url)
    if not resp:
        return []
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # 尋找所有新聞標題的連結 (Yahoo 的 class 經常變動，這裡抓取具備 href 的標題節點)
    links = []
    for a in soup.find_all('a', href=True):
        if '/news/' in a['href'] and 'html' in a['href']:
            full_url = "https://tw.stock.yahoo.com" + a['href'] if a['href'].startswith('/') else a['href']
            links.append(full_url)
    return list(set(links))


def scroll_and_collect_links(list_url: str, max_scroll: int = 15, wait: float = 1.2) -> List[str]:
    """Use headless browser scroll to load more news; fallback to single-page fetch if Selenium is unavailable."""

    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument(f'user-agent={headers["User-Agent"]}')

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(list_url)
        last_height = driver.execute_script("return document.body.scrollHeight")
        for _ in range(max_scroll):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(wait + random.uniform(0, 0.5))
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            if '/news/' in a['href'] and 'html' in a['href']:
                full_url = "https://tw.stock.yahoo.com" + a['href'] if a['href'].startswith('/') else a['href']
                links.append(full_url)
        return list(set(links))
    finally:
        driver.quit()

def extract_industry_and_clean(article_url: str) -> Dict:
    """進入內文頁，抓取內容與自動辨識產業"""
    resp = polite_get(article_url)
    if not resp:
        return {"url": article_url, "industry": "未知", "content": ""}
    soup = BeautifulSoup(resp.text, 'html.parser')

    # 1. 嘗試抓取 Yahoo 的「分類」標籤 (通常在標題上方或下方)
    # Yahoo 有時會標註「半導體」、「電子上游」等
    category = "綜合"
    cat_tag = soup.find('div', class_='caas-attr-item-category') # 範例 class
    if cat_tag:
        category = cat_tag.get_text()

    # 2. 提取全文並去除雜訊
    article_body = soup.find('div', class_='caas-body')
    content = ""
    if article_body:
        # 清除「延伸閱讀」等雜訊
        for s in article_body(['script', 'style', 'aside']):
            s.decompose()
        content = article_body.get_text(separator='\n').strip()
        if "原始連結" in content:
            content = content.split("原始連結")[0]

    # 3. 關鍵字二次判定 (如果分類抓不到)
    category = _detect_industry(content) if category == "綜合" else category

    return {
        "url": article_url,
        "industry": category,
        "content": content
    }


def load_existing_urls(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        urls = [item.get("url") for item in data if isinstance(item, dict) and item.get("url")]
        return set(urls)
    except (json.JSONDecodeError, OSError):
        return set()


def process_news(list_url: str, limit: int = 5, output_path: Optional[str] = "spyder/article.json") -> List[Dict]:
    links = scroll_and_collect_links(list_url)
    results: List[Dict] = []
    seen = load_existing_urls(output_path) if output_path else set()

    for link in links:
        if len(results) >= limit:
            break
        if link in seen:
            continue
        raw = fetch_yahoo_stock_news(link)
        if not raw:
            continue
        cleaned = clean_news_data(raw)
        results.append(cleaned)
        seen.add(link)
        print(f"處理完成: {cleaned['title']}")

    if output_path:
        # merge with existing data
        existing = []
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    existing = [existing]
            except (json.JSONDecodeError, OSError):
                existing = []
        combined = existing + results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=4)

    return results

def main(url):
    processed = process_news(url, limit=10000, output_path="spyder/article.json")
    print(f"成功處理 {len(processed)} 則新聞，結果寫入 spyder/article.json")

if __name__ == "__main__":
    url = 'https://tw.stock.yahoo.com/tw-market'
    main(url)