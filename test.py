import json
from pathlib import Path
from typing import Dict, List, Tuple

import tiktoken


def load_articles(path: Path) -> List[Dict]:
	text = path.read_text(encoding="utf-8")
	data = json.loads(text)
	if isinstance(data, dict):
		return [data]
	return data


def count_tokens_text(text: str, enc) -> int:
	return len(enc.encode(text))


def count_tokens_articles(articles: List[Dict], enc) -> Tuple[int, int, int]:
	per_article = []
	for a in articles:
		payload = "\n".join(
			[
				a.get("title", ""),
				a.get("content", ""),
				" ".join(a.get("keyword", []) if isinstance(a.get("keyword"), list) else []),
				" ".join(a.get("stock", []) if isinstance(a.get("stock"), list) else []),
			]
		).strip()
		per_article.append(len(enc.encode(payload)))
	return sum(per_article), max(per_article) if per_article else 0, len(per_article)


if __name__ == "__main__":
	json_path = Path("spyder/article.json")
	enc = tiktoken.get_encoding("cl100k_base")

	text = json_path.read_text(encoding="utf-8")
	total_raw = count_tokens_text(text, enc)

	articles = load_articles(json_path)
	sum_payload, max_payload, n_articles = count_tokens_articles(articles, enc)

	print(f"raw JSON tokens: {total_raw}")
	print(f"articles: {n_articles}")
	print(f"title+content+keyword+stock tokens total: {sum_payload}")
	print(f"max per article: {max_payload}")
