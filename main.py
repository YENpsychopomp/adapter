import Adapter
import os
import dotenv
import time

start_time = time.time()
dotenv.load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", None)
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", None)
AZURE_API_KEY = os.getenv("API_KEY", None)

with open("test.txt", "r") as file:
    text = file.read()

client = Adapter.adapter(api_key=AZURE_API_KEY, endpoint=AZURE_ENDPOINT, api_version=AZURE_API_VERSION)

body = {
    "model": "gpt-4.1-nano",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Please using chinese teach me the article:\n" + text}
    ],
    "stream": True,
    "temperature": 1.0,
}
if body.get("stream"):
    stream = client.create(body)
    strs = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if hasattr(delta, 'content') and delta.content:
            print(delta.content, end="", flush=True)
            strs.append(delta.content)

        if chunk.choices[0].finish_reason == "stop":
            print("\n--- 總結完成 ---")
            tokens = client.compute_token(input_text=text, output_text="".join(strs))
            print(f"輸入文字token數量: {tokens['input_token_count']}\n模型輸出token數量: {tokens['output_token_count']}\n總token數量: {tokens['total_token_count']}")
            break
else:
    response = client.create(body)
    print(response.choices[0].message.content)
    print("\n--- 總結完成 ---")
    print(f"輸入文字token數量: {response.usage.prompt_tokens}\n模型輸出token數量: {response.usage.completion_tokens}\n總token數量: {response.usage.total_tokens}")
end_time = time.time()
print(f"執行時間: {end_time - start_time} 秒")