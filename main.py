import Adapter

text = None
with open("test.txt", "r") as file:
    text = file.read()

client = Adapter.adapter()

body = {
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "You are a professional summarization assistant, you only respond with concise summaries, do not include any additional information."},
        {"role": "user", "content": text}
    ],
    "stream": False,
    "temperature": 0.7,
    "max_tokens": 100
}

response = client.create(body)
print(response) 