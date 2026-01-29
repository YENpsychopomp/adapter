This is a adapter service for Azure OpenAI API, It receives requests from clients and forwards them to Azure OpenAI API.
>Make sure to set the following environment variables:
>* AZURE_API_ENDPOINT: Your Azure OpenAI API endpoint
>* AZURE_API_KEY: Your Azure OpenAI API key
>* AZURE_API_VERSION: The API version to use (e.g., 2024-08-01-preview)
>* AZURE_DEPLOYMENT_NAME: The deployment name of your model in Azure

## Example usage:
```python
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