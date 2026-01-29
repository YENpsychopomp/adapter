import os
import tiktoken
from openai import AzureOpenAI

class adapter:
    def __init__(self, api_key=None, endpoint=None, api_version=None):
        if not api_key and not endpoint and not api_version:
            raise ValueError("At least one parameter must be provided")
        self.api_key = api_key if api_key else os.getenv("API_KEY", None)
        self.endpoint = endpoint
        self.api_version = api_version
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )
    
    def create(self, body):
        if not self.api_key:
            raise ValueError("API key is required")
        try:
            model = body.get("model", None)
            if not model:
                raise ValueError("Model name is required in the body")
            response = self.client.chat.completions.create(**body)
            if not body.get("stream"):
                res_dict = response.model_dump()
                res_dict.pop("prompt_filter_results", None)
                for choice in res_dict.get("choices", []):
                    choice.pop("content_filter_results", None)
                return res_dict
    
            return response
        except Exception as e:
            raise ValueError(f"Invalid endpoint or deployment name\ndetails: {e}") from e
    
    def compute_token(self, input_text, output_text):
        encoding = tiktoken.get_encoding("cl100k_base")
        input_tokens = encoding.encode(input_text)
        output_tokens = encoding.encode(output_text)
        return {
            "input_token_count": len(input_tokens),
            "output_token_count": len(output_tokens),
            "total_token_count": len(input_tokens) + len(output_tokens),
        }