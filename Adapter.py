import os
from dotenv import load_dotenv
import requests
load_dotenv()

DEFULT_AZURE_ENDPOINT = os.getenv("DEFULT_AZURE_ENDPOINT", None)
DEFULT_AZURE_API_VERSION = os.getenv("DEFULT_AZURE_API_VERSION", None)
DEFULT_AZURE_DEPLOYMENT_NAME = os.getenv("DEFULT_AZURE_DEPLOYMENT_NAME", None)

deployment_mapping = {
    "gpt-4": "gpt-4-deployment",
    "gpt-3.5-turbo": "gpt-35-turbo-deployment",
    "whisper": "whisper-deployment",
    "gpt-4o": "gpt-4o-deployment"
}

class adapter:
    def __init__(self, api_key=None, endpoint=None, model=None, api_version=None):
        self.api_key = api_key
        self.endpoint = endpoint if endpoint else DEFULT_AZURE_ENDPOINT
        self.deployment_name = deployment_mapping.get(model, DEFULT_AZURE_DEPLOYMENT_NAME) if model else DEFULT_AZURE_DEPLOYMENT_NAME
        self.api_version = api_version if api_version else DEFULT_AZURE_API_VERSION
    
    def create(self, body):
        if not self.api_key:
            raise ValueError("API key is required")
        try:
            azure = self.endpoint + "/openai/deployments/" + self.deployment_name + "/chat/completions?api-version=" + self.api_version
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            response = requests.post(azure, headers=headers, json=body)
            return response.json()
        except Exception as e:
            raise ValueError("Invalid endpoint or deployment name") from e