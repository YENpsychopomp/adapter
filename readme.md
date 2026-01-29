# This is a adapter service for Azure OpenAI API
# It receives requests from clients and forwards them to Azure OpenAI API
# Make sure to set the following environment variables:
# AZURE_API_ENDPOINT: Your Azure OpenAI API endpoint
# AZURE_API_KEY: Your Azure OpenAI API key
# AZURE_API_VERSION: The API version to use (e.g., 2024-08-01-preview)
# AZURE_DEPLOYMENT_NAME: The deployment name of your model in Azure
# Example usage:
# curl -X POST "http://localhost:8000/adapter" -H "Content-Type: application/json" -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello, world!"}]}'