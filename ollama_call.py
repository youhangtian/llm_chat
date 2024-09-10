import requests

Ollama_api_url = 'http://localhost:11434/api/chat'
Model_name = 'qwen2:72b'

prompt = '写一篇500字的作文'

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': prompt}
]

request_data = {
    'model': Model_name,
    'messages': messages,
    'stream': False,
    'options': {
        'temperature': 0.7,
        'num_predict': 2048
    }
}

response = requests.post(Ollama_api_url, json=request_data)
output = response.json()['message']['content']
print(output)