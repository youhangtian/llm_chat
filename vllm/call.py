import requests
import time

Vllm_api_url = 'http://localhost:8800/v1/chat/completions'
Model_name = '/models/Qwen2.5-32B-Instruct'

prompt = '你好'

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': prompt}
]

request_data = {
    'model': Model_name,
    'messages': messages,
}

t1 = time.time()
response = requests.post(Vllm_api_url, json=request_data)
output = response.json()['choices'][0]['message']['content']
t2 = time.time()

print('output:', output)
print('time:', t2 - t1)