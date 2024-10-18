import requests
import time

VLLM_API_URL = 'http://localhost:8800/v1/chat/completions'
MODEL_NAME = '/models/Qwen2.5-32B-Instruct'

prompt = '你好'

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': prompt}
]

request_data = {
    'model': MODEL_NAME,
    'messages': messages,
}

t1 = time.time()
response = requests.post(VLLM_API_URL, json=request_data)
output = response.json()['choices'][0]['message']['content']
t2 = time.time()

print('output:', output)
print('time:', t2 - t1)