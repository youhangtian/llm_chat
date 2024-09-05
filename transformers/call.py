import requests 

api_url = 'http://0.0.0.0:8866/chat'
prompt = '你是谁'

request_data = {
    'input': prompt,
    'stream': False,
    'temp': 0.8
}

response = requests.post(api_url, json=request_data)
output = response.json()['output']
print('output:', output)
