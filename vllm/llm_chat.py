import requests
import json
import streamlit as st 

VLLM_API_URL = 'http://localhost:8800/v1/chat/completions'
MODEL_NAME = '/models/Qwen2.5-32B-Instruct'

def response_generator(user_input):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': user_input}
    ]

    request_data = {
        'model': MODEL_NAME,
        'messages': messages,
        'stream': True
    }

    response = requests.post(VLLM_API_URL, json=request_data, stream=True)

    for data in response.iter_content(None, decode_unicode=True):
        if data.startswith('data: '): data = data[6:]
        if data and not data.startswith('[DONE]'):
            word = json.loads(data)['choices'][0]['delta']['content']
            yield word

if __name__ == '__main__':
    st.title(f'LLM({MODEL_NAME}) Chat Demo')

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if user_input := st.chat_input('What is up?'):
        with st.chat_message('user'):
            st.markdown(user_input)

        st.session_state.messages.append({
            'role': 'user',
            'content': user_input
        })

        with st.chat_message('assistant'):
            stream = response_generator(user_input)
            response = st.write_stream(stream)

        st.session_state.messages.append({
            'role': 'assistant',
            'content': response
        })
    