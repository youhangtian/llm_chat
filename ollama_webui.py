import streamlit as st 
import requests
import json

Ollama_api_url = 'http://localhost:11434/api/chat'
Model_name = 'qwen2:1.5b'

def response_generator(user_input):
    messages = [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': user_input}
    ]

    request_data = {
        'model': Model_name,
        'messages': messages,
        'stream': True
    }

    response = requests.post(Ollama_api_url, json=request_data, stream=True)

    for data in response.iter_content(None, decode_unicode=True):
        word = json.loads(data)['message']['content']
        yield word

if __name__ == '__main__':
    st.title(f'LLM({Model_name}) Chat Demo')

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
            #st.markdown(response)

        st.session_state.messages.append({
            'role': 'assistant',
            'content': response
        })
    
