import streamlit as st 
import requests
import json
import sys

def response_generator(user_input, api_url):
    request_data = {
        'input': user_input,
        'stream': True
    }

    response = requests.post(api_url, json=request_data, stream=True)

    for data in response.iter_content(None, decode_unicode=True):
        word = json.loads(data)
        yield word.get('word', '')

if __name__ == '__main__':
    st.title('Large Language Model Chat Demo')

    port = sys.argv[1]
    api_url = f'http://0.0.0.0:{port}/chat'

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
            stream = response_generator(user_input, api_url)
            response = st.write_stream(stream)
            #st.markdown(response)

        st.session_state.messages.append({
            'role': 'assistant',
            'content': response
        })
    
