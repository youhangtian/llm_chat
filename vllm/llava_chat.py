import requests
import json
import cv2
import base64
import streamlit as st 

Vllm_api_url = 'http://localhost:8801/v1/chat/completions'
Model_name = '/file/tian/models/llava1.5-7b-hf'

def response_generator(img_base64, text):
    if img_base64 is not None:
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_base64}'}},
            ]
        }]
    else:
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text},
            ]
        }]

    request_data = {
        'model': Model_name,
        'messages': messages,
        'stream': True
    }

    response = requests.post(Vllm_api_url, json=request_data, stream=True)

    for data in response.iter_content(None, decode_unicode=True):
        if len(data) > 6 and not data[6:12] == '[DONE]':
            choices = json.loads(data[6:])['choices']
            if choices:
                word = choices[0]['delta']['content']
                yield word

if __name__ == '__main__':
    st.title(f'LLM({Model_name}) Chat Demo')

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if message['is_img']:
                st.image(message['content'])
            else:
                st.markdown(message['content'])

    if 'img_base64' not in st.session_state:
        st.session_state.img_base64 = None 

    if prompt := st.chat_input('What is up?'):
        is_img = False
        img_suffix = ['.jpg', '.jpeg', '.png']
        with st.chat_message('user'):
            if 'http' in prompt:
                try:
                    cap = cv2.VideoCapture(prompt)
                    _, img = cap.read(cv2.IMREAD_IGNORE_ORIENTATION)
                    is_img = True 
                except Exception as e:
                    st.markdown(f'{prompt}\n{e}')
            elif any([suffix in prompt.lower() for suffix in img_suffix]):
                try:
                    img = cv2.imread(prompt)
                    is_img = True 
                except Exception as e:
                    st.markdown(f'{prompt}\n{e}')
            
            if is_img:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img)

                img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                st.session_state.img_base64 = img_base64

                st.session_state.messages.append({
                    'role': 'user',
                    'is_img': is_img,
                    'content': img
                })
            else:
                st.markdown(prompt)

                st.session_state.messages.append({
                    'role': 'user',
                    'is_img': is_img,
                    'content': prompt
                })

        with st.chat_message('assistant'):
            if is_img: 
                response = '请根据图片提问'
                st.markdown(response)
            else:
                stream = response_generator(st.session_state.img_base64, prompt)
                response = st.write_stream(stream)

        st.session_state.messages.append({
            'role': 'assistant',
            'is_img': False,
            'content': response
        })
    