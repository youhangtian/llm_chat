import streamlit as st 
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from transformers.generation import GenerationConfig 

import torch
torch.manual_seed(1234)

st.title('Vision Language Model Chat Demo')
MODEL_PATH = '/home/tian/models/Qwen-VL-Chat'

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        device_map='auto',
        trust_remote_code=True,
        resume_download=True,
        revision='master'
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map='auto',
        trust_remote_code=True,
        resume_download=True,
        revision='master'
    ).eval()

    model.generation_config = GenerationConfig.from_pretrained(
        MODEL_PATH,
        trust_remote=True,
        resume_download=True,
        revision='master'
    )

    return tokenizer, model

def response_generator(tokenizer, model, img, text, history=None):
    if img:
        query = tokenizer.from_list_format([
            {'image': img},
            {'text': text}
        ])
    else:
        query = tokenizer.from_list_format([
            {'text': text}
        ])

    generator = model.chat_stream(
        tokenizer,
        query=query,
        history=history,
        system='你是一个智能问答助手。'
    )

    cur_len = 0
    for response in generator:
        new_text = response[cur_len:]
        cur_len = len(response)
        yield new_text

if __name__ == '__main__':
    tokenizer, model = load_model()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if message['is_img']:
                st.image(message['content'])
            else:
                st.markdown(message['content'])

    if 'img' not in st.session_state:
        st.session_state.img = None 

    if prompt := st.chat_input('What is up?'):
        img_suffix = ['jpg', 'jpeg', 'JPG', 'JPEG']
        is_img = False
        with st.chat_message('user'):
            if any([suffix in prompt for suffix in img_suffix]):
                try:
                    st.image(prompt)
                    is_img = True
                    st.session_state.img = prompt 
                except:
                    st.markdown(prompt)
            else:
                st.markdown(prompt)

        st.session_state.messages.append({
            'role': 'user',
            'is_img': is_img,
            'content': prompt
        })

        with st.chat_message('assistant'):
            if is_img:
                stream = response_generator(tokenizer, model, st.session_state.img, '描述这张图片')
            else:
                stream = response_generator(tokenizer, model, st.session_state.img, prompt)
            response = st.write_stream(stream)
            #st.markdown(response)

        st.session_state.messages.append({
            'role': 'assistant',
            'is_img': False,
            'content': response
        })
    