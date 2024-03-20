import torch
import streamlit as st 
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers_stream_generator import init_stream_support
from utils import get_config_from_yaml

st.title('Large Language Model Chat Demo')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if 'config' not in st.session_state:
    st.session_state['config'] = get_config_from_yaml('cfg.yaml')

if 'model' not in st.session_state:
    init_stream_support()
    config = st.session_state['config']

    st.session_state['model'] = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        device_map='auto',
        trust_remote_code=True 
    ).eval()

    st.session_state['tokenizer'] = AutoTokenizer.from_pretrained(
        config.model_path,
        device_map='auto',
        trust_remote_code=True
    )

def response_generator(tokenizer, model, prompt):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]

    text= tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        text,
        return_tensors='pt',
        add_specal_tokens=False
    ).input_ids.to(model.device)

    generator = model.generate(
        model_inputs,
        do_stream=True,
        do_sample=True,
        max_new_tokens=20,
        repetition_penalty=1.2,
        early_stopping=True,
    )

    for token in generator:
        token = token.cpu().numpy().tolist()
        word = tokenizer.decode(token, skip_special_tokens=True)
        yield word 

if prompt := st.chat_input('What is up?'):
    with st.chat_message('user'):
        st.markdown(prompt)

    st.session_state.messages.append({
        'role': 'user',
        'content': prompt
    })

    with st.chat_message('assistant'):
        tokenizer = st.session_state['tokenizer']
        model = st.session_state['model']
        stream = response_generator(tokenizer, model, prompt)
        response = st.write_stream(stream)
        st.markdown(response)

    st.session_state.messages.append({
        'role': 'assistant',
        'content': response
    })
