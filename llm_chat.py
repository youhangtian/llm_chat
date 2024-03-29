import streamlit as st 
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from utils import get_config_from_yaml

st.title('Large Language Model Chat Demo')

@st.cache_resource
def load_model():
    config = get_config_from_yaml('cfg.yaml')

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        device_map='auto',
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        device_map='auto',
        trust_remote_code=True
    ).eval()

    return tokenizer, model

def response_generator(tokenizer, model, prompt):
    messages = [
        {'role': 'system', 'content': '你是一个智能问答助手，会用简短的语句回答用户的提问。'},
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
        add_special_tokens=False
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        decode_kwargs=dict(
            skip_special_tokens=True,
        )
    )

    generation_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=512,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for word in streamer:
        if '<|im_end|>' in word:
            yield word[:word.find('<|im_end|>')]
        else:
            yield word

if __name__ == '__main__':
    tokenizer, model = load_model()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input('What is up?'):
        with st.chat_message('user'):
            st.markdown(prompt)

        st.session_state.messages.append({
            'role': 'user',
            'content': prompt
        })

        with st.chat_message('assistant'):
            stream = response_generator(tokenizer, model, prompt)
            response = st.write_stream(stream)
            st.markdown(response)

        st.session_state.messages.append({
            'role': 'assistant',
            'content': response
        })
    