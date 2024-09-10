import cv2 
import torch
import streamlit as st 
from transformers import AutoProcessor, LlavaForConditionalGeneration

st.title('Llava Demo')
MODEL_PATH = '/home/tyh/codes/llamafactory/sft_output/llava-7b-sft'

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map='auto',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval()

    return processor, model

if __name__ == '__main__':
    processor, model = load_model()

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

                st.session_state.img = img 
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
            else:
                input_prompt = f'USER: <image>\n{prompt}\nASSISTANT:'
                inputs = processor(input_prompt, st.session_state.img, return_tensors='pt').to(model.device, torch.float16)
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                response = processor.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            st.markdown(response)

        st.session_state.messages.append({
            'role': 'assistant',
            'is_img': False,
            'content': response
        })
