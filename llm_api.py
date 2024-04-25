from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from threading import Thread
import json

tokenizer = None
model = None

def create_app(model_path):
    global tokenizer
    global model 

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype='auto', 
        device_map='auto').eval()
    
    app = FastAPI(title='LLM Server')
    app.post('/chat', summary='大模型对话')(chat)
    return app

async def chat(data: dict = Body({}, description='用户输入')):
    input = data.get('input', '')
    stream = data.get('stream', False)

    messages = [
        {'role': 'system', 'content': '你是一个智能问答对话助手。'},
        {'role': 'user', 'content': input}
    ]

    global tokenizer
    global model 

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

    if stream:
        def get_stream():
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
                    word = word[:word.find('<|im_end|>')]
                
                ret = {'word': word}
                yield json.dumps(ret, ensure_ascii=False)

        return StreamingResponse(get_stream())
    else:
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.5
        )

        generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):]

        output = tokenizer.decode(generated_ids, skip_special_tokens=True)

        ret = {'output': output}
        return json.dumps(ret, ensure_ascii=False)
