import asyncio
import uvicorn
import subprocess
from multiprocessing import Process
from llm_api import create_app

#MODEL_PATH = '/file/models/Qwen1.5-110B-Chat'
MODEL_PATH = '/home/tyh/git/LLaMA-Factory/my_ft_output/qwen-14b-yc'
HOST = '0.0.0.0'
LLM_PORT = 8876
ST_PORT = 8877

def run_api_server():
    app = create_app(MODEL_PATH)
    uvicorn.run(app, host=HOST, port=LLM_PORT)

def run_webui():
    cmd = ['streamlit', 'run', 'llm_webui.py', f'{LLM_PORT}',
           '--server.address', HOST,
           '--server.port', f'{ST_PORT}']
    
    p = subprocess.Popen(cmd)
    p.wait()


async def start_server():
    p1 = Process(
        target=run_api_server,
        name=f'LLM API Server',
        daemon=True,
    )
    p1.start()

    p2 = Process(
        target=run_webui,
        name=f'Webui Server',
        daemon=True,
    )
    p2.start()

    p1.join()
    p2.join()

if __name__ == '__main__':
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        
    loop.run_until_complete(start_server())