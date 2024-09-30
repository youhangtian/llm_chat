docker run --name qwen32b_server \
  --gpus all \
  -v /file/models:/models \
  -p 8800:8000 \
  --ipc host \
  vllm/vllm-openai:latest \
  --model /models/Qwen2.5-32B-Instruct \
  --max-model-len 10000 \
  --tensor-parallel-size 4
