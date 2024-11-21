docker run --name qwen7b_server \
  --gpus='"device=1"' \
  -v /file/models:/models \
  -ti \
  -p 8807:8000 \
  --ipc host \
  vllm/vllm-openai:latest \
  --model /models/Qwen2.5-7B-Instruct \
  --max-model-len 10000 \
  --tensor-parallel-size 1
