export CUDA_VISIBLE_DEVICES=4,5
vllm serve /file/tian/models/llava-v1.6-vicuna-7b-hf \
  --chat_template chat_template.jinja \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --port 8801