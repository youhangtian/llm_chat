export CUDA_VISIBLE_DEVICES=2
vllm serve /file/tian/models/Qwen2-VL-7B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --port 8801
