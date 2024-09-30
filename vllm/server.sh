vllm serve /file/models/Qwen2.5-32B-Instruct \
 --tensor-parallel-size 8 \
 --max-model-len 10000 \
 --port 8800