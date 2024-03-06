# Build
`docker build -t vllm-env .`

# Huggingface
## Run
`docker run --gpus all -it --rm --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 vllm-env`

## Launch API server
- runs on a 4090, may need to change `--max-model-len` and `--max-num-seqs` depending on hardware

`python -m vllm.entrypoints.api_server --model serpdotai/sparsetral-16x7B-v2 --trust-remote-code --gpu-memory-utilization 1.0 --tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 128`

# Local Weights
## Run
`docker run --gpus all -it --rm --ipc=host -v "PATH_TO_WEIGHTS_FOLDER:/vllm/vllm/model_weights" -p 8000:8000 vllm-env`

## Launch API server
`python -m vllm.entrypoints.api_server --model model_weights/sparsetral-16x7B-v2 --trust-remote-code --gpu-memory-utilization 1.0 --tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 128`