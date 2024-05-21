FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set a working directory inside the container
WORKDIR /vllm

COPY requirements.txt requirements.txt


# Install dependencies
RUN pip install -r requirements.txt

COPY requirements-build.txt requirements-build.txt
RUN pip install -r requirements-build.txt

# max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads
# make sure punica kernels are built (for LoRA)
ENV VLLM_INSTALL_PUNICA_KERNELS=1
COPY . .
RUN pip install -e .

WORKDIR /vllm/vllm
