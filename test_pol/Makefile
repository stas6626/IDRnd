help:
	@cat Makefile

DATA?="/media/stas/ssd/data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
PYTHON_VERSION?=3.6
CUDA_VERSION?=9.0
CUDNN_VERSION?=7
TEST=tests/
SRC?="/media/stas/ssd/Notebooks/IDRND/test_pol"
SRCd?="/media/stas/ssd/Notebooks/IDRND"
test:
	nvidia-docker run --memory 20g -v $(SRC):/src/workspace/ -v $(SRC):/workspace/ -w /workspace/ -it boosters_gpu_python python3 script.py
