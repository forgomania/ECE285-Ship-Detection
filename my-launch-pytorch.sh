#!/bin/sh

set -a  # mark all variables below as exported (environment) variables

# Indentify this script as source of job configuration
K8S_CONFIG_SOURCE=${BASH_SOURCE[0]}

K8S_CUDA_VERSION=${K8S_CUDA_VERSION:-"8"}
if [ "$K8S_CUDA_VERSION" == "9" ]; then
	K8S_DOCKER_IMAGE=${K8S_PY3TORCH_DOCKER_IMAGE:-"ucsdets/instructional:ets-pytorch-py3-cuda9-20181019v1"}
else 
	K8S_DOCKER_IMAGE=${K8S_PY3TORCH_DOCKER_IMAGE:-"ucsdets/instructional:ets-pytorch-py3-latest"}
fi

K8S_ENTRYPOINT="/run_jupyter.sh"

K8S_NUM_GPU=1  # max of 2 (contact ETS to raise limit)
K8S_NUM_CPU=4  # max of 8 ("")
K8S_GB_MEM=16  # max of 64 ("")
K8S_TIMEOUT_SECONDS=43200
# Controls whether an interactive Bash shell is started
SPAWN_INTERACTIVE_SHELL=YES

# Sets up proxy URL for Jupyter notebook inside 
PROXY_ENABLED=YES
PROXY_PORT=8888

exec /software/common64/dsmlp/bin/launch.sh "$@"

