#!/bin/bash

# See https://stackoverflow.com/a/28336473
SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

# this one reset the enviroment varibles if you have ever activated other venv.
# it will not trigged in slurm, if you only use one venv


RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

[[ "$0" != "${SOURCE_PATH}" ]] && echo "The activation script must be sourced, otherwise the virtual environment will not work." || ( echo "Vars script must be sourced." && exit 1) ;

source "${ABSOLUTE_PATH}/config.sh" "$@"
source "${ABSOLUTE_PATH}/modules.sh"


#=======
ENV_PROJ_DIR="$(dirname "$(realpath $ENV_DIR)")"
export TMPDIR="${ENV_PROJ_DIR}/.tmp"
export TMP=${TMPDIR}
export TEMP=${TMPDIR}
export PIP_CACHE_DIR="${ENV_PROJ_DIR}/.cache/pip"

PROJECT_DIR="$(dirname "$(realpath $ABSOLUTE_PATH)")"
export CACHE_DIR="${PROJECT_DIR}/.cache"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton"
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TORCH_HOME="${PROJECT_DIR}/.cache/torch/hub"
export TORCH_EXTENSIONS_DIR="${PROJECT_DIR}/.cache/torch/torch_extensions"
export MPLCONFIGDIR="${PROJECT_DIR}/.cache/matplotlib"
export WANDB_CACHE_DIR="${PROJECT_DIR}/.cache/wandb"
# ======
export PYTHONPATH="$(echo "${ENV_DIR}"/lib/python*/site-packages):${PYTHONPATH}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

export http_proxy=http://134.94.199.178:7008; 
export https_proxy=$http_proxy 
export HTTP_PROXY=$http_proxy 
export HTTPS_PROXY=$http_proxy
export PROXY_CONNECTED=$(nc -zv 134.94.199.178 7008 >/dev/null 2>&1 && echo "true" || echo "false")
[ "$PROXY_CONNECTED" = "false" ] && unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY



# PYTHON_BIN_CHECK=$(which python)
# CURRENT_PYTHON_BIN_PATH="$ENV_DIR/bin/python3"
# rm $CURRENT_PYTHON_BIN_PATH
# ln -s $PYTHON_BIN_CHECK $CURRENT_PYTHON_BIN_PATH
source "${ENV_DIR}/bin/activate"


# this one used to add the NVIDIA's CUDA Runtime Libraries to the LD_LIBRARY_PATH
NPKG=$(python -c "import site, os; d=[p for p in site.getsitepackages() if p.endswith('site-packages')][0]+'/nvidia'; print(d)")
# Only touch LD_LIBRARY_PATH if that directory exists
if [ -d "$NPKG" ]; then
    export LD_LIBRARY_PATH="$NPKG/nvjitlink/lib:$NPKG/cusparse/lib:${LD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$NPKG/nvshmem/lib:$LD_LIBRARY_PATH"
    export NVSHMEM_DIR="$NPKG/nvshmem"
    export PATH="${NVSHMEM_DIR}/bin:$PATH"

    export LD_LIBRARY_PATH="$(
    printf '%s' "${LD_LIBRARY_PATH:-}" \
    | tr ':' '\n' \
    | grep -vE '/stages/20[0-9]{2}/software/CUDA' \
    | awk 'NF' \
    | paste -sd: -
    )"
fi


# this one used to add the NVIDIA's CUDA Runtime Libraries to the LD_LIBRARY_PATH
NPKG=$(python -c "import site, os; d=[p for p in site.getsitepackages() if p.endswith('site-packages')][0]+'/nvidia'; print(d)")
# Only touch LD_LIBRARY_PATH if that directory exists
if [ -d "$NPKG" ]; then
    export LD_LIBRARY_PATH="$NPKG/nvjitlink/lib:$NPKG/cusparse/lib:${LD_LIBRARY_PATH:-}"
fi


