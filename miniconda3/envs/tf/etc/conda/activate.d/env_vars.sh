CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_PATH/lib:$CONDA_PREFIX/lib/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/
