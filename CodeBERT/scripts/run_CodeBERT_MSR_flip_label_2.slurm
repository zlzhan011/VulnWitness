#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 20:20:00
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH --account=loni_quantum04
#SBATCH --output=logs/flip_label_2_training_noise_label_output.log
#SBATCH --error=logs/flip_label_2_training_noise_label_error.log




module purge
module load cuda/11.8.0
export CUDA_HOME=/usr/local/packages/cuda/11.8.0/7oedjh4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate LLaMA-Factory
python -c "import sys; print(sys.executable)"  # 确认 Python 路径


python -c "import deepspeed; print(deepspeed.__version__)"
python -c "import torch; print(torch.cuda.is_available())"


sh run_msr_flip_label_loni_2.sh



conda deactivate

