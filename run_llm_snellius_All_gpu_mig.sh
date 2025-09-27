#!/bin/bash
#SBATCH --job-name=llm-event-extraction-All
#SBATCH --output=logs/event_llm_%j.out   
#SBATCH --error=logs/event_llm_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_mig
#SBATCH --reservation=terv92681
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@60

# -----------------------------
# Load modules
# -----------------------------
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load 2025
module load CUDA/12.8.0   # only if Ollama + LLM need GPU

# -----------------------------
# Setup Python environment
# -----------------------------
VENV_DIR="~/projects/llm_text_to_event/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"


pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

python -m spacy download en_core_web_lg

# -----------------------------
# Setup Ollama
# -----------------------------
export PATH="$HOME/.ollama/bin:$PATH"

# Start Ollama service in background
ollama serve &

# Give it a few seconds to initialize
sleep 10

# Pull the model you want (adjust model name if different)
# Common choices: llama2, mistral, etc.
# ollama pull llama3.1:70b

# -----------------------------
# Environment variables
# -----------------------------
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Please create it with MIMICPATH before running."
    exit 1
fi
export $(grep -v '^#' .env | xargs)

# -----------------------------
# Run the target script
# -----------------------------
nvidia-smi
cd scripts
python 05_run_llm_on_P-SET.py --attribute_output True
nvidia-smi
# python event_extractor.py
# sbatch --mail-type=ALL --mail-user=a.p.s.susaiyah@tue.nl run_llm_snellius.sh
# srun --partition=gpu_a100 --gres=gpu:1 --cpus-per-task=18 --mem=100G --time=8:00:00 --pty bash -i
srun --partition=gpu_mig --reservation=terv92681 --gres=gpu:a100_3g.20gb:1 --cpus-per-gpu=9 --mem=60G --mail-type=BEGIN,END,FAIL --mail-user=your_email@example.com --pty bash
# srun --partition=gpu_a100 --gres=gpu:1 --cpus-per-task=18 --mem=100G --time=8:00:00 jupyter lab --no-browser --port=8888
