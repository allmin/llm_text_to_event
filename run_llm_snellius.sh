#!/bin/bash
#SBATCH --job-name=allmin-llm-event-extraction
#SBATCH --output=logs/event_llm_%j.out   
#SBATCH --error=logs/event_llm_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_mig
#SBATCH --reservation=terv92681


# -----------------------------
# Load modules
# -----------------------------
module purge
module load python/3.12.1
module load cuda/12.1   # only if Ollama + LLM need GPU

# -----------------------------
# Setup Python environment
# -----------------------------
if [ ! -d "$SLURM_TMPDIR/venv" ]; then
    python -m venv $SLURM_TMPDIR/venv
fi
source $SLURM_TMPDIR/venv/bin/activate

pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

python -m spacy download en_core_web_lg

# -----------------------------
# Setup Ollama
# -----------------------------
# Install Ollama if not already present
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    export PATH="$HOME/.ollama/bin:$PATH"
fi

# Start Ollama service in background
ollama serve &

# Give it a few seconds to initialize
sleep 10

# Pull the model you want (adjust model name if different)
# Common choices: llama2, mistral, etc.
ollama pull llama3.1:70b

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
cd scripts
python 05_run_llm_on_F-SET.py
