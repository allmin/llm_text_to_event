#!/bin/bash
#SBATCH --job-name=event_llm
#SBATCH --output=logs/event_llm_%j.out
#SBATCH --error=logs/event_llm_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu          # change if you don’t need GPU
#SBATCH --gres=gpu:1             # remove if CPU-only
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# -----------------------------
# Load modules
# -----------------------------
module purge
module load python/3.12.1
module load cuda/12.1   # only if using GPU + Ollama supports it
# (adjust module versions to match your cluster)

# -----------------------------
# Set up environment
# -----------------------------
# Create virtual environment if not exists
if [ ! -d "$SLURM_TMPDIR/venv" ]; then
    python -m venv $SLURM_TMPDIR/venv
fi
source $SLURM_TMPDIR/venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model (only once)
python -m spacy download en_core_web_lg

# -----------------------------
# Ensure environment variables
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
