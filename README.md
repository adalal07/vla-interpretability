# vla-interpretability
CS 182 Project

## To use LeRobot library
### Setting up the environment
```bash
# Environment setup
conda create -y -n lerobot python=3.10

# Activate environment
conda activate lerobot

# Video processing
conda install ffmpeg -c conda-forge
```

### Installing LeRobot

#### From source
```bash
# Install LeRobot (if not already installed)
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Install library in editable mode 
pip install -e .
```

#### From PyPI
```bash
# Install library
pip install lerobot
```

#### Extra libraries (If necessary)
```bash
# Extra required libraries that might not have been installed
pip install transformers accelerate sentencepiece num2words
```

### Running scripts
```bash
# Give permissions to bash file
chmod +x run_sweep.sh

# Run sweep script
./run_sweep.sh
```


