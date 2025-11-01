# Related-key MCP Attention Inception-based Neural Distinguisher

This repository provides code for training and evaluating neural distinguishers for the PRESENT-80 block cipher using related-key, multi-pair, attention, and inception-based deep learning architectures.

---

## Features

- Neural distinguisher for PRESENT-80 using Inception and Attention modules.
- Related-key and multi-pair data generation.
- Automatic selection of the best delta key for related-key cryptanalysis.
- Training pipeline with Keras/TensorFlow, supporting both GPU (with CuPy) and CPU (with NumPy).
- Evaluation scripts with statistical significance testing.

---

## Folder structure

```
.
├── main.py                  # Main training and evaluation pipeline
├── train_nets.py            # Model, training utilities, and delta key selection
├── make_data_train.py       # Data generator for neural distinguisher
├── RKmcp.py                 # Inception-based model architecture with attention
├── eval_nets.py             # Evaluation and statistical analysis functions
├── cipher/                  # PRESENT-80 cipher implementation
├── requirements.txt         # Python dependencies
└── README.md                 
```

---

## Requirements and setup (Windows)

This project uses TensorFlow/Keras and CuPy. On Windows, you need a compatible CUDA toolkit to use CuPy. Steps below assume Command Prompt (cmd.exe).

1) Create and activate a virtual environment
```bat
python -m venv .venv
.venv\Scripts\activate
```

2) Install core dependencies
```bat
pip install --upgrade pip
pip install -r requirements.txt
```

3) Install CuPy matching your CUDA version (REQUIRED)
- CUDA 12.x:
```bat
pip install cupy-cuda12x
```
- CUDA 11.x:
```bat
pip install cupy-cuda11x
```
Refer to the official guide if unsure: https://docs.cupy.dev/en/stable/install.html

4) Quick verify
```bat
python -c "import cupy, tensorflow as tf; print('CuPy:', cupy.__version__); print('TF:', tf.__version__)"
```

Notes
- CuPy is currently required because several modules import it at top-level.
- If you later want CPU-only support, code changes are needed to make CuPy optional.

---

## Usage

### 1) Quick start (defaults)
```bat
python main.py
```
- Defaults: `--cipher present80`, `--rounds 7`, `--pairs 8`.
- The script will: select best delta key, generate data, train the model, and run evaluation.

### 2) Choose cipher dynamically
```bat
python main.py --cipher present80
python main.py --cipher simon3264
python main.py --cipher speck3264
python main.py --cipher speck64128
python main.py --cipher simmeck3264
python main.py --cipher simmeck4896
```
Alternatively via environment variable (cmd.exe):
```bat
set CIPHER_NAME=simon3264
python main.py
```

### 3) Override rounds and pairs
```bat
python main.py --rounds 9 --pairs 16
:: short forms
python main.py -r 9 -p 16
```
Or via environment variables:
```bat
set CIPHER_ROUNDS=9
set PAIRS=16
python main.py
```

### 4) Outputs and logs
- Checkpoints (per cipher and rounds):
  - `checkpoints/<cipher>/<cipher>_best_<rounds>r.keras`
  - `checkpoints/<cipher>/<cipher>_last_<rounds>r.weights.h5` (saved every epoch)
  - `checkpoints/<cipher>/<cipher>_final_<rounds>r.keras` (saved after training)
- Logs per run_id (timestamp):
  - TensorBoard: `logs/<cipher>/<run_id>/`
  - CSV: `logs/<cipher>/<run_id>/training_<rounds>r.csv`
  - History JSON: `logs/<cipher>/<run_id>/history_<rounds>r.json`

### 5) Notes
- For best performance, use an NVIDIA GPU with proper CUDA + CuPy installation.
- If CuPy/CUDA isn’t ready yet, consider disabling any GPU self-tests inside cipher modules (avoid running on import), or complete CUDA setup first.

---

## Citation

If you use this code for research, please cite the original authors or this repository.

---

## License

MIT License

---
