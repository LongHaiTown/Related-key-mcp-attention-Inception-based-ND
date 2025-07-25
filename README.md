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

## Folder Structure

```
.
├── main.py                  # Main training and evaluation pipeline
├── train_nets.py            # Model, training utilities, and delta key selection
├── make_data_train.py       # Data generator for neural distinguisher
├── RKmcp.py                 # Inception-based model architecture with attention
├── eval_nets.py             # Evaluation and statistical analysis functions
├── cipher/                  # PRESENT-80 cipher implementation
├── requirements.txt         # zz
└── README.md                 
```

---

## Requirements

- Python zzz
- TensorFlow zz
- NumPy zz
- CuPy (optional, for NVIDIA GPU acceleration)
- scikit-learn zz
- tqdm zz
- scipy zz

Install dependencies:
```sh
pip install -r requirements.txt
```
If you have an NVIDIA GPU and CUDA installed, install the appropriate CuPy version (see [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)).  
If you **do not have an NVIDIA GPU**, set `use_gpu=False` in all generator calls.

---

## Usage

### 1. Train and Evaluate

Edit `main.py` if needed, then run:
```sh
python main.py
```
- The script will:
  - Select the best delta key automatically.
  - Generate training and validation data.
  - Train the neural distinguisher.
  - Evaluate the model and print statistical results.

### 2. Configuration

You can change parameters such as number of rounds, pairs, batch size, and epochs at the top of `main.py`.

### 3. Notes

- For best performance, use a machine with an NVIDIA GPU and CUDA.
- If you do not have a GPU, set `use_gpu=False` in `main.py` and `make_data_train.py`.
- Model checkpoints and logs will be saved automatically.

---

## Citation

If you use this code for research, please cite the original authors or this repository.

---

## License

MIT License

---
