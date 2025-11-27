# Related-key MCP Attention Inception-based Neural Distinguisher

This repository provides code for training and evaluating neural distinguishers for the PRESENT-80 block cipher using related-key, multi-pair, attention, and inception-based deep learning architectures.

---

## Features

- Neural distinguisher for PRESENT-80 using Inception and Attention modules.
- Related-key and multi-pair data generation.
- Automatic selection of the best delta key for related-key cryptanalysis.
- Manual override of delta-key bit via `--delta-key-bit`.
- Training pipeline with Keras/TensorFlow, supporting both GPU (with CuPy) and CPU (with NumPy).
- Evaluation scripts with statistical significance testing.
- Input-difference sweep (PCA/KMeans metrics) and delta-key search tools with CSV logging.
- Multi-stage (curriculum) training via `staged_train.py` (progressive rounds / LRs / epochs).

---

## Folder structure

```
.
├── main.py                  # Main training/evaluation; can auto-pick input diff from sweep CSV
├── finding_input.py         # Sweep HW=1 input differences (PCA/KMeans) and write CSVs
├── finding_delta_key.py     # Delta-key search on top-K input diffs from a sweep CSV
├── train_nets.py            # Model, training utilities, and delta key selection
├── make_data_train.py       # Data generator for neural distinguisher
├── RKmcp.py                 # Inception-based model architecture with attention
├── eval_nets.py             # Evaluation and statistical analysis functions
├── utils/                   # Helper modules: cipher_utils, pca_utils, cluster_utils
├── docs/
│   └── module_apis.md       # Detailed module API reference
├── cipher/                  # PRESENT-80 cipher implementation
├── requirements.txt         # Python dependencies
└── README.md                 
```

---

## Requirements and setup (Windows)

This project uses TensorFlow/Keras and optionally CuPy. On Windows, use a compatible CUDA toolkit to enable CuPy for GPU acceleration. Steps below assume Command Prompt (cmd.exe).

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

3) Install CuPy matching your CUDA version (RECOMMENDED for GPU)
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
- CuPy is optional; code falls back to NumPy when CuPy/CUDA is unavailable (some runs will be slower on CPU).
- Ensure your NVIDIA drivers and CUDA runtime match the CuPy package you install.

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

### 4) Use sweep results to auto-pick input difference (no re-sweep)
If you already ran a sweep, let `main.py` choose the best input difference from its CSV:
```bat
python main.py ^
  --cipher present80 ^
  --rounds 7 ^
  --pairs 8 ^
  --sweep-csv "differences_findings\logs\present80\<timestamp>\sweep_results.csv" ^
  --diff-metric biased_pcs
```
Fallback (manual input difference):
```bat
python main.py --cipher present80 --rounds 7 --pairs 8 --input-diff 0x00000080
```

Optional: manually fix the delta-key bit (skip automatic search):
```bat
python main.py --cipher present80 --rounds 7 --pairs 8 --input-diff 0x00000080 --delta-key-bit 107
```
If provided, `--delta-key-bit` bypasses delta-key silhouette scoring.

### 5) Input-difference sweep and delta-key search (standalone tools)
Run sweep only (PCA/KMeans + CSV logging):
```bat
python finding_input.py ^
  --cipher-module cipher.present80 ^
  --nr 7 ^
  --pairs 1 ^
  --datasize 50000 ^
  --clusters 27 ^
  --max-bits 64
```
This produces `sweep_results.csv` and additional sorted views:
- `sweep_sorted_by_biased_pcs.csv`
- `sweep_sorted_by_max_diff.csv`
- `sweep_sorted_by_silhouette_clusters.csv`
- `sweep_sorted_by_silhouette_true.csv`

Sweep multiple rounds in one go (outputs go into nr<round> subfolders):
```bat
python finding_input.py ^
  --cipher-module cipher.present80 ^
  --nr-sweep 5:9 ^
  --pairs 1 ^
  --datasize 50000 ^
  --clusters 27 ^
  --max-bits 64
```
python finding_input.py --cipher-module cipher.simmeck64128 --nr-sweep 10:20 --pairs 1 --datasize 50000 --clusters 27 --max-bits 64

Run delta-key search only (from a sweep CSV):
```bat
python finding_delta_key.py ^
  --cipher-module cipher.present80 ^
  --nr 7 ^
  --pairs 1 ^
  --sweep-csv "differences_findings\logs\present80\<timestamp>\sweep_results.csv" ^
  --top-k 5 ^
  --dk-metric biased_pcs ^
  --dk-datasize 100000 ^
  --dk-batch-size 5000
```
Or auto-pick the latest sweep CSV:
```bat
python finding_delta_key.py ^
  --cipher-module cipher.present80 ^
  --nr 7 ^
  --pairs 1 ^
  --auto-latest ^
  --top-k 5 ^
  --dk-metric biased_pcs
```

Outputs are written under `differences_findings/logs/<cipher>/<timestamp>/` unless `--out-dir` is specified.

### 6.5) Auto-select best round + input difference from multi-round sweep
After generating a multi-round sweep with `finding_input.py --nr-sweep ...`, you can let `main.py` automatically pick:
```bat
python main.py ^
  --cipher present80 ^
  --auto-latest-sweep ^
  --pairs 8 ^
  --diff-metric biased_pcs
```
Or specify a particular sweep parent directory:
```bat
python main.py ^
  --cipher present80 ^
  --sweep-parent "differences_findings\logs\present80\20251113-101500" ^
  --pairs 8 ^
  --diff-metric max_diff
```
This overrides `--rounds` with the best round discovered and uses that round's best input difference.

### 6) Outputs and logs
- Checkpoints (per cipher and rounds):
  - `checkpoints/<cipher>/<cipher>_best_<rounds>r.keras`
  - `checkpoints/<cipher>/<cipher>_last_<rounds>r.weights.h5` (saved every epoch)
  - `checkpoints/<cipher>/<cipher>_final_<rounds>r.keras` (saved after training)
- Logs per run_id (timestamp):
  - TensorBoard: `logs/<cipher>/<run_id>/`
  - CSV: `logs/<cipher>/<run_id>/training_<rounds>r.csv`
  - History JSON: `logs/<cipher>/<run_id>/history_<rounds>r.json`
 - Sweep and delta-key outputs: `differences_findings/logs/<cipher>/<timestamp>/`

### 6.7) Staged (Curriculum) Training

Use `staged_train.py` to incrementally train the distinguisher across multiple cipher round depths and learning rates. This is helpful when deeper-round training is unstable from scratch.

Example:
```bat
python staged_train.py ^
  --cipher present80 ^
  --pairs 8 ^
  --input-diff 0x00000080 ^
  --delta-key-bit 107 ^
  --stages-rounds 5,6,7 ^
  --stages-epochs 15,10,10 ^
  --stages-lrs 1e-3,5e-4,1e-4 ^
  --stages-train-samples 800000,1000000,1200000 ^
  --stages-val-samples 200000,250000,300000 ^
  --batch-size 5000 ^
  --val-batch-size 20000 ^
  --use-gpu ^
  --save-final
```

Key points:
- Model is built once; rounds vary per stage through data generator.
- Learning rate is adjusted before each stage (`Adam` optimizer reused).
- Stage artifacts: `<cipher>_stage<i>_<rounds>r_last.weights.h5` in `staged_runs/`.
- Final artifacts: `<cipher>_staged_final.weights.h5` and optionally `<cipher>_staged_final.keras`.
- Combined JSON history: `<cipher>_staged_history.json` (list of per-stage histories).

Tune curriculum:
- Start with fewer rounds & higher LR, then increase rounds while reducing LR.
- Adjust samples upward to stabilize deeper stages.

Minimal run (defaults 3 stages of rounds=7):
```bat
python staged_train.py --cipher present80 --input-diff 0x00000080 --delta-key-bit 107 --use-gpu
```

**Resume from pre-trained model/weights:**

Start from a full .keras model:
```bat
python staged_train.py ^
  --cipher present80 ^
  --pairs 8 ^
  --input-diff 0x00000080 ^
  --delta-key-bit 107 ^
  --stages-rounds 6,7,8 ^
  --stages-epochs 10,8,8 ^
  --stages-lrs 5e-4,1e-4,5e-5 ^
  --init-model checkpoints\present80\present80_final_7r.keras ^
  --use-gpu
```

Start from weights only:
```bat
python staged_train.py ^
  --cipher present80 ^
  --pairs 8 ^
  --input-diff 0x00000080 ^
  --delta-key-bit 107 ^
  --stages-rounds 7,8 ^
  --stages-epochs 12,8 ^
  --stages-lrs 5e-4,1e-4 ^
  --init-weights checkpoints\present80\present80_last_7r.weights.h5 ^
  --use-gpu
```
python staged_train.py --cipher present80 --pairs 8 --input-diff 0x00000080 --delta-key-bit 25 --stages-rounds 6,7 --stages-epochs 12,8 --stages-lrs 5e-4,1e-4 --init-weights checkpoints\present80\present80_last_7r.weights.h5 --use-gpu

Notes:
- `--init-model` loads a full .keras model (takes priority if both are provided).
- `--init-weights` loads only weights into a fresh model architecture.
- Ensure `--pairs` and cipher match the pre-trained model architecture.
- Learning rate is reset to the first stage's LR value.

  ### Evaluate a trained model
  Use `eval_nets.py` to run repeated-test evaluation with statistical reporting (Z-score, p-value). Requires a saved `.keras` model file and runs the generator with GPU when available.

  ```bat
  python eval_nets.py ^
    --cipher-module cipher.present80 ^
    --model-path checkpoints\present80\present80_final_7r.keras ^
    --rounds 7 ^
    --pairs 8 ^
    --input-diff 0x00000080 ^
    --delta-key-bit 107 ^
    --n-repeat 20 ^
    --test-samples 1000000 ^
    --batch-size 10000
  ```

  <!-- python eval_nets.py --cipher-module cipher.present80 --model-path checkpoints\present80\present80_best_8r.weights.h5 --rounds 8 --pairs 8 --input-diff 0x00000080 --delta-key-bit 56 --n-repeat 2 --test-samples 1000000 --batch-size 10000 -->

  If your delta-key is a bitmask (multi-bit), pass it as hex:
  ```bat
  python eval_nets.py ^
    --cipher-module cipher.present80 ^
    --model-path checkpoints\present80\present80_final_7r.keras ^
    --rounds 7 ^
    --pairs 8 ^
    --input-diff 0x00000080 ^
    --delta-key-hex 0x00000000000000000001 ^
    --n-repeat 30
  ```

### 7) Notes
- For best performance, use an NVIDIA GPU with proper CUDA + CuPy installation.
- If CuPy/CUDA isn’t ready yet, consider disabling any GPU self-tests inside cipher modules (avoid running on import), or complete CUDA setup first.

---

## Citation

If you use this code for research, please cite the original authors or this repository.

---

## License

MIT License

---
