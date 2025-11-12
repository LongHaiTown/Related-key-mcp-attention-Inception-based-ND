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

## Module API reference

This project now exposes modular helpers used by the orchestrator `finding_input.py`. Below is a concise reference of the new modules and their primary functions.

### finding_input.py (orchestrator)
- Purpose: Run a sweep over HW=1 input differences, compute PCA/KMeans metrics, log results to CSV/JSON, and optionally run a delta-key search for the top-K input differences.
- Run:
  ```bat
  python finding_input.py --cipher-module cipher.present80 --nr 7 --pairs 1 --datasize 50000 --clusters 27 --top-k 3 --dk-metric biased_pcs --dk-datasize 100000 --dk-batch-size 5000 --save-eigen-csv
  ```
- Outputs:
  - `logs/<cipher>/<timestamp>/config.json`
  - `logs/<cipher>/<timestamp>/sweep_results.csv`
  - `logs/<cipher>/<timestamp>/eigen_ratios_bit_XX.csv` (if `--save-eigen-csv`)
  - `logs/<cipher>/<timestamp>/best_delta_key_summary.csv` (if `--top-k > 0`)
  - `logs/<cipher>/<timestamp>/delta_key_scores_for_0x....csv` (per diff)

### cipher_utils.py
- `resolve_cipher_module(module_path: str) -> module`
  - Dynamically import a cipher module by dotted path and validate it exposes `encrypt`, `plain_bits`, and `key_bits`.
- `integer_to_binary_array(int_val: int, num_bits: int) -> (1, num_bits) array`
  - Convert integer to a 1×num_bits bit array. Uses CuPy when available, otherwise NumPy.
- `build_generator_for_diff(cipher_mod, input_diff_int: int, *, nr: int, pairs: int, n_samples: int, batch_size: int, seed: int = 42, use_gpu: bool = True)`
  - Construct `NDCMultiPairGenerator` for a given input difference with `delta_key = 0`.
- Constants: `_USING_CUPY: bool` indicates if CuPy is active.

### pca_utils.py
- `EigenValueDecomposition(dataset, alg=None, title=None, visualize_ratio='no') -> (explained_variance_ratio, components)`
  - StandardScaler + PCA fit; optionally visualizes variance.
- `DimensionReduction(dataset, n_components=3, alg=None, title=None)`
  - StandardScaler + PCA transform to n components.
- `Visualize2D(pca_results_2D, title=None)` and `Visualize3D(pca_results_3D, title=None)`
  - Optional scatter visualizations.

### cluster_utils.py
- `kmeans_clustering(data, num_clusters, n_init=10) -> labels`
  - KMeans clustering returning label array.
- `calculate_silhouette(data, labels) -> float`
  - Silhouette score for given labels.
- `visualize_clusters_2D(data, labels, title=None)` / `visualize_clusters_3D(data, labels, title=None)`
  - Optional cluster visualization helpers.

### sweep_input_diffs.py
- `sweep_input_differences(cipher_mod, *, nr: int, pairs: int, datasize: int, clusters: int, max_bits: Optional[int], use_gpu: bool, lambda_base: float, t0: float, results_csv: Optional[pathlib.Path] = None, save_eigen_csv: bool = False, eigen_dir: Optional[pathlib.Path] = None) -> List[Dict]`
  - Sweeps HW=1 input differences, computes metrics, and optionally writes CSVs. Returns a list of rows like:
    ```python
    {
      'bit_pos': int,
      'input_diff_hex': str,
      'biased_pcs': int,
      'max_diff': float,
      'silhouette_clusters': float,
      'silhouette_true': Optional[float],
      'elapsed_sec': float,
    }
    ```

### delta_key_search.py
- `select_best_delta_key(encryption_function, *, input_difference: int, plain_bits: int, key_bits: int, n_round: int, pairs: int, n_samples: int = 100_000, batch_size: int = 5_000, use_gpu: bool = True, random_seed: int = 42) -> (best_bit: int, best_score: float, all_scores: Dict[int, float])`
  - Searches HW=1 `delta_key` bits using StandardScaler + PCA(2D) + silhouette.
- `run_delta_key_search_for_topK(cipher_mod, picked_rows: List[Dict], *, nr: int, pairs: int, n_samples: int, batch_size: int, use_gpu: bool, random_seed: int, out_dir: pathlib.Path) -> pathlib.Path`
  - Orchestrates delta-key search across top-K diffs and writes:
    - `best_delta_key_summary.csv`
    - `delta_key_scores_for_<input_diff_hex>.csv` per diff

Notes
- All modules keep GPU optional. When CuPy isn’t available, code falls back to NumPy automatically.
- PCA and clustering are CPU-bound (scikit-learn). GPU mainly affects the generator stage if available.

---

## Citation

If you use this code for research, please cite the original authors or this repository.

---

## License

MIT License

---
