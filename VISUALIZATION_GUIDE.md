# Hướng dẫn sử dụng visualize_dataset.py

## Tổng quan

`visualize_dataset.py` là công cụ phân tích và trực quan hóa dataset cho neural distinguisher attacks sử dụng PCA và KMeans clustering. Tool này tập trung vào **related-key attacks với multi-pair configuration**.

## Tính năng chính

- ✅ **Auto-sweep input differences**: Tự động tìm input difference tốt nhất
- 📊 **PCA Analysis**: Phân tích principal components và explained variance
- 🎯 **KMeans Clustering**: Clustering với đánh giá silhouette score
- 📈 **Visualization**: Tạo các biểu đồ 2D, 3D, elbow curve
- 💾 **Export Results**: Lưu data, plots, và summary report
- ⏱️ **Performance Tracking**: Đo thời gian execution

## Cài đặt

### Yêu cầu

```bash
pip install numpy scikit-learn matplotlib
```

# Visualization Guide

This guide explains how to generate and analyze visualization artifacts (PCA + clustering) for neural distinguishers in this repository.

The main entrypoint is `visualize_dataset.py`, which creates a dataset via the standard generator, projects it with PCA, performs KMeans clustering, and saves figures and summary metrics.

## Quick Start

- Related-key with a combined difference (delta_plain + delta_key encoded together):

```powershell
python visualize_dataset.py --cipher present80 --scenario related-key --rounds 7 --pairs 8 --difference 0xa00000000000000000000000000000000 --samples 50000 --plot
```

- Related-key with plaintext-only delta and an explicit key bit:

```powershell
python visualize_dataset.py --cipher present80 --scenario related-key --rounds 7 --pairs 8 --difference 0x80 --key-bit 12 --samples 50000 --plot
```

- Single-key (no key delta):

```powershell
python visualize_dataset.py --cipher present80 --scenario single-key --rounds 7 --pairs 8 --difference 0x80 --samples 50000 --plot
```

If you do not pass `--difference`, the script will auto-sweep all HW=1 plaintext bit positions and select the best one based on `--sweep-metric` (default `silhouette_true`).

## Concepts

- **Scenario**:

  - `single-key`: Uses only a plaintext difference; key difference is forced to zero.
  - `related-key`: Supports both plaintext and key differences.
- **Pairs (`--pairs`)**: Number of (ΔC || C || C*) per sample. More pairs generally strengthen the statistical signal but increase dimensionality and runtime.
- **Rounds (`--rounds`)**: Cipher rounds used for data generation and evaluation.
- **Difference (`--difference`)**:

  - Plaintext-only: Any int/hex that fits in `plain_bits` (e.g., `0x80`).
  - Combined (related-key): Encode both deltas in a single integer:
    - Lower `plain_bits` → delta_plain
    - Next `key_bits` → delta_key
      The script auto-detects this format when the integer bit-length exceeds `plain_bits`.
- **Key Bit (`--key-bit`)**: For related-key, you can set a single key bit as the delta. Ignored in `single-key` mode.

## CLI Reference

```text
python visualize_dataset.py [options]

--cipher            Cipher under the `cipher/` package (e.g., present80, speck3264)
--scenario          single-key | related-key (default: related-key)
--rounds, -r        Number of cipher rounds (default: 7)
--pairs, -p         Pairs per sample (default: 8)
--difference        Input difference (hex or int). If omitted, auto-sweep picks the best HW=1 input bit.
--key-bit           Key difference as a single bit index (related-key only). Default: -1 (no key diff)
--samples           Dataset size (default: 100000)
--pca-components    PCA components (default: 16)
--kmeans-k          KMeans clusters (default: 2)
--sweep-metric      Sweep ranking metric: biased_pcs|max_diff|silhouette_clusters|silhouette_true
--sweep-samples     Samples per sweep candidate (default: 10000)
--max-bits          Limit sweep to first N plaintext bit positions
--plot              Save PCA/KMeans figures
--elbow-kmax        If >0, compute elbow curve for k=2..kmax
--out               Base output directory (default: analysis_results/)
--save-sweep        Save sweep CSV when sweep is used
--add-timestamp     Append timestamp to run folder
--log-file          Also write logs to a file
--verbose           Enable debug logging
--use-gpu           Use GPU if available (default: enabled)
```

## Input Difference Formats

- Plaintext-only example (64-bit PRESENT plaintext width): `0x0000000000000080`
- Combined example (PRESENT-80: 64-bit plaintext + 80-bit key):
  - `diff_int` lower 64 bits → delta_plain
  - `diff_int` next 80 bits  → delta_key
  - Example: `0xa00000000000000000000000000000000` (detected automatically as combined)

The helper will always truncate/mask any oversized integer to its target bit width to avoid shape errors.

## Outputs

Results are saved under:

```
analysis_results/<cipher>_<scenario>_r<rounds>_p<pairs>_<difference>[/_<timestamp>]/
```

Artifacts include:

- `pca_evr.png`                 — Explained variance ratio bar chart
- `pca_scatter_labels.png`      — PCA scatter colored by dataset labels (true)
- `pca_scatter_kmeans.png`      — PCA scatter colored by KMeans labels
- `kmeans_elbow.png`            — Optional elbow curve (when `--elbow-kmax > 2`)
- `pca_compare_3d.png`          — 3D PCA comparison true vs KMeans (when `--pca-components >= 3`)
- `projected_data.npy`          — PCA-projected data
- `eigenvalue_ratios.npy`       — PCA explained variance ratios
- `dataset_labels.npy`          — True labels (cipher vs random)
- `kmeans_labels.npy`           — KMeans cluster labels
- `summary.json`                — Run summary with parameters, metrics, timings
- `sweep_results.csv`           — When auto-sweep is used and `--save-sweep` is set

## Choosing a Good Difference

You have two paths:

1) Use the optimizer (`finding_input_new.py`) to discover strong differences.

   - Pick the best weighted difference from the generated CSV/logs.
   - Use it in visualize with `--difference` (and `--key-bit` if you separate plaintext/key deltas).
2) Let `visualize_dataset.py` auto-sweep input bits (HW=1) when `--difference` is empty.

   - It ranks candidates using `--sweep-metric` and selects the best.
   - Optionally save the full sweep via `--save-sweep`.

## Examples

- Auto-sweep, select best by silhouette on true labels, save plots and sweep CSV:

```powershell
python visualize_dataset.py --cipher present80 --scenario related-key --rounds 7 --pairs 8 --sweep-metric silhouette_true --sweep-samples 10000 --samples 100000 --plot --save-sweep
```

- Quick experiment with fewer samples and pairs:

```powershell
python visualize_dataset.py --cipher speck3264 --scenario single-key --rounds 7 --pairs 4 --difference 0x80 --samples 30000 --plot
```

## Notes & Troubleshooting

- Combined differences are supported: the script splits lower `plain_bits` to `delta_plain` and the next `key_bits` to `delta_key`.
- If you previously saw an error like "cannot reshape array of size ... into shape (1, 64)", it was caused by passing a combined difference without splitting; current code masks/splits safely.
- GPU usage is enabled by default (`--use-gpu`). If you encounter GPU/CuPy issues, you can disable GPU by removing `--use-gpu` (or setting it false if updated) and run on CPU.
- KMeans/PCA randomness: Results may vary slightly between runs due to random initialization. If deterministic runs are required, consider standardizing random seeds throughout the pipeline.

## Relationship to Optimizer (`finding_input_new.py`)

- Use the optimizer to search for promising input differences across rounds.
- For related-key, you can sweep `delta_key` single-bit positions to find the key bit that maximizes clustering separation (e.g., silhouette on true labels).
- Take the selected difference and (optionally) key bit into `visualize_dataset.py` to inspect PCA projections and clustering behavior in more detail.

### Ví dụ 4: Manual difference với specific key bit

```bash
python visualize_dataset.py \
    --cipher present80 \
    --rounds 7 \
    --pairs 8 \
    --difference 0x80 \
    --key-bit 107 \
    --samples 100000 \
    --plot \
    --out my_results \
    --add-timestamp
```

**Kết quả:**

- Không sweep, dùng trực tiếp 0x80
- Delta key = bit 107
- Lưu vào `my_results/` với timestamp

### Ví dụ 5: Batch analysis cho nhiều rounds

```bash
# Bash script
for rounds in 5 6 7 8; do
    python visualize_dataset.py \
        --cipher present80 \
        --rounds $rounds \
        --pairs 8 \
        --samples 50000 \
        --plot \
        --add-timestamp \
        --save-sweep
done
```

## Output files

### Folder structure

```
analysis_results/
└── present80_r7_p8_0x80_20251127-143052/
    ├── summary.json                    # Summary report
    ├── sweep_results.csv              # Sweep results (nếu dùng --save-sweep)
    ├── projected_data.npy             # PCA projected data
    ├── eigenvalue_ratios.npy          # Explained variance ratios
    ├── dataset_labels.npy             # True labels
    ├── kmeans_labels.npy              # KMeans predicted labels
    ├── pca_evr.png                    # Explained variance ratio plot
    ├── pca_scatter_labels.png         # PCA scatter với true labels
    ├── pca_scatter_kmeans.png         # PCA scatter với KMeans labels
    ├── kmeans_elbow.png               # Elbow curve (nếu dùng --elbow-kmax)
    └── pca_compare_3d.png             # 3D comparison plot
```

### summary.json format

```json
{
  "cipher": "present80",
  "rounds": 7,
  "pairs": 8,
  "input_difference": "0x0000000000000080",
  "input_diff_int": 128,
  "key_bit": 107,
  "samples": 100000,
  "pca_components": 16,
  "kmeans_k": 2,
  "use_gpu": true,
  "results": {
    "pca_explained_variance_ratio": [0.1234, 0.0987, ...],
    "kmeans_inertia": 12345.67,
    "kmeans_silhouette": 0.8765,
    "accuracy_vs_true": 0.9234,
    "adjusted_rand_index": 0.8567
  },
  "timing": {
    "total_seconds": 45.67,
    "data_generation_seconds": 12.34
  },
  "auto_sweep": {
    "used": true,
    "metric": "biased_pcs",
    "sweep_samples": 10000
  }
}
```

## Metrics giải thích

### PCA Metrics

- **Explained Variance Ratio**: Tỷ lệ variance được giải thích bởi mỗi PC
- **Cumulative Variance**: Tổng variance của k PCs đầu tiên

### Clustering Metrics

- **Inertia**: Tổng squared distances đến cluster centers (càng thấp càng tốt)
- **Silhouette Score**: Đo độ tách biệt clusters (-1 to 1, càng cao càng tốt)
- **Accuracy**: So sánh KMeans labels vs true labels (chỉ cho k=2)
- **Adjusted Rand Index (ARI)**: Đo similarity giữa hai clustering (0 to 1)

### Sweep Metrics

- **biased_pcs**: Số principal components bị biased (khác uniform)
- **max_diff**: Difference lớn nhất giữa eigenvalue và baseline
- **silhouette_clusters**: Silhouette score của KMeans clustering
- **silhouette_true**: Silhouette score với true labels

## Tips & Best Practices

### 1. Chọn số samples

- **Sweep**: 5k-10k samples (nhanh, đủ để rank)
- **Main analysis**: 50k-100k samples (chính xác hơn)
- **Publication**: 100k-1M samples (cao nhất)

### 2. Chọn sweep metric

- **biased_pcs**: Good default, đo non-uniformity
- **max_diff**: Focus vào strongest bias
- **silhouette_clusters**: Tốt cho clustering quality
- **silhouette_true**: Best cho distinguisher (cần true labels)

### 3. GPU vs CPU

- GPU: Nhanh hơn 10-50x cho large datasets
- CPU: Ổn định hơn, ít memory hơn
- Nếu GPU OOM → giảm `--samples` hoặc dùng `--use-gpu false`

### 4. Batch size optimization

- Tool tự động cap batch_size ≤ 100k
- Nếu vẫn OOM → giảm `--samples`
- Check available GPU memory trước

### 5. Output organization

- Dùng `--add-timestamp` để tránh ghi đè
- Dùng `--out custom_dir` để organize experiments
- Dùng `--save-sweep` để track sweep history

## Troubleshooting

### Error: Out of memory (GPU)

```bash
# Solution 1: Giảm samples
python visualize_dataset.py --samples 50000 ...

# Solution 2: Dùng CPU
python visualize_dataset.py --use-gpu false ...

# Solution 3: Giảm sweep samples
python visualize_dataset.py --sweep-samples 5000 ...
```

### Error: Cipher module not found

```bash
# Check cipher module exists
ls cipher/present80.py

# Ensure cipher package is importable
export PYTHONPATH="${PYTHONPATH}:."
```

### Error: Invalid difference value

```bash
# Sử dụng hex format với 0x prefix
--difference 0x80    # Correct
--difference 128     # Also correct
--difference 80      # Wrong (missing 0x)
```

### Warning: Low silhouette score

- Có thể difference không tốt
- Thử sweep với metric khác
- Tăng số samples
- Thử rounds khác

## Advanced Usage

### 1. Parallel sweeps cho nhiều ciphers

```bash
# parallel_sweep.sh
for cipher in present80 speck3264 simon3264; do
    python visualize_dataset.py \
        --cipher $cipher \
        --rounds 7 \
        --pairs 8 \
        --samples 100000 \
        --plot \
        --save-sweep \
        --add-timestamp &
done
wait
```

### 2. Compare results với Python script

```python
import json
import numpy as np
from pathlib import Path

results_dir = Path("analysis_results")
summaries = []

for summary_file in results_dir.glob("*/summary.json"):
    with open(summary_file) as f:
        summaries.append(json.load(f))

# Compare silhouette scores
for s in sorted(summaries, key=lambda x: x['results']['kmeans_silhouette'], reverse=True):
    print(f"{s['cipher']} r{s['rounds']}: sil={s['results']['kmeans_silhouette']:.4f}")
```

### 3. Load và analyze saved data

```python
import numpy as np

# Load saved PCA projection
proj = np.load("analysis_results/present80_r7_p8_0x80/projected_data.npy")
labels = np.load("analysis_results/present80_r7_p8_0x80/dataset_labels.npy")

# Further analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
proj3d = pca.fit_transform(proj)

# Custom visualization
import matplotlib.pyplot as plt
plt.scatter(proj3d[:, 0], proj3d[:, 1], c=labels, alpha=0.5)
plt.show()
```

## Tích hợp với workflow

### 1. Tìm best difference cho cipher/rounds mới

```bash
python visualize_dataset.py \
    --cipher new_cipher \
    --rounds 8 \
    --pairs 8 \
    --save-sweep \
    --verbose
```

→ Check `sweep_results.csv` để xem tất cả differences

### 2. Validate difference trước khi train

```bash
python visualize_dataset.py \
    --cipher present80 \
    --rounds 7 \
    --difference 0x80 \
    --samples 100000 \
    --plot
```

→ Check silhouette score > 0.5 thì OK

### 3. Debug training issues

Nếu model không học được:

1. Check PCA plots → có separable không?
2. Check silhouette score → clustering quality?
3. Try different input difference
4. Try more pairs
