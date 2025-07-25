import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from make_data_train import NDCMultiPairGenerator
from train_nets import integer_to_binary_array

def evaluate_with_statistics(
    model,
    round_number,
    n_repeat=20,
    log_path=None,
    encryption_function=None,
    plain_bits=64,
    key_bits=80,
    input_difference=None,
    delta_key=None,
    pairs=8
):
    """
    Evaluate the model multiple times and compute statistics to check if the result is statistically significant.

    Parameters:
        model: Trained model.
        round_number: Number of PRESENT rounds for test data generation.
        n_repeat: Number of test set generations and evaluations (recommended: 20–30).
        log_path: If provided, save log results to this file.
        encryption_function: Cipher encryption function.
        plain_bits: Number of bits in plaintext.
        key_bits: Number of bits in key.
        input_difference: Input difference (int or array).
        delta_key: Key difference (array).
        pairs: Number of pairs per sample.

    Returns:
        dict: Contains avg_acc, std_acc, z_score, p_value.
    """
    print(f"\n📊 Evaluating model on {n_repeat} fresh test sets for round {round_number}...")

    if encryption_function is None or input_difference is None or delta_key is None:
        raise ValueError("encryption_function, input_difference, and delta_key must be provided.")

    test_accs = []
    for i in tqdm(range(n_repeat)):
        test_gen = NDCMultiPairGenerator(
            encryption_function=encryption_function,
            plain_bits=plain_bits,
            key_bits=key_bits,
            nr=round_number,
            delta_state=integer_to_binary_array(input_difference, plain_bits),
            delta_key=delta_key,
            n_samples=1000000,
            batch_size=10000,
            pairs=pairs,
            use_gpu=True
        )
        _, acc = model.evaluate(test_gen, verbose=0)
        test_accs.append(acc)

    # Compute statistics
    accs = np.array(test_accs)
    avg_acc = np.mean(accs)
    std_acc = np.std(accs)

    # Z-score: deviation from random guessing (50%)
    mean_random = 0.5
    std_random = 0.0005  # For each test set of 1 million samples
    std_mean = std_random / np.sqrt(n_repeat)
    z_score = (avg_acc - mean_random) / std_mean
    p_value = 1 - norm.cdf(z_score)

    print(f"\n✅ Average Accuracy: {avg_acc:.5f} ± {std_acc:.5f}")
    print(f"📐 Z-score = {z_score:.2f},  P-value = {p_value:.4e}")
    if p_value < 0.01:
        print("✨ Statistically significant improvement over random.")
    else:
        print("⚠️  Accuracy may still be due to random guessing.")

    # Save log if needed
    if log_path:
        with open(log_path, 'w') as f:
            for i, acc in enumerate(accs):
                f.write(f"Test {i+1}: {acc:.6f}\n")
            f.write(f"\nAverage: {avg_acc:.6f}, Std: {std_acc:.6f}\n")
            f.write(f"Z-score: {z_score:.2f}, P-value: {p_value:.4e}\n")

    return {
        'avg_acc': avg_acc,
        'std_acc': std_acc,
        'z_score': z_score,
        'p_value': p_value
    }
