import numpy as np
import os
from random import randint
import sys
from os import urandom
from copy import deepcopy
from types import FunctionType
from timeit import default_timer as timer
import numpy as np
import pandas as pd

NUM_GENERATIONS = 5    # 50 in the paper, set to 5 here for demonstration
NUM_SAMPLES = 10**3     # 10**4 in the paper. The number of samples used to compute the bias score

def bitArrayToIntegers(arr):
    packed = np.packbits(arr,  axis = 1)
    return [int.from_bytes(x.tobytes(), 'big') for x in packed]


def empirical_threshold_estimation(n, plain_bits):
    bits = np.random.randint(2, size = (1000,n,plain_bits))
    scores = np.average(np.abs(0.5-np.average(bits, axis = 1)), axis=1)
    return scores

# Computes the bias scores of several candidate_differences, based on the initial plaintexts and keys pt0, keys and the corresponding ciphertexts C0, for nr rounds of a cipher with plain_bits plaintext bits and key_bits key bits.
def evaluate_multiple_differences(candidate_differences, pt0, keys, C0, nr, plain_bits, key_bits, encrypt, scenario = "single-key"):
    dp = candidate_differences[:, :plain_bits]
    pt1   = (np.broadcast_to(dp[:, None, :], (len(candidate_differences), len(pt0), plain_bits))^pt0).reshape(-1, plain_bits)
    if scenario == "related-key":
        dk = candidate_differences[:, plain_bits: ]
    else:
        dk = np.zeros((len(candidate_differences), key_bits), dtype=np.uint8)
    keys1 = (np.broadcast_to(dk[:, None, :], (len(candidate_differences), len(pt0), key_bits))^keys).reshape(-1, key_bits)
    C1 = encrypt(pt1, keys1, nr)
    differences_in_output =  C1.reshape(len(candidate_differences), len(pt0),-1)^C0
    scores = np.average(np.abs(0.5-np.average(differences_in_output, axis = 1)), axis=1)
    zero_diffs = np.where(np.sum(candidate_differences, axis = 1)==0)
    scores[zero_diffs] = 0
    return scores

# Evolutionary algorithm based on the encryption function f, running for n generations, using differences of num_bits bits, a population size of L, an optional initial population gen, and verbosity set to 0 for silent or 1 for verbose.
def evo(f, n=NUM_GENERATIONS, num_bits=32, L = 32, gen=None, verbose = 0, method='evo',rounds=7):
    
    # Keep original evolutionary algorithm as default
    if method == 'evo':
        mutProb = 100
        if gen is None:
            gen = np.random.randint(2, size = (L**2, num_bits), dtype=np.uint8)
        scores = f(gen)
        idx = np.arange(len(gen))
        explored = np.copy(gen)
        good = idx[np.argsort(scores)][-L:]
        gen = gen[good]
        scores = scores[good]
        cpt = len(gen)
        for generation in range(n):
            # New generation
            kids = np.array([gen[i] ^ gen[j] for i in range(len(gen)) for j in range(i+1, len(gen))], dtype = np.uint8);
            # Mutation: selecting mutating kids
            selected = np.where(np.random.randint(0,100, len(kids))>(100-mutProb))
            numMut = len(selected[0])
            # Selected kids are XORed with 1<<r (r random)
            tmp = kids[selected].copy()
            kids[selected[0].tolist(), np.random.randint(num_bits, size = numMut)] ^=1
            # Removing kids that have been explored before and duplicates
            if len(explored) > 0:
                kids = np.unique(kids[(kids[:, None] != explored).any(-1).all(-1)], axis=0)
            # Appending to explored
            explored = np.vstack([explored, kids]) if len(kids)>0 else explored
            cpt+=len(kids)
            # Computing the scores
            if len(kids)>0:
                scores = np.append(scores, f(kids))
                gen = np.vstack([gen, kids])
             # Sorting, keeping only the L best ones
                idx = np.arange(len(gen))
                good = idx[np.argsort(scores)][-L:]
                gen = gen[good]
                scores = scores[good]
            if verbose:
                genInt = np.packbits(gen[-4:, :],  axis = 1)
                hexGen = [hex(int.from_bytes(x.tobytes(), 'big')) for x in genInt]
                print(f'Generation {generation}/{n}, {cpt} nodes explored, {len(gen)} current, best is {[x for x in hexGen]} with {scores[-4:]}', flush=True)
            if np.all(scores == 0.5):
                break
        return gen, scores

    # Delegate to separate heuristic functions when requested
    HEURISTICS = {
        'pso': _pso_heuristic,
        'ga': _ga_heuristic,
        'de': _de_heuristic,
        'aco': _aco_heuristic,
        'sa': _sa_heuristic,
        'gwo': _gwo_heuristic
    }
    if method in HEURISTICS:
        return HEURISTICS[method](f=f, n=n, num_bits=num_bits, L=L, gen=gen, verbose=verbose, rounds=rounds)



def _top_k_from_population(population, scores_arr, k=32):
    idx = np.argsort(scores_arr)[-k:]
    return population[idx], scores_arr[idx]


def _pso_heuristic(f, n=NUM_GENERATIONS, num_bits=32, L=32, gen=None, verbose=0, rounds=7, **kwargs):
    if gen is None:
        pop = np.random.randint(2, size=(L, num_bits), dtype=np.uint8)
    else:
        pop = np.array(gen, dtype=np.uint8)
    num_particles = L
    particles = np.array(pop[:num_particles], dtype=np.uint8)
    velocities = np.zeros((num_particles, num_bits), dtype=float)
    pbest = particles.copy()
    pbest_scores = f(particles)
    gbest_idx = int(np.argmax(pbest_scores))
    gbest = pbest[gbest_idx].copy()

    w, c1, c2 = 0.5, 1.0, 1.5
    for it in range(n):
        scores_now = f(particles)
        for i in range(num_particles):
            velocities[i] = w * velocities[i] + \
                            c1 * np.random.rand(num_bits) * (pbest[i].astype(float) - particles[i].astype(float)) + \
                            c2 * np.random.rand(num_bits) * (gbest.astype(float) - particles[i].astype(float))
            sigmoid = 1 / (1 + np.exp(-velocities[i]))
            particles[i] = (np.random.rand(num_bits) < sigmoid).astype(np.uint8)

            score = scores_now[i]
            if score > pbest_scores[i]:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = score

        gbest_idx = int(np.argmax(pbest_scores))
        gbest = pbest[gbest_idx].copy()
        if verbose:
            print(f'[PSO] Iter {it+1}/{n} | Best: {pbest_scores[gbest_idx]:.6f}')

    final_pop, final_scores = _top_k_from_population(pbest, pbest_scores, k=min(L, len(pbest_scores)))
    return final_pop, final_scores


def _ga_heuristic(f, n=NUM_GENERATIONS, num_bits=32, L=32, gen=None, verbose=0, rounds=7, **kwargs):
    if gen is None:
        population = np.random.randint(2, size=(L, num_bits), dtype=np.uint8)
    else:
        population = np.array(gen, dtype=np.uint8)
    scores = f(population)
    for gen_idx in range(n):
        sorted_idx = np.argsort(scores)[::-1]
        population = population[sorted_idx]
        scores = scores[sorted_idx]
        next_gen = list(population[:max(2, L//4)])
        while len(next_gen) < L:
            p1, p2 = population[np.random.randint(0, min(len(population), max(2, L//2)), 2)]
            cp = np.random.randint(1, num_bits)
            child = np.concatenate([p1[:cp], p2[cp:]])
            if np.random.rand() < 0.1:
                mutation_point = np.random.randint(0, num_bits)
                child[mutation_point] ^= 1
            next_gen.append(child)
        population = np.array(next_gen, dtype=np.uint8)
        scores = f(population)
        if verbose:
            print(f'[GA] Gen {gen_idx+1}/{n} | Best: {scores.max():.6f}')
    final_pop, final_scores = _top_k_from_population(population, scores, k=min(L, len(scores)))
    return final_pop, final_scores


def _de_heuristic(f, n=NUM_GENERATIONS, num_bits=32, L=32, gen=None, verbose=0, rounds=7, **kwargs):
    if gen is None:
        population = np.random.randint(2, size=(L, num_bits), dtype=np.uint8)
    else:
        population = np.array(gen, dtype=np.uint8)
    scores = f(population)
    F = 0.5
    CR = 0.7
    for gen_idx in range(n):
        for i in range(len(population)):
            idxs = [j for j in range(len(population)) if j != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = (a.astype(float) + F * (b.astype(float) - c.astype(float)))
            trial = population[i].copy()
            cross_points = np.random.rand(num_bits) < CR
            trial[cross_points] = (mutant[cross_points] > 0.5).astype(np.uint8)
            score_trial = f(trial.reshape(1, -1))[0]
            if score_trial > scores[i]:
                population[i] = trial
                scores[i] = score_trial
        if verbose:
            print(f'[DE] Gen {gen_idx+1}/{n} | Best: {scores.max():.6f}')
    final_pop, final_scores = _top_k_from_population(population, scores, k=min(L, len(scores)))
    return final_pop, final_scores


def _aco_heuristic(f, n=NUM_GENERATIONS, num_bits=32, L=32, gen=None, verbose=0, rounds=7, **kwargs):
    if gen is None:
        # initial population not required for ACO
        pass
    pheromone = np.ones(num_bits)
    best_score = 0.0
    best_solution = None
    for it in range(n):
        solutions = (np.random.rand(L, num_bits) < (pheromone / pheromone.max())).astype(np.uint8)
        scores = f(solutions)
        idx_best = int(np.argmax(scores))
        if scores[idx_best] > best_score:
            best_score = scores[idx_best]
            best_solution = solutions[idx_best].copy()
        pheromone = (1 - 0.1) * pheromone + 0.1 * solutions[idx_best]
        if verbose:
            print(f'[ACO] Iter {it+1}/{n} | Best: {best_score:.6f}')
    final_pop = np.vstack([best_solution for _ in range(L)])
    final_scores = np.array([best_score for _ in range(L)])
    return final_pop, final_scores


def _sa_heuristic(f, n=NUM_GENERATIONS, num_bits=32, L=32, gen=None, verbose=0, rounds=7, **kwargs):
    if gen is None:
        pop = np.random.randint(2, size=(L, num_bits), dtype=np.uint8)
    else:
        pop = np.array(gen, dtype=np.uint8)
    current = pop[0].copy()
    current_score = f(current.reshape(1, -1))[0]
    best = current.copy()
    best_score = current_score
    T = 1.0
    cooling = 0.99
    for it in range(n * 10):
        neighbor = current.copy()
        idx = np.random.randint(0, num_bits)
        neighbor[idx] ^= 1
        neighbor_score = f(neighbor.reshape(1, -1))[0]
        if neighbor_score > current_score or np.random.rand() < np.exp((neighbor_score - current_score) / T):
            current = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        T *= cooling
        if verbose and (it % max(1, (n*10)//10) == 0):
            print(f'[SA] Iter {it+1}/{n*10} | Best: {best_score:.6f}')
    final_pop = np.vstack([best for _ in range(L)])
    final_scores = np.array([best_score for _ in range(L)])
    return final_pop, final_scores


def _gwo_heuristic(f, n=NUM_GENERATIONS, num_bits=32, L=32, gen=None, verbose=0, rounds=7, **kwargs):
    if gen is None:
        wolves = np.random.randint(2, size=(L, num_bits), dtype=np.uint8)
    else:
        wolves = np.array(gen, dtype=np.uint8)
    scores = f(wolves)
    for it in range(n):
        sorted_idx = np.argsort(scores)[::-1]
        wolves = wolves[sorted_idx]
        scores = scores[sorted_idx]
        alpha, beta, delta = wolves[:3]
        a = 2 - 2 * (it / max(1, n))
        new_wolves = wolves.copy()
        for i in range(len(wolves)):
            r1, r2 = np.random.rand(num_bits), np.random.rand(num_bits)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            D_alpha = np.abs(C1 * alpha.astype(float) - wolves[i].astype(float))
            X1 = (alpha.astype(float) - A1 * D_alpha) > 0.5

            r1, r2 = np.random.rand(num_bits), np.random.rand(num_bits)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            D_beta = np.abs(C2 * beta.astype(float) - wolves[i].astype(float))
            X2 = (beta.astype(float) - A2 * D_beta) > 0.5

            r1, r2 = np.random.rand(num_bits), np.random.rand(num_bits)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            D_delta = np.abs(C3 * delta.astype(float) - wolves[i].astype(float))
            X3 = (delta.astype(float) - A3 * D_delta) > 0.5

            new_wolves[i] = ((X1.astype(np.uint8) + X2.astype(np.uint8) + X3.astype(np.uint8)) / 3.0 > 0.5).astype(np.uint8)
        wolves = new_wolves
        scores = f(wolves)
        if verbose:
            print(f'[GWO] Iter {it+1}/{n} | Best: {scores.max():.6f}')
    final_pop, final_scores = _top_k_from_population(wolves, scores, k=min(L, len(scores)))
    return final_pop, final_scores


def DataframeFromSortedDifferences(differences, scores, scenario, plain_bits, key_bits=0):
    idx = np.arange(len(differences))
    good = idx[np.argsort(scores)]
    sorted_diffs = differences[good]
    sorted_scores = scores[good].round(4)
    diffs_to_print = bitArrayToIntegers(sorted_diffs)
    data = []
    for idx, d in enumerate(diffs_to_print):
        if scenario == "related-key":
            data.append([({hex(d>>key_bits)}, {hex(d&(2**key_bits-1))}), {sorted_scores[idx]}])
        else:
            data.append([{hex(d)}, {sorted_scores[idx]}])
    df = pd.DataFrame(data, columns=['Difference', 'Weighted score'])
    return df

def PrettyPrintBestEpsilonCloseDifferences(differences, scores, epsilon, scenario, plain_bits, key_bits=0):
    idx = np.arange(len(differences))
    order = idx[np.argsort(scores)]
    sorted_diffs = differences[order]
    sorted_scores = scores[order].round(4)
    best_score = sorted_scores[-1]
    threshold = best_score*(1-epsilon)
    keep = np.where(sorted_scores>threshold)
    diffs_to_print = bitArrayToIntegers(sorted_diffs[keep])
    scores_to_print = sorted_scores[keep]
    resStr = ''
    for idx, d in enumerate(diffs_to_print):
        if scenario == "related-key":
            resStr = resStr + f'[{hex(d)} ({hex(d>>key_bits)}, {hex(d&(2**key_bits-1))}), {scores_to_print[idx]}]\n'
        else:
            resStr = resStr + f'[{hex(d)}, {scores_to_print[idx]}]\n'
    return resStr, sorted_diffs[keep], diffs_to_print


def PrettyPrintBestNDifferences(differences, scores, n, scenario, plain_bits, key_bits=0):
    idx = np.arange(len(differences))
    good = idx[np.argsort(scores)]
    sorted_diffs = differences[good]
    sorted_scores = scores[good].round(4)[-n:]
    diffs_to_print = bitArrayToIntegers(sorted_diffs)[-n:]
    resStr = ''
    for idx, d in enumerate(diffs_to_print):
        if scenario == "related-key":
            resStr = resStr + f'[{hex(d)} ({hex(d>>key_bits)}, {hex(d&(2**key_bits-1))}), {sorted_scores[idx]}]\n'
        else:
            resStr = resStr + f'[{hex(d)}, {sorted_scores[idx]}]\n'
    return resStr, sorted_diffs[-n:], diffs_to_print


def _wrap_encrypt_for_numpy(encryption_function):
    """
    Wrap cipher encrypt to accept NumPy arrays and return NumPy arrays.
    If CuPy is available/required by cipher, convert inputs to CuPy and back.
    """
    try:
        import cupy as cp
        def _enc_numpy(pt, keys, nr):
            cpt = cp.asarray(pt)
            ckeys = cp.asarray(keys)
            cts = encryption_function(cpt, ckeys, nr)
            try:
                return cp.asnumpy(cts)
            except Exception:
                # If encryption already returns NumPy
                return np.array(cts)
        return _enc_numpy
    except Exception:
        # Fallback: assume encryption_function works with NumPy directly
        def _enc_numpy(pt, keys, nr):
            return encryption_function(pt, keys, nr)
        return _enc_numpy


def optimize(plain_bits, key_bits, encryption_function, nb_samples=NUM_GENERATIONS, scenario = "single-key", log_file = None, epsilon=0.1, method='evo', rounds=7):
    allDiffs = None
    totalScores = {}
    diffs = None
    T = 0.05 # The bias score threshold
    current_round = 1
    if scenario == "single-key":
        bits_to_search = plain_bits
    else:
        bits_to_search = plain_bits+key_bits
    # Loop over cipher rounds up to `rounds`, stop early if scores drop below threshold
    highest_non_random_round = 0
    # Ensure encrypt works with NumPy regardless of cipher backend
    encrypt_np = _wrap_encrypt_for_numpy(encryption_function)
    for current_round in range(1, rounds + 1):
        print("Evaluating differences at round ", current_round)
        keys0 = (np.frombuffer(urandom(nb_samples*key_bits),dtype=np.uint8)&1).reshape(nb_samples, key_bits)
        pt0 = (np.frombuffer(urandom(nb_samples*plain_bits),dtype=np.uint8)&1).reshape(nb_samples, plain_bits)
        C0 = encrypt_np(pt0, keys0, current_round)
        # The initial set of differences can be set to None, or to the differences returned for the previous round. We use the second option here.
        diffs, scores = evo(
            f=lambda x: evaluate_multiple_differences(x, pt0, keys0, C0, current_round, plain_bits, key_bits, encrypt_np, scenario=scenario),
            num_bits=bits_to_search, L=32, gen=diffs, verbose=1, method=method, rounds=rounds)
        if allDiffs is None:
            allDiffs = diffs
        else:
            allDiffs = np.concatenate([allDiffs, diffs])
        if len(scores) == 0 or scores[-1] < T:
            highest_non_random_round = current_round - 1
            break
        highest_non_random_round = current_round

    # Reevaluate all differences for best round:
    finalScores = {i:None for i in range(1, current_round)}
    allDiffs = np.unique(allDiffs, axis=0)
    cumulativeScores = np.zeros(len(allDiffs))
    weightedScores = np.zeros(len(allDiffs))
    if log_file != None:
        with open(log_file, 'a') as f:
            f.write(f'New log start, reached round {str(highest_non_random_round)} \n')
    for nr in range(1, current_round):
        keys0 = (np.frombuffer(urandom(nb_samples*key_bits),dtype=np.uint8)&1).reshape(nb_samples, key_bits)
        pt0 = (np.frombuffer(urandom(nb_samples*plain_bits),dtype=np.uint8)&1).reshape(nb_samples, plain_bits)
        C0 = encrypt_np(pt0, keys0, nr)
        finalScores[nr] = evaluate_multiple_differences(allDiffs, pt0, keys0, C0, nr, plain_bits, key_bits, encrypt_np, scenario = scenario)
        cumulativeScores += np.array(finalScores[nr])
        weightedScores += nr*np.array(finalScores[nr])

        result, _, _ = PrettyPrintBestNDifferences(allDiffs, finalScores[nr], 5, scenario, plain_bits, key_bits)
        resStr = f'Best at {nr}: \n{result}'
        if log_file != None:
            with open(log_file, 'a') as f:
                f.write(resStr)

    result, _, _ = PrettyPrintBestNDifferences(allDiffs, cumulativeScores, 5, scenario, plain_bits, key_bits)
    resStr = f'Best Cumulative: \n{result}'
    if log_file != None:
        with open(log_file, 'a') as f:
            f.write(resStr)


    result, _, _ = PrettyPrintBestNDifferences(allDiffs, weightedScores, 5, scenario, plain_bits, key_bits)
    resStr = f'Best Weighted: \n{result}'
    if log_file != None:
        with open(log_file, 'a') as f:
            f.write(resStr)

    result, diffs_as_binary, diffs_as_hex = PrettyPrintBestEpsilonCloseDifferences(allDiffs, weightedScores, epsilon, scenario, plain_bits, key_bits)
    df = DataframeFromSortedDifferences(allDiffs, weightedScores, scenario, plain_bits, key_bits)
    df.to_csv(f'{log_file}_best_weighted_differences.csv')
    return(diffs_as_hex, highest_non_random_round)

