import numpy as np
import itertools
import time

def precompute_opponents(N: int):
    M = 2**N
    opps = [[None]*(N+1) for _ in range(M)] 

    for K in range(1, N+1):
        block = 2**K
        half = 2**(K-1)
        for pos in range(M):
            base = (pos // block) * block
            offset = pos - base
            if offset < half:
                opps[pos][K] = np.arange(base + half, base + block, dtype=int)
            else:
                opps[pos][K] = np.arange(base, base + half, dtype=int)
    return opps

def stage_win_probs_by_team(N: int, p: np.ndarray, seed_pos_to_team: np.ndarray, opps):
    M = 2**N
    seed = seed_pos_to_team

    Ppos = np.ones(M, dtype=float)
    Wpos = [Ppos.copy()]

    for K in range(1, N+1):
        Pnext = np.zeros(M, dtype=float)
        for pos in range(M):
            opp_positions = opps[pos][K]
            ti = seed[pos]
            tj = seed[opp_positions]
            s = np.sum(p[ti, tj] * Ppos[opp_positions])
            Pnext[pos] = Ppos[pos] * s
        Ppos = Pnext
        Wpos.append(Ppos.copy())

    Wpos = np.stack(Wpos, axis=0)  # (N+1, M)

    pos_of_team = np.empty(M, dtype=int)
    pos_of_team[seed] = np.arange(M)
    W = Wpos[:, pos_of_team]
    return W

def compute_pre_scores(p: np.ndarray) -> np.ndarray:
    return p.sum(axis=1)

def target_exit_from_ranking(N: int, ranked_teams: np.ndarray) -> np.ndarray:
    M = 2**N
    target = np.zeros(M, dtype=int)

    champ = ranked_teams[0]
    target[champ] = -1

    idx = 1
    for k in range(N, 0, -1):      
        need = 2**(N - k)        
        for _ in range(need):
            if idx >= M:
                break
            t = ranked_teams[idx]
            target[t] = k
            idx += 1
    return target

def score_seeding(N: int, p: np.ndarray, seed: np.ndarray, target_exit: np.ndarray, opps) -> float:
    W = stage_win_probs_by_team(N, p, seed, opps)
    M = 2**N
    score = 0.0
    for t in range(M):
        k = target_exit[t]
        if k == -1:
            score += W[N, t]            
        else:
            score += (W[k-1, t] - W[k, t])  
    return float(score)

def canonicalize_block(block):
    L = len(block)
    if L == 1:
        return block
    if L == 2:
        a, b = block
        return (a, b) if a < b else (b, a)

    half = L // 2
    left = canonicalize_block(block[:half])
    right = canonicalize_block(block[half:])
    return left + right if left <= right else right + left

def is_canonical_seed(seed_tuple):
    return seed_tuple == canonicalize_block(seed_tuple)



def generate_transitive_epsilon_matrix(M: int, N: int) -> np.ndarray:
    epsilon = 1.0 / ((2 ** (N + 1)) * N)
    p = np.zeros((M, M), dtype=float)
    
   
    true_strength = np.random.permutation(M) 
    
    for i in range(M):
        for j in range(i + 1, M):
            if true_strength[i] < true_strength[j]:
                p[i, j] = 1.0 - epsilon
                p[j, i] = epsilon
            else:

                p[i, j] = epsilon
                p[j, i] = 1.0 - epsilon
    return p

def get_classic_seed(N: int, ranked_teams: np.ndarray) -> np.ndarray:
    seed_ranks = [0]
    for _ in range(N):
        next_ranks = []
        L = len(seed_ranks) * 2 - 1
        for r in seed_ranks:
            next_ranks.extend([r, L - r])
        seed_ranks = next_ranks
    return np.array([ranked_teams[r] for r in seed_ranks], dtype=int)


if __name__ == "__main__":
    N = int(input("Введите N (рекомендуется 2 или 3): "))
    num_matrices = int(input("Введите количество генерируемых матриц: "))
    M = 2**N
    epsilon = 1.0 / ((2 ** (N + 1)) * N)

    canonical_templates = []
    teams_1idx = tuple(range(1, M+1))
    
    for seed_1idx in itertools.permutations(teams_1idx):
        if is_canonical_seed(seed_1idx):
            canonical_templates.append(np.array(seed_1idx, dtype=int) - 1)
            
    opps = precompute_opponents(N)
    
    accuracies = []

    for i in range(num_matrices):
        p = generate_transitive_epsilon_matrix(M, N)
        pre = compute_pre_scores(p)
        ranked = np.argsort(-pre)
        target_exit = target_exit_from_ranking(N, ranked)
        
        classic_seed = get_classic_seed(N, ranked)
        classic_score = score_seeding(N, p, classic_seed, target_exit, opps)
        
        max_score = -float('inf')
        min_score = float('inf')
        
        for seed_template in canonical_templates:
            sc = score_seeding(N, p, seed_template, target_exit, opps)
            if sc > max_score:
                max_score = sc
            if sc < min_score:
                min_score = sc
                
        diff = max_score - min_score
        if diff > 1e-12:
            accuracy = (classic_score - min_score) / diff
        else:
            accuracy = 1.0 
            
        accuracies.append(accuracy)
        
        print(f"Матр {i+1:>3}/{num_matrices} | Min: {min_score:.4f} | Max: {max_score:.4f} | Класс: {classic_score:.4f} | Точность: {accuracy * 100:>6.2f}%")

    print("-" * 85)
    print(f"Итоги по {num_matrices} транзитивным epsilon-матрицам:")
    print(f"  Средняя точность:      {np.mean(accuracies) * 100:.2f}%")
    print(f"  Максимальная точность: {np.max(accuracies) * 100:.2f}%")
    print(f"  Минимальная точность:  {np.min(accuracies) * 100:.2f}%")
