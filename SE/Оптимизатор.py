import numpy as np
import itertools

def precompute_opponents(N: int):
    M = 2**N
    opps = [[None]*(N+1) for _ in range(M)]  # opps[pos][K] -> np.array

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

N = int(input("Введите N: "))
M = 2**N

print(f"Введите матрицу вероятностей {M}x{M} построчно:")
p = np.array([list(map(float, input().split())) for _ in range(M)], dtype=float)

pre = compute_pre_scores(p)
ranked = np.argsort(-pre)  # 0..M-1
target_exit = target_exit_from_ranking(N, ranked)

opps = precompute_opponents(N)

results = []
teams_1idx = tuple(range(1, M+1))

for seed_1idx in itertools.permutations(teams_1idx):
    if not is_canonical_seed(seed_1idx):
        continue
    seed_0idx = np.array(seed_1idx, dtype=int) - 1
    sc = score_seeding(N, p, seed_0idx, target_exit, opps)
    results.append((sc, seed_1idx))

results.sort(key=lambda x: x[0], reverse=True)

print("\nРанжирование по pre-score (team_id: pre_score):")
for t in ranked:
    print(f"Team {t+1}: {pre[t]:.6f}")

print("\nВсе посевы без симметрий (по убыванию score):")
for i, (sc, seed) in enumerate(results, 1):
    print(f"{i:>4}. score={sc:.6f}  seed:", *seed)
