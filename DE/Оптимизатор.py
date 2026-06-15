import numpy as np
import itertools

def compute_pre_scores(p: np.ndarray) -> np.ndarray:
    return p.sum(axis=1)

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

def build_de_matches(N: int):
    M = 2 ** N
    matches = []
    winners_rounds = {}
    losers_rounds = {}

    def add_match(stage, src_a, src_b):
        mid = len(matches)
        matches.append((stage, src_a, src_b))
        return mid

    winners_rounds[1] = []
    for j in range(M // 2):
        t1 = 2 * j
        t2 = 2 * j + 1
        mid = add_match("Winners Round 1", ("team", t1), ("team", t2))
        winners_rounds[1].append(mid)

    for r in range(2, N + 1):
        winners_rounds[r] = []
        prev = winners_rounds[r - 1]
        for j in range(len(prev) // 2):
            mid = add_match(
                f"Winners Round {r}",
                ("W", prev[2 * j]),
                ("W", prev[2 * j + 1])
            )
            winners_rounds[r].append(mid)

    if N >= 2:
        losers_rounds[1] = []
        for j in range(len(winners_rounds[1]) // 2):
            mid = add_match(
                "Losers Round 1",
                ("L", winners_rounds[1][2 * j]),
                ("L", winners_rounds[1][2 * j + 1])
            )
            losers_rounds[1].append(mid)

    if N == 1:
        grand_final = add_match(
            "Grand Final",
            ("W", winners_rounds[1][0]),
            ("L", winners_rounds[1][0])
        )
    elif N == 2:
        losers_rounds[2] = []
        mid = add_match(
            "Losers Round 2",
            ("W", losers_rounds[1][0]),
            ("L", winners_rounds[2][0])
        )
        losers_rounds[2].append(mid)

        grand_final = add_match(
            "Grand Final",
            ("W", winners_rounds[2][0]),
            ("W", losers_rounds[2][0])
        )
    else:
        for t in range(2, N):
            odd_round = 2 * t - 2
            even_round = 2 * t - 1

            losers_rounds[odd_round] = []
            prev_l = losers_rounds[odd_round - 1]
            cur_w = winners_rounds[t]

            for j in range(len(cur_w)):
                mid = add_match(
                    f"Losers Round {odd_round}",
                    ("W", prev_l[j]),
                    ("L", cur_w[j])
                )
                losers_rounds[odd_round].append(mid)

            losers_rounds[even_round] = []
            prev2 = losers_rounds[odd_round]
            for j in range(len(prev2) // 2):
                mid = add_match(
                    f"Losers Round {even_round}",
                    ("W", prev2[2 * j]),
                    ("W", prev2[2 * j + 1])
                )
                losers_rounds[even_round].append(mid)

        final_lb_round = 2 * N - 2
        losers_rounds[final_lb_round] = []
        mid = add_match(
            f"Losers Round {final_lb_round}",
            ("W", losers_rounds[final_lb_round - 1][0]),
            ("L", winners_rounds[N][0])
        )
        losers_rounds[final_lb_round].append(mid)

        grand_final = add_match(
            "Grand Final",
            ("W", winners_rounds[N][0]),
            ("W", losers_rounds[final_lb_round][0])
        )

    stage_order = []
    for r in range(1, N + 1):
        stage_order.append(f"Winners Round {r}")
    for r in sorted(losers_rounds.keys()):
        stage_order.append(f"Losers Round {r}")
    stage_order.append("Grand Final")

    elimination_order = []
    if N >= 2:
        for r in range(1, max(losers_rounds.keys()) + 1):
            elimination_order.append(f"Losers Round {r}")
    elimination_order.append("Grand Final")

    elimination_counts = {}
    if N >= 2:
        for r, mids in losers_rounds.items():
            elimination_counts[f"Losers Round {r}"] = len(mids)
    elimination_counts["Grand Final"] = 1

    return matches, winners_rounds, losers_rounds, grand_final, stage_order, elimination_order, elimination_counts

def de_stage_and_elim_probs_by_team(N: int, p: np.ndarray, seed_pos_to_team: np.ndarray, bracket_data):
    M = 2 ** N
    matches, winners_rounds, losers_rounds, grand_final, stage_order, elimination_order, elimination_counts = bracket_data

    stage_probs = {name: np.zeros(M, dtype=float) for name in stage_order}
    elim_probs = {name: np.zeros(M, dtype=float) for name in elimination_order}
    tournament_win = np.zeros(M, dtype=float)

    winner_of = [-1] * len(matches)
    loser_of = [-1] * len(matches)

    def resolve_source(src):
        kind, value = src
        if kind == "team":
            pos = value
            return seed_pos_to_team[pos]
        if kind == "W":
            return winner_of[value]
        if kind == "L":
            return loser_of[value]
        raise ValueError("bad source")

    def dfs(mid, prob):
        if mid == len(matches):
            champion = winner_of[grand_final]
            tournament_win[champion] += prob
            return

        stage, src_a, src_b = matches[mid]
        a = resolve_source(src_a)
        b = resolve_source(src_b)

        pab = p[a, b]
        pba = p[b, a]

        winner_of[mid] = a
        loser_of[mid] = b
        stage_probs[stage][a] += prob * pab
        if stage in elim_probs:
            elim_probs[stage][b] += prob * pab
        dfs(mid + 1, prob * pab)

        winner_of[mid] = b
        loser_of[mid] = a
        stage_probs[stage][b] += prob * pba
        if stage in elim_probs:
            elim_probs[stage][a] += prob * pba
        dfs(mid + 1, prob * pba)

        winner_of[mid] = -1
        loser_of[mid] = -1

    dfs(0, 1.0)
    return stage_probs, elim_probs, tournament_win

def target_exit_from_ranking_de(N: int, ranked_teams: np.ndarray, elimination_order, elimination_counts) -> np.ndarray:
    M = 2 ** N
    target = np.empty(M, dtype=object)

    champ = ranked_teams[0]
    target[champ] = "Champion"

    idx = 1
    for stage in reversed(elimination_order):
        need = elimination_counts[stage]
        for _ in range(need):
            if idx >= M:
                break
            t = ranked_teams[idx]
            target[t] = stage
            idx += 1

    return target

def score_seeding_de(N: int, p: np.ndarray, seed: np.ndarray, target_exit: np.ndarray, bracket_data) -> float:
    _, elim_probs, tournament_win = de_stage_and_elim_probs_by_team(N, p, seed, bracket_data)

    M = 2 ** N
    score = 0.0
    for t in range(M):
        if target_exit[t] == "Champion":
            score += tournament_win[t]
        else:
            score += elim_probs[target_exit[t]][t]
    return float(score)

N = int(input("Введите N: "))
M = 2 ** N

print(f"Введите матрицу вероятностей {M}x{M} построчно:")
p = np.array([list(map(float, input().split())) for _ in range(M)], dtype=float)

pre = compute_pre_scores(p)
ranked = np.argsort(-pre)

bracket_data = build_de_matches(N)
_, _, _, _, stage_order, elimination_order, elimination_counts = bracket_data

target_exit = target_exit_from_ranking_de(N, ranked, elimination_order, elimination_counts)

results = []
teams_1idx = tuple(range(1, M + 1))

for seed_1idx in itertools.permutations(teams_1idx):
    if not is_canonical_seed(seed_1idx):
        continue
    seed_0idx = np.array(seed_1idx, dtype=int) - 1
    sc = score_seeding_de(N, p, seed_0idx, target_exit, bracket_data)
    results.append((sc, seed_1idx))

results.sort(key=lambda x: x[0], reverse=True)

print("\nРанжирование по pre-score (team_id: pre_score):")
for t in ranked:
    print(f"Team {t+1}: {pre[t]:.6f}")

print("\nЦелевые стадии вылета / победы:")
for t in ranked:
    print(f"Team {t+1}: {target_exit[t]}")

print("\nВсе посевы без симметрий (по убыванию score):")
for i, (sc, seed) in enumerate(results, 1):
    print(f"{i:>4}. score={sc:.6f}  seed:", *seed)
