import numpy as np

N = int(input("Введите N: "))
M = 2 ** N

print(f"Введите матрицу вероятностей {M}x{M} построчно:")

p = []
for _ in range(M):
    row = list(map(float, input().split()))
    p.append(row)

p = np.array(p, dtype=float)

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

stage_probs = {name: np.zeros(M) for name in stage_order}
tournament_win = np.zeros(M)

winner_of = [-1] * len(matches)
loser_of = [-1] * len(matches)

def resolve_source(src):
    kind, value = src
    if kind == "team":
        return value
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
    dfs(mid + 1, prob * pab)

    winner_of[mid] = b
    loser_of[mid] = a
    stage_probs[stage][b] += prob * pba
    dfs(mid + 1, prob * pba)

    winner_of[mid] = -1
    loser_of[mid] = -1

dfs(0, 1.0)

for stage_name in stage_order:
    print(f"\nВероятности выиграть {stage_name}:")
    for i in range(M):
        print(f"Команда {i+1}: {stage_probs[stage_name][i]:.6f}")

print("\nВероятности выиграть турнир:")
for i in range(M):
    print(f"Команда {i+1}: {tournament_win[i]:.6f}")
