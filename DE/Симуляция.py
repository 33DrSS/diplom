import numpy as np
import random

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

winner_of = [-1] * len(matches)
loser_of = [-1] * len(matches)
losses = [0] * M
elim_stage = [None] * M

def resolve_source(src):
    kind, value = src
    if kind == "team":
        return value
    if kind == "W":
        return winner_of[value]
    if kind == "L":
        return loser_of[value]
    raise ValueError("bad source")

for mid, (stage, src_a, src_b) in enumerate(matches):
    a = resolve_source(src_a)
    b = resolve_source(src_b)

    if random.random() < p[a, b]:
        winner = a
        loser = b
    else:
        winner = b
        loser = a

    winner_of[mid] = winner
    loser_of[mid] = loser

    losses[loser] += 1
    if losses[loser] == 2:
        elim_stage[loser] = stage

champion = winner_of[grand_final]
grand_final_loser = loser_of[grand_final]

groups = {}
groups[(1, 1)] = [champion + 1]
groups[(2, 2)] = [grand_final_loser + 1]

stage_to_teams = {}
for team in range(M):
    if team == champion or team == grand_final_loser:
        continue
    stage = elim_stage[team]
    stage_to_teams.setdefault(stage, []).append(team + 1)

lb_stage_names = [f"Losers Round {r}" for r in sorted(losers_rounds.keys(), reverse=True)]

next_place = 3
for stage in lb_stage_names:
    teams_here = sorted(stage_to_teams.get(stage, []))
    if not teams_here:
        continue
    cnt = len(teams_here)
    start = next_place
    end = next_place + cnt - 1
    groups[(start, end)] = teams_here
    next_place += cnt

for key in sorted(groups):
    start, end = key
    place_str = str(start) if start == end else f"{start}-{end}"
    teams_str = " ".join(map(str, groups[key]))
    print(f"{place_str}: {teams_str}")
