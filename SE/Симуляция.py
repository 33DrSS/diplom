import numpy as np
import random

N = int(input("Введите N: "))
M = 2 ** N

print(f"Введите матрицу вероятностей {M}x{M} построчно:")

p = []
for _ in range(M):
    row = list(map(float, input().split()))
    p.append(row)

p = np.array(p)

teams = list(range(M))
places = [0] * M
round_eliminated = [0] * M

current_round = teams.copy()

for round_num in range(1, N + 1):
    next_round = []
    for i in range(0, len(current_round), 2):
        a = current_round[i]
        b = current_round[i + 1]

        if random.random() < p[a][b]:
            winner = a
            loser = b
        else:
            winner = b
            loser = a

        next_round.append(winner)
        round_eliminated[loser] = round_num

    current_round = next_round

champion = current_round[0]
round_eliminated[champion] = N + 1

groups = {}

for team in range(M):
    r = round_eliminated[team]
    if r == N + 1:
        key = (1, 1)
    elif r == N:
        key = (2, 2)
    else:
        start = 2 ** (N - r) + 1
        end = 2 ** (N - r + 1)
        key = (start, end)

    groups.setdefault(key, []).append(team + 1)

for key in sorted(groups):
    start, end = key
    if start == end:
        place_str = f"{start}"
    else:
        place_str = f"{start}-{end}"

    teams_str = " ".join(map(str, groups[key]))
    print(f"{place_str}: {teams_str}")
