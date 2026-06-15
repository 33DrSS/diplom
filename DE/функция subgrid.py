import math

N = 4
M = 2 ** N
team_array = [i + 1 for i in range(M)]

print(team_array)


def build_de_bracket(N):
    M = 2 ** N
    entrants = [{i} for i in range(1, M + 1)]

    W = {}
    L = {} 

    W[1] = []
    for j in range(0, M, 2):
        W[1].append((entrants[j], entrants[j + 1]))

    for r in range(2, N + 1):
        W[r] = []
        prev = W[r - 1]
        for j in range(0, len(prev), 2):
            left = prev[j][0] | prev[j][1]
            right = prev[j + 1][0] | prev[j + 1][1]
            W[r].append((left, right))

    def loser_of_W(round_no, match_no):
        a, b = W[round_no][match_no]
        return a | b

    def winner_of_L(round_no, match_no):
        a, b = L[round_no][match_no]
        return a | b

    L[1] = []
    for j in range(0, len(W[1]), 2):
        L[1].append((loser_of_W(1, j), loser_of_W(1, j + 1)))

    if N == 2:
        L[2] = [(L[1][0][0] | L[1][0][1], loser_of_W(2, 0))]
    else:
        for t in range(2, N):
            odd_round = 2 * t - 2
            even_round = 2 * t - 1

            L[odd_round] = []
            for j in range(len(W[t])):
                L[odd_round].append((winner_of_L(odd_round - 1, j), loser_of_W(t, j)))

            L[even_round] = []
            for j in range(0, len(L[odd_round]), 2):
                left = L[odd_round][j][0] | L[odd_round][j][1]
                right = L[odd_round][j + 1][0] | L[odd_round][j + 1][1]
                L[even_round].append((left, right))

        final_L_round = 2 * N - 2
        left = L[final_L_round - 1][0][0] | L[final_L_round - 1][0][1]
        right = loser_of_W(N, 0)
        L[final_L_round] = [(left, right)]

    GF = [(W[N][0][0] | W[N][0][1], L[2 * N - 2][0][0] | L[2 * N - 2][0][1])]

    return W, L, GF


W_bracket, L_bracket, GF_bracket = build_de_bracket(N)


def DG(i, bracket, K=None):
    """
    i        - номер команды (с 1)
    bracket  - "W", "L" или "F"
    K        - номер раунда для W/L
    """

    if bracket == "W":
        if K not in W_bracket:
            return []
        matches = W_bracket[K]

    elif bracket == "L":
        if K not in L_bracket:
            return []
        matches = L_bracket[K]

    elif bracket == "F":
        matches = GF_bracket

    else:
        raise ValueError('bracket должен быть "W", "L" или "F"')

    opponents = set()

    for left, right in matches:
        if i in left:
            opponents |= right
        if i in right:
            opponents |= left

    opponents.discard(i)
    return sorted(opponents)


# Примеры
print("Winners round 1, team 5:", DG(5, "W", 1))
print("Winners round 2, team 5:", DG(5, "W", 2))
print("Winners round 4, team 5:", DG(5, "W", 4))

print("Losers round 1, team 5:", DG(5, "L", 1))
print("Losers round 2, team 5:", DG(5, "L", 2))
print("Losers round 3, team 5:", DG(5, "L", 3))

print("Grand Final, team 5:", DG(5, "F"))
