import numpy as np
import math

N = int(input("Введите N: "))
M = 2**N

print(f"Введите матрицу вероятностей {M}x{M} построчно:")

p = []
for _ in range(M):
    row = list(map(float, input().split()))
    p.append(row)

p = np.array(p)



def SG_indices(i, K):
    i1 = i + 1  
    l = math.ceil(i1 * 2**(1 - K))
    block_size = 2**(K - 1)

    if l % 2 == 0:
        start = (l - 2) * block_size
    else:
        start = l * block_size

    return np.arange(start, start + block_size)


P_current = np.ones(M)  
all_stages = [P_current.copy()]

for k in range(1, N + 1):
    P_next = np.zeros(M)

    for i in range(M):
        opps = SG_indices(i, k)
        s = np.sum(p[i, opps] * P_current[opps])
        P_next[i] = P_current[i] * s

    P_current = P_next
    all_stages.append(P_current.copy())



for k in range(1, N + 1):
    print(f"\nВероятности выиграть {k} этап:")
    for i in range(M):
        print(f"Команда {i+1}: {all_stages[k][i]:.6f}")

print("\nВероятности выиграть турнир:")
for i in range(M):
    print(f"Команда {i+1}: {all_stages[N][i]:.6f}")
