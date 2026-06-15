import math
N = 4
team_array = [i+1 for i in range(2**N)]
print(team_array)
def SG(i, K):
    segments_number = 2**(N - K + 1)
    l = math.ceil(i*2**(1-K))
    if (l%2 == 0):
        return team_array[(l - 2) * (2**(K - 1)):(((l - 1) * (2**(K-1)) - 1)) + 1]
    if (l%2 == 1): 
        return team_array[l * (2**(K - 1)):(((l + 1) * (2**(K-1)) - 1)) + 1]

print(SG(5,4))
