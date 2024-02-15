def ED(str_1, str_2):
    m = len(str_1)
    n = len(str_2)
    # Initializes the dynamic programming matrix with sizes m+1 and n+1 respectively
    Distance = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(n + 1):
        Distance[0][i] = i
    #
    for j in range(m + 1):
        Distance[j][0] = j
    # Initialize the first row and column of the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            distance_delete = Distance[i - 1][j] + 1
            distance_add = Distance[i][j - 1] + 1
            if str_1[i - 1] == str_2[j - 1]:
                distance_change = Distance[i - 1][j - 1]
            else:
                distance_change = Distance[i - 1][j - 1] + 1
            Distance[i][j] = min(distance_delete, distance_add, distance_change)
    # Count the items from bottom to top
    return Distance[m][n]


# define our own similarity function
def similarity(x, scale):
    return 1 / (1 + scale * x)