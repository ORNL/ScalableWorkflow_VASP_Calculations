import numpy as np  # summation

def flatten(l):
    return [item for sublist in l for item in sublist]

def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n))
