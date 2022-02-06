import numpy as np

def basal_species(A):
    v = A.sum(axis=0)==0
    return np.array(v,dtype=int)


def top_species(A):
    v = A.sum(axis=1)==0
    return np.array(v,dtype=int)


def top_species_cann(A): # with cannibalism
    B = A.to_numpy()
    np.fill_diagonal(B,0)
    v = B.sum(axis=1)==0
    return np.array(v,dtype=int)


def compute_connectance(A):
    return A.sum().sum()/len(A)**2