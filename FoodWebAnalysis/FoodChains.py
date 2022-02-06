import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##########
        
        
    
def basal_species(A):
    v = sum(A)==0
    return np.array(v,dtype=int)


def top_species(A):
    v = sum(A.T)==0
    return np.array(v,dtype=int)

    
    
    
# The idea is to count how many food chains of length l connects starting nodes v0 to each node n.
# We can do this propagating v1 in the food web -> applying A to v1.
# The key point is that step 1 (v1) tells us which nodes can be reached in 1 jump (starting point), 
# step 2 tells us which nodes can be reached in 2 jumps and so on ...
# Iterations stops when vS = zeros , no food chain can procede.
#
# For this reason, if we build the SxN matrix R := (v1,
#                                                   v2,
#                                                   v3,
#                                                   ...,
#                                                   vS)
#
# we have that Rij = number of food chain of length j+1 which connects starting nodes to node i
# 
# Then it becomes very easy to compute the mean chains length wich connects starting nodes to node i

def food_chains(A, v1, normalization=False, direction=0, eps=10**(-4)):
    ''' Function that returns average number of needed "jumps" to reach each node in a random walk which can 
     starts in multiple given nodes (top predators or basal species).
     Args:
                # A               : is NxN binary adjacency matrix
                # v1              : is a vector which indicates all possible starting points. 
                #                   v1[n] should be 1 if n is a the starting node, 0 otherwise.
                #
                # normalization   : if true A columns are normalized
                # direction = 0  -> from top predators to basal species
                #           = 1  -> from basal species to top predators '''
    
    # Transposing
    if(direction==1):
        A = A.T
       
     
    # Normalizing columns, in this case the number of food chains is weigthed by the link
    # and I am not sure about the interpretation
    if(normalization):
        A = A/(sum(A)+2**(-31)) 
        
        
    R = []
    R.append(v1)
    n_steps = 1
    converged = False
    
    while(not converged):
        v1 = np.dot(A,v1)
        
        # natural end condition
        converged = np.sum(v1) < eps 
        
        # check for loops, in that case break the while
        if ( any((R[:]==v1).all(1)) ):
            print("LOOP DETECTED!")
            break
        
        n_steps += 1  
        # keeping in memory
        R.append(v1)             
     
    
    # normalizing matrix of food chain counts in order to performe the average 
    R = np.array(R)
    R = R/(sum(R)+2**(-31))
    
    # vector useful to compute average, it is vector of all possile chains length
    j = np.linspace(1,n_steps,n_steps)-1
    
    # averaging number of needed jump to reach each nodes, this works because columns are normalized
    chains_length = np.dot(j,R)
    
    return np.round(chains_length,3)
