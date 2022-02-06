import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

############################ Extintion and influence score ###########################
    
# Extintion_vector(n)[i] = quanto la specie n compone la dieta della specie i, per via diretta o indiretta,
#                          assumento link uniformi. Valore tra 0 e 1, si assume Extintion_vector(n)[n] = 1
#
# Influence_vector(n)[i] = quanto la specie i compone la dieta della specie n, per via diretta o indiretta,
#                          assumento link uniformi. Valore tra 0 e 1, si assume Influence_vector(n)[n] = 1.
#
# L'extintion score e' la somma delle componenti dell'omonimo vettore

def node_score_vector(A, n, interpretation, eps=10**(-4)):
    """
        Function that returns extintion vector and score of species n (node n)
        OR influence vector and score of species n (node n)
        Args: 
             A                : Is binary adjacency matrix
             n                : Is the number of the killed node (species)
             interpretation   : Can be "extintion" or "influence"
                
    """
    if(interpretation=="extintion"):
        # Adj matrix obtained by normalizing out degree and then transposing
        A = A/(sum(A)+2**(-31))
        A = A.T
    elif(interpretation=="influence"):
        # Adj matrix obtained by normalizing in degree
        A = A.T/(sum(A.T)+2**(-31))
        A = A.T
    else:
        print("Invalid score")
        return
    
    # Setting row i of matrix = 0 (removing inputs of dead species, useful to handle loop)
    mask = np.ones(len(A))
    mask[n] = 0
    A = (A.T*mask).T
    
    
    # Iterative procedure: 
    # vector      considering extintion interpretation it tells us how much a node is dead 
    #             in a certain iteration step. It is summed of updates over iteration step
    vector = np.zeros(len(A))
    vector[n] = 1 
    update = vector 
    
    converged = False
    while(not converged):
        
        # Propagating death in the network, only last reached nodes are "activated" with their vector value
        active_nodes = np.array(update!=0,dtype=int)
        active_vector = vector*active_nodes
    
        update = np.dot(A,active_vector)

        # Nodes lose intensity
        A = A-A*(active_vector)

        # update vector
        vector = vector + update
        
        # check convergence
        converged = np.sum(update) < eps
          
            
    vector = np.round((vector),3)
    score = np.round(sum(vector),3)
    return ( vector , score )



def score_distribution(A,interpretation):
    """Scores every node in the universe"""
    return  [node_score_vector(A,i,interpretation)[1] for i in range(0,A.shape[0])]


