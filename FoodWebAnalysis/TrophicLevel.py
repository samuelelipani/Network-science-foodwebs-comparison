import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import networkx as nx


def basal_species(A):
    v = sum(A)==0
    return np.array(v,dtype=int)


def top_species(A):
    v = sum(A.T)==0
    return np.array(v,dtype=int)


def compute_trophic_levels(A,df=True):
    
    """Computes trophic level given a of food web
        Args:
            A (pandas.DataFrame)     : Square dataframe containing adjacency matrix and species 
                                    names 
            df (bool)                : If false the function consider the argument as np.array and returns np.array
        Returns:
            Tr_Level (pandas.Series) : Series containing species name and respective trophic 
                                        Level."""
    
    N = len(A)       # number of specs
    S = np.zeros(N)  # initializes to zero the trophic level container
    
    ## assigns the pure level 1 and pure level 2 trophic levels
    ## as a starting condition for the iteratove procedure
    
    # assign trophic levels to pure level 1 (basal) species
    out_deg  = A.sum(axis=0)      # output degree of foodweb (number of preys for each spec)
    level_1_mask = (out_deg==0)   # mask for species on trophic level 1
    S[level_1_mask] = 1           # assign trophic level value
    
    if(df):
        out_deg2  = (A.loc[np.logical_not(level_1_mask)]).sum(axis=0)       # number of preied without considering basals
    else:
        out_deg2  = (A[np.logical_not(level_1_mask)]).sum(axis=0)    
     
    # assign trophic levels to pure level 2 species   
    level_2_mask = np.logical_and((out_deg2==0),(level_1_mask==False))  # mask for species on trophic level 2
    S[level_2_mask] = 2                                                 # assign trophic level value
    
    # mask for alredy computed trophic level values
    precomputed = np.logical_or(level_1_mask,level_2_mask)
    
    ## complete trophic levels assignments iteratively
    
    epsilon = 0.0001*N     # prophic level increment treshold to stop iteration
    incr = epsilon+1       # in order to enter the cicle

    i = 0
    while(incr > epsilon):
       # print(np.dot(A.T,S)/out_deg)

        with warnings.catch_warnings(): # in order to ignore the divide by zero warning
            warnings.simplefilter("ignore")
            new_S = np.where(precomputed,S,1 + np.dot(A.T,S)/out_deg)    # update with self consistence equation where needed
        incr  = np.abs(new_S-S).sum()                                # overall variation
        S     = new_S
        
        if(i>100):
            print("Trophic level computation is not converging! No basal species or isolated nodes")
            break
    if(df):    
        return pd.Series(S,index=A.index)
    else:
        return S


def robustness(A):
    
    size = A.shape[0]
    k = np.zeros(size)

    # computing degree
    for i in range(size):
        k[i] = np.sum(A[i]) + np.sum(A[:,i])

    inh_ratio = np.sum(k**2)/np.sum(k)
    breaking_point = 1-1/(inh_ratio-1)

    return np.round(inh_ratio,3) , np.round(breaking_point,3)



def size_giant_component(G):
    largest_cc = max(nx.connected_components(G), key=len)
  
    return len(largest_cc)


def directed_robustness(A): 
    Adj = A.copy()

    sums = A.sum(axis = 0) + 2**(-32)    
    M = A/sums                           
    E = PageRank(M,c=0.7)

    size = len(E)
    removed_nodes = 0
    
    R = 1
    for i in range(size-1):
        removed = np.argmax(E)
        removed_nodes += 1

        E[removed] = 0
        
        # remove row
        Adj[removed] = 0
        # remove column
        Adj[:,removed] = 0
    
        G = nx.from_numpy_matrix(Adj, create_using=nx.Graph)
        
        if( size_giant_component(G) <= (size-removed_nodes)/2 ):
            
            R = removed_nodes/size
            return np.round(R,3)
        
    
    return R

def norm2 (v,w):
    return np.sum( (v-w)**2 )


def power_iteration(v0,            # initial guess
                           M,             # martix
                           q,             # teleportation vector
                           eps=10**(-11),  # precision
                           c=1            # 1 - teleportation probability
                          ):  
    """Power iteration to calculate pagerank & affini 
        Args:
            v0(vector)           : Initial guess
            M (matrix)           : Adjacency matrix (normalized)
            q (vector)           : Teleportation 
            eps (float) = 1e-11  : Precision
            c (float)   = 1      : 1 - teleport prob
            """
    
    converged = False
    v = v0/np.sum(v0)
    
    while(not converged):                          # iteration loop
        new_v = M.dot(v*c) + (1-c)*q               # updating v
        new_v = new_v/np.sum(new_v)                # normalization
        converged = norm2(v,new_v) < eps           # check condition
        
        v = new_v 
        
    return v

    
def PageRank( A , c = 0.85, q = None ,plot=False,**kwargs):
    
    """Calculates PageRank for an adjacency matrix A
        Args:
            A (matrix)     : Adjacency matrix
            c (float)      : 1 - teleportation prob
                            defaults to = 0.85
            q (vector)     : Teleportation vector, if None, it will default
                            to vector of equal probabilities for each node.
                            default = None
            plot (bool)    : If true will plot pagerank against degree
            **kwargs       : args to be passed to power_iteration
            
        Returns: vector containing pagerank values 
        """
    Indexes = None
    series_out=False
    if type(A)!=np.ndarray:
        series_out = True
        Indexes = A.index
        A = np.array(A)
        
    lenA = A.shape[0]
    q    = q if q else np.ones(lenA)/len(A)
    v0   = np.copy(q)
    v = power_iteration(v0 , A , q , c = c, **kwargs)
    
    if plot:
        plotRank(A,v)
    
    return pd.Series(v,index=Indexes) if series_out else v

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


