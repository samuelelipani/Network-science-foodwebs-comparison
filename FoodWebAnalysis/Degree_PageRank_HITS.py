import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


########## DEGREE DISTRIBUTION ##################################################################################

def find_degree(A0):
    
    """
    Return both degree in and degree out for an adjacency matrix A
        Args:
            A0 (array or dataframe)  :  matrix on wich degree counting will be performed
            
        Returns:
            deg_in:   in degree of each node
            deg_out:  out degree of each node
    """

    if type(A0)!=np.ndarray:
        A0 = np.array(A0)
        
    lenA = A0.shape[0]
    
    deg_in  = (A0).dot(np.ones(lenA)) 
    deg_out = (A0.T).dot(np.ones(lenA))
    
    return deg_in , deg_out



def degree_distr(\
    A,
    which_deg = 1,
    plot = False,
    figsize = [16,5],
    title = None
    ):
    
    """
    Calculates and plots degree distribution
        Args : 
            A (matrix or df)    :  Adjacency matrix
            which_deg (int)     :  if is to use out or in degree
                                    defaults to = 1
            plot (bool)         :  if True, it will plot the cumulative and non cumulative sidtributions
                                    defaults to = False
                                    
        Returns:
            C_unique   :   Unique rank values present in the network
            probability:   Fraction of nodes associated to each unique value
            cumprob    :   Inverse cumulative distribution of "probability"
    """
    if type(A)!=np.ndarray:
        A = np.array(A)    

    counts = A.sum(axis=which_deg)
    C = np.sort(counts)
    C = np.array(C[C>0])
    C_unique,probability = np.unique(C,return_counts=True)
    probability = probability/np.sum(probability)
    cumprob = np.cumsum(probability[::-1])
    #probability
    if plot:
        fig,ax = plt.subplots(1,2,figsize=figsize)

        ax[0].loglog(C_unique,probability,'k.')
        ax[0].grid(which='both', linestyle='--', linewidth=0.5)
        ax[0].set_title('Degree distribution')
        ax[0].set_ylabel('$p_k$')
        ax[0].set_xlabel('k')

        ax[1].loglog(C_unique[::-1],cumprob,'k.')
        ax[1].grid(which='both', linestyle='--', linewidth=0.5)
        ax[1].set_title('Degree cumulative distribution')
        ax[1].set_ylabel('$P_k$')
        ax[1].set_xlabel('k')
                
        if title:
             plt.suptitle(title, fontsize=14)
                
        plt.show()

                
    return C_unique, probability, cumprob




## PAGE RANK #################################################################################################

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
    
    
############## HITS  ###########################################
    
def HITS(A,plot, **kwargs):
    
    """Function to calculate the HITS algorithm.
        Args
            A (matrix)       : Adjacency matrix
            **kwargs         : To be passed to power_iteration
        
        Returns : two vectors, a and h containing autorities and hubs"""
    
        
    if type(A)!=np.ndarray:
        A = np.array(A)
    
    N = A.shape[0]
    
    #Authorities
    a0 = np.ones(N)
    Ma = np.dot(A, A.T)

    a = power_iteration(a0,Ma,q=1, **kwargs) # q can be setted equal to anything since 1-c = 0 in this case


    # hubs
    h0 = np.ones(N)
    Mh = np.dot(A.T, A)

    h = power_iteration(h0,Mh,q=1 **kwargs)
    
    return a , h



############ PLOTS ####################################

def plotRank(A,ra):
    """Plots rank :)"""

    
    plt.figure(figsize=(10, 10))
    
    ins,outs = find_degree(A)


    plt.plot(ins, ra, 'o', markersize = 2, color = "k")
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.title("Authorities", size = 20)
    plt.xlabel("In degree", size = 18)
    plt.ylabel("PageRank", size = 18)

    plt.show()
    
    
def Hist_of_Rank(dataframes, which = 'PR'  , bins=16 , range=None, c = 0.7, ignore_position = True ,scatter=True ):
    
    """Function to plot the histogram distribution of PageRank, Trophic level or both for a certain dataframe
        Args:
            dataframes (str or list of str)   : Filename(s) of dataframes to analyze
            which ('PR','TR' or 'both')       : Str to decide what to plot
            bins (int)                        : N of bins
            range  (list)                     : Range o values for hist
            c (float)                         : Teleport prob to be passed to PageRank
            ignore_position (bool)            : if to ignore the relative position of the bins edges in the
                                                histogram
            scatter (bool)                    : If True, bins height will just be a scatter plot
        """
    
    extented_name = {'PR':'Page Rank' , 'TR':'Trophic Level' }
    
    what_to_plot = ('PR','TR') if which == 'both' else ([which])
    
    first_df = True
    
    
    if type(dataframes)==pd.DataFrame:
        dataframes = ([dataframes])
        
    for what in what_to_plot:
        
        plt.figure(figsize=[15,7])
        plt.title('Distribution of ' + extented_name[what])

        for df_name in dataframes:
            df = load_df(df_name , to_bool = True)
            
            if what=='PR':
                param_to_plot = PageRank(df.T, plot=False , c=c)   
            elif what =='TR':
                param_to_plot = compute_trophic_levels(df)
                
                
            heights,pre_edges = np.histogram(param_to_plot , bins = bins , 
                                         range = range , density = False)
            
            heights = np.concatenate((np.array([0]),heights))/param_to_plot.size
            
            if ignore_position and first_df: 
                edges = pre_edges
                first_df = False
            elif not ignore_position:
                edges = pre_edges
                
            
            if scatter:
                plt.scatter(edges[:-1],heights[1:]   ,label=df_name)
                plt.plot(edges[:-1],heights[1:])
            else:
                plt.step(edges , heights    , label = df_name )

        plt.grid()
        plt.xlabel(extented_name[what])
        plt.ylabel('Density')
        plt.legend()
        plt.show()        

