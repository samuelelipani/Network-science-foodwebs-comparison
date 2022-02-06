import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


############# COMPUTE GAMMA ###############

def gamma_liner_fit(d , log_kmin , log_min=None, log_max=None , nbins = 16, ax = None, title=None,use_fit=True):
    
    """Finds and plots the best gamma for the degree distribution
    Args:
        d  (vector)     : Degree of each node
        log_min (float) : Start of the linear fit
        log_max (float) : Stop of the linear fit
        log_kmin (float): Boh
        nbins ( int)    : you know """
    
    # Log binning 
    d = d[d!=0]
    log_min = log_min if log_min else np.log10(np.min(d))
    log_max = log_max if log_max else np.log10(np.max(d))
        
    
    bins = np.logspace(log_min, log_max, nbins)

    # Computing histogram and bins middle point (needed in the fit)
    hist = np.histogram(d, bins=bins, density=True)
    logbin_p_k = hist[0]
    middle_bins = np.sqrt( np.delete(bins, 0) * np.delete(bins, nbins-1) )

    # considering k>kmin
    mask = np.log10(middle_bins) > log_kmin
    x = np.log10( middle_bins[mask] )
    y = np.log10( logbin_p_k[mask] )


    # Linear fit
    params , cov_matrix = np.polyfit(x, y, deg =1,cov = True)
    lin_gamma = -params[0]
    if not use_fit:
        kmin = 10**log_kmin # multiple values have been tried (with k = 50 the resulting fit is good)

        # Fit gamma only in the chosen interval
        d2 = d[d>kmin]
        lin_gamma = 1 + 1/np.mean(np.log(d2/kmin))
        params[0] = -lin_gamma
    
    error = np.sqrt(cov_matrix[0,0])

    if not ax:
        fig,ax = plt.subplots(figsize=(10, 8))

    ax.loglog(middle_bins, logbin_p_k , 'o', markersize = 4)
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.set_title(title if title else "Gamma linear fit with log binnin", size = 20)
    ax.set_xlabel("k", size = 18)
    ax.set_ylabel("$p_k$", size = 18)

    # plotting fit
    x2 = np.logspace(log_min,log_max,100)
    y2 = np.full( 100, 10**(params[1])*x2**(-lin_gamma) ) 
    ax.plot(x2,y2, 
             label ="$\gamma=$"+str( np.round(lin_gamma,1)) + "$\pm$" + str(np.round(error,1)),
             color="red", linestyle="--")
    # plot k-min
    ax.vlines(10**log_kmin,np.min(y2),np.max(y2) , 
               linestyle = "--",linewidth=0.4,
               label = "$k_{min}$ =" + f"{10**log_kmin:.0f}" )
    ax.legend(fontsize=18)

    
    return lin_gamma, error


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

def gamma_linear_fit2(df,k_min,k_max=None,plot=False):
    """computes and maybe plot gamma fit with cumulative distr
        Returns: 
            gamma and computed error on gamma"""

    deg,_,Pk = degree_distr(df)
    deg = deg[::-1]
    mask = np.array(deg>k_min) 
    if k_max:
        mask = mask & np.array(deg<k_max)
    logdeg,logPk = np.log10(deg), np.log10(Pk)
    params,cov = np.polyfit(logdeg[mask],logPk[mask],1,cov=True)
    gamma,q = params[0],params[1]


    # Fit gamma only in the chosen interval

    if plot:
        liney  =logdeg*gamma+q
        if k_max:
            plt.vlines(np.log10(k_max),liney.max(),liney.min())
   
        plt.vlines(np.log10(k_min),liney.max(),liney.min())
        plt.plot(logdeg,logPk,'k.')
        plt.plot(logdeg, logdeg*gamma+q)
    return -(gamma-1),np.sqrt(cov[0,0])


def compute_assortaivity(A,plot=False):
    """Compute assortativity 
        (ATTENTION!!!!!!!A needs to be a boolean matrix)
        Returns:
                mu and computed error on mu"""
    deg = [np.sum(a) for a in A.T if a.sum()]
    Knn = np.array([np.sum(A[mask])/(mask.sum()) for mask in A.T if mask.sum()])
    unique_deg,counts = np.unique(deg,return_counts=True)
    Knn_redux = []
    for u_deg,count in zip(unique_deg,counts):
        mask = deg==u_deg
        Knn_redux.append(Knn[mask].sum()/count)

    logu,logKnn = np.log10(unique_deg),np.log10(Knn_redux)
    params,cov = np.polyfit(logu,logKnn,1,cov=True)
    mu,q = params[0],params[1]
    #plt.plot(logdeg,logKnn,'k.',label = 'mu = ' +str(mu))
    if plot:
        plt.plot(np.log10(unique_deg),np.log10(Knn_redux),'r.')
        plt.plot(logu,logu*mu+q,'b--',label = '$\mu$ = ' +str(mu))
        plt.legend()
        plt.show()
    return mu