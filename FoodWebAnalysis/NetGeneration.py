import pandas as pd
import numpy as np
import matplotlib as mpl

from Observables import *
from TrophicLevel import *


########################## GENERATIVE MODELS #######################################

def generate_random(s,c,seed=None):

    """
    random network with s nodes and probability of activation c for each connection
    without isolated nodes
        Args
            s : number of nodes of the originally generated network
            c : activation probability of each possible connection

        Return
            random newtork  without isolated nodes adj matrix
    """
    
    if seed is not None: np.random.seed(seed)
    
    # generate network
    A = np.zeros((s,s))            # adj matrix template
    p = np.random.uniform(0,1,s*s)
    mask = (p <= c).reshape(s,s)   # estraction of active links
    A[mask] = 1                    # assignment of new links

    # delete isoleted nodes
    rowsum = A.sum(axis = 1)
    colsum = A.sum(axis = 0)
    mask = (rowsum + colsum) == 0
    A = A[~mask][:,~mask]          

    return A.astype(int)



def generate_cascade(s,c,seed=None):
    
    """
    cascade model network without isolated nodes
    Args
        s   :   number of nodes of the originally generated network
        c   :   
    Return
        adj matrix of a cascade model sampling without the isolated nodes
    """

    if seed is not None: np.random.seed(seed)
    
    # generate network
    A = np.zeros((s,s))                    # adj matrix template
    p = 2*c*s/(s-1)                        #
    pc = np.random.uniform(0,1, size = s)
    A[:, pc < p] = 1
    A = np.triu(A, k = 1)

    # delete isolated nodes
    rowsum = A.sum(axis = 1)
    colsum = A.sum(axis = 0)
    mask = (rowsum + colsum) == 0
    A = A[~mask][:,~mask]

    return A.astype('int')




def generate_niche(s,c,seed=None):
    """
    niche model network without isolated nodes
    Args
        s   :   number of nodes of the originally generated network
        c   :   
    Return
        adj matrix of a niche model sampling without the isolated nodes
    """
    
    if seed is not None: np.random.seed(seed)
    
    dataframe = pd.DataFrame(np.zeros((s,s))) #, index = np.arange(5) + 1) 
    dataframe.columns = np.arange(s) + 1      # set column index from 1 to s included
    dataframe.set_index(dataframe.columns)    # set row    index from 1 to s included

    # we then substitute columns names and indices with niche values np.random.uniform(0, 1)
    ni = np.random.uniform(0, 1, size = s)  # extraction of niche values

    dataframe = dataframe.set_index(ni)    # set column index to corresponding niche value
    dataframe.columns = ni                 # set row    index to corresponding niche value
    y = np.random.uniform(0, 1, size = s)  
    ri = ni*(1 - (1 - y)**(2*c/(1-2*c)))   
    ci = np.random.uniform(ri/2, ni)
    lower_bound = ci - ri/2
    upper_bound = ci + ri/2
    dataframe[((dataframe.index > lower_bound[:,None]).T)*((dataframe.index < upper_bound[:,None]).T)] = 1
    n = np.array(dataframe)

    # delete isolated nodes
    rowsum = n.sum(axis = 1)
    colsum = n.sum(axis = 0)
    mask = (rowsum + colsum) == 0
    n = n[~mask][:,~mask]
    
    return n.astype('int')




def generate_nested(s,c,seed=None, check = False):
    ''' check is useful to understand if the generation has gone fine'''
    if seed is not None: np.random.seed(seed)
    dataframe = pd.DataFrame(np.zeros((s,s)))#, index = np.arange(5) + 1) 
    ni = np.random.uniform(0, 1, size = s)
    dataframe = dataframe.set_index(ni)
    dataframe.columns = ni
    dataframe.columns = dataframe.columns.sort_values() 
    dataframe = dataframe.set_index(dataframe.columns)
    y = np.random.uniform(0, 1, size = s)
    ri = ni*(1 - (1 - y)**(2*c/(1-2*c)))
    ri /= np.sum(ri)
    ri = (ri*s*s*c).astype(int)
    ri[0] = 0 # in order to have at least a basal species
    ri[ri > s-1] = s-1
    for i in range(s):
        while dataframe[dataframe.columns[i]].sum() < ri[i]:
            randomprey = np.random.randint(i+1) # 1
            dataframe[dataframe.columns[i]].iloc[randomprey] = 1
            if dataframe.iloc[randomprey].sum() == 1: # 2 
                True
            elif dataframe[dataframe.columns[i]].sum() < ri[i]: 
                try: 
                    a = np.random.choice(dataframe[dataframe.columns[dataframe.iloc[randomprey] != 0]].columns)
                    randomprey1 = np.random.choice(dataframe[dataframe[a] != 0].index)
                    dataframe[dataframe.columns[i]].loc[randomprey1] = 1
                except: continue
            if (dataframe[dataframe.columns[i]].sum() < ri[i]) and (dataframe.iloc[randomprey].sum() == 1):
                randomprey0 = np.random.randint(i,s)
                dataframe[dataframe.columns[i]].iloc[randomprey0] = 1
            elif (dataframe[dataframe.columns[i]].sum() < ri[i]) and (dataframe.iloc[randomprey].sum() != 1):
                randomprey2 = np.random.randint(i,s) 
                dataframe[dataframe.columns[i]].iloc[randomprey2] = 1
    if (dataframe.sum().sum() == sum(ri)) and check == True: print("sanity check")
    dataframe.iloc[s-1][dataframe.columns[:-1]] = np.zeros(s-1) # in order to have at least a top species
    A = np.array(dataframe)
    
    # delete isoleted nodes
    rowsum = A.sum(axis = 1)
    colsum = A.sum(axis = 0)
    mask = (rowsum + colsum) == 0
    A = A[~mask][:,~mask]       
    
    return A.astype(int)










########################## GENERATE DATASETS FOR COMPARISON ####################################


def within_xpercent(model,S,C,xpercent):

    """
    finds a generated model that satisfies certain properties:
    1) Smodel is within x% of S   2) Cmodel is within x% of C   3) The trophic level computation converges
    Args:
        model    :   model generation function (i.e. generate_nice(s,c,seed) )
        S        :   required number of species of the generated model
        C        :   required connectance value of the generated model
        xpercent :   relative tollerance on the fluctuations of the real values around the requested ones
    Out:
        A        :   adj matrix of the model with the requested properties
        seed     :   seed used to generate A with model generaton function
        Ceff     :   C    used to generate A with model generaton function
        Seff     :   S    used to generate A with model generaton function
    """
    
    debug = False
    
    i = 0
    
    Ceff = C
    Seff = S
    
    found = False
    while( not found ):
        
        seed = np.random.randint(10000000)+i
        A    = model(Seff,Ceff,seed=seed)
        
        CA   = compute_connectance(A)
        SA   = len(A)
        
        if( np.abs((CA-C)/C)<(xpercent/100) and np.abs((SA-S)/S)<(xpercent/100) ):
            
            try:
                compute_trophic_levels(A,df=False)
                found=True
            except:
                if debug: print("Rejectef because of TL\n\n")
                found=False
        else:
            
            # modify Ceff to allow/fasten convenrgence
            Ceff = Ceff+C*(xpercent/100) if CA<C else Ceff-C*(xpercent/100) # modifies Ceff of a quantity equal to the allowed percent error

            # modify Seff to allow/fasten convenrgence
            Seff = int(Seff+np.ceil(S*(xpercent/100)) if SA<S else Seff-np.ceil(S*(xpercent/100))) # modifies Seff of a quantity equal to the allowed percent error

        
        i += 1
        if debug: print(f"{i}:\tCg={CA}\tCeff={Ceff}\tC={C}\t:\tSg={SA}\tSeff={Seff}\tS={S}\t:\tseed={seed}")
        
    
    return A, seed, Ceff, Seff




def generate_FW_DATA(exp_data,models,SEED=None,CEFF=None,SEFF=None,xpercent=None):
    
    """
    When SEED, CEFF, and SEFF are provided, the function outputs "DATA" which is the dataframe of adj
    matrices. The first column contains the experimental ones, the other the generated ones, generated 
    according to the corresponding values of seed Ceff Seff.
    When SEED, CEFF, and SEFF are not provided the function computes them a new and provides them in
    the outputs, tigether with "DATA"
    Args:
        exp_data    :   ds.Series of experimental adjacency matrices, labelled by their name (i.e. "FW_008")
        models      :   ds.Series of model generation functions, labelled by their name (i.e. "nested")
        SEED        :   pd.DataFrame of seed values passed to the model generation functions, 
        CEFF        :   pd.DataFrame of Ceff values passed to the model generation functions
        SEFF        :   pd.DataFrame of Ceff values passed to the model generation functions
                        columns = (i.e. "niche", "nested", ...)    idx = (i.e. "FW_004", "FW_008",...)
        xpercent    :   when CEFF SEFF and SEED are not provided, the precision with wich the models are generated
    Out:
        DATA        :   pd.DataFrame of adj matrices (both for experimental and generated datasets)
        SEED        :   pd.DataFrame of seeds        (only for generated dateasets)
        CEFF        :   pd.DataFrame of Ceff         (only for generated dateasets)
        SEFF        :   pd.DataFrame of Seff         (only for generated dateasets)
    """
    
    if (SEED is not None) and (CEFF is not None) and (SEFF is not None):
        
        GEN_FW = pd.DataFrame(\
            data    = [ [ pd.DataFrame(models.iloc[j]( s   = SEFF.iloc[i,j],
                                                       c   = CEFF.iloc[i,j],
                                                       seed= SEED.iloc[i,j]    ))  for j in range(len(models)) ] for i in range(len(exp_data)) ],
            columns = models.index,
            index   = exp_data.index
            )
        
        DATA = pd.concat([pd.DataFrame({"experimetal":exp_data}),GEN_FW],axis=1)
        
        return DATA
        
    else:
        
        GEN_FW = pd.DataFrame({})
        SEED   = pd.DataFrame({})
        CEFF   = pd.DataFrame({})
        SEFF   = pd.DataFrame({})

        for m,m_nam in zip(models,models.index):

            Agen_list = []
            seed_list = []
            Ceff_list = []
            Seff_list = []
            
            for A,A_nam in zip(exp_data,exp_data.index):

                print(A_nam,m_nam)

                Agen,seed,Ceff,Seff = within_xpercent(m,len(A),compute_connectance(A),xpercent=xpercent)

                Agen_list.append(pd.DataFrame(Agen))
                seed_list.append(seed)
                Ceff_list.append(Ceff)
                Seff_list.append(Seff)
        
            GEN_FW[m_nam] = pd.Series(data=Agen_list,index=exp_data.index)
            SEED[m_nam]   = pd.Series(data=seed_list,index=exp_data.index)
            CEFF[m_nam]   = pd.Series(data=Ceff_list,index=exp_data.index)
            SEFF[m_nam]   = pd.Series(data=Seff_list,index=exp_data.index)
            
        DATA = pd.concat([pd.DataFrame({"experimental":exp_data}),GEN_FW],axis=1)

        return DATA, SEED, CEFF, SEFF