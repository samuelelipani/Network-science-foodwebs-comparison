import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def normalize_columns(df):
    weights = df.sum(axis=0).to_numpy()
    weights = np.where(weights > 0 , weights, 1 )
    return df/weights

# DEFINITIVE IMPORT FUNCTION ###
def load_df(file_csv, normalize = True , 
            headers = True , to_bool = False ,
            to_matrix = False, **kwargs):
    
    """Loads dataframe of foodwebs.
    
        Args:
            file_csv (string)    : Filename to be loaded (WARNING! the file needs to be in the directory `data`)
            
            normalize (bool)     : Whether to normalize columns or not 
                                    defaults to = True
            headers (bool)       : If the input file contains headers or not
                                    defaults to = True
            to_bool (bool)       : If the dataset needs to be converted to bool (losing all info on weights)
                                    defaults to = False

            to_matrix(bool)      : If true a tuple will be returned containing the connection matrix and a list
                                    of the headers
                                    defaults to = False
                                    
            **kwargs             : Arguments to be passed to pd.read_csv. 
            
        Returns: Either a df of the foodweb or the adjacency matrix and a list of headers
            
            """
            
    index_col, header = (0, 0) if headers else (None, None) 

    df = pd.read_csv(file_csv , sep = ",", index_col = index_col , header = header , **kwargs ).fillna(0)
    
    df = df.combine_first(df.T*0).fillna(0)

    df = df > 0 if to_bool else df
    
    df = normalize_columns(df) if normalize else df
    
    return (np.array(df) , list(df.columns)) if to_matrix else df


