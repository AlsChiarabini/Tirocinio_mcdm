import numpy as np
import pandas as pd

def give_rank(M):
    diffM=np.abs(np.diff(M ,axis=0 , prepend=1))
    rank=np.cumsum((np.sum(diffM,1)!=0).astype(int))
    return rank

def compute_normalized_rank_and_median(df):
    rank_df=df.rank()
    rank_df=rank_df/rank_df.max()
    rank_df['median']=rank_df.median(1)
    return rank_df['median']

def compute_disentanglement(sorted_M,df):
    temp_df=pd.DataFrame(sorted_M)
    temp_df['median']=compute_normalized_rank_and_median(df)
    temp_df=temp_df.sort_values(by='median',ascending=False)
    temp_df['disentagle']=1.
    cols_to_groupby = list(set(temp_df.columns) - {'median', 'disentagle'})
    temp_df['disentagle']=0.01*temp_df.groupby(cols_to_groupby)['disentagle'].cumsum()-0.01
    temp_df=temp_df.sort_index()
    return temp_df['disentagle']


def compute_rank_and_mean(df):
    rank_df=df.rank(ascending=True)
    rank_df=rank_df/rank_df.max()
    rank_df['mean']=rank_df.mean(1)
    return rank_df['mean']

def compute_zscores(df):
    score=((df-df.mean())/df.std())
    score=score.mean(axis=1)
    return score



def voting_ordering(M, method='majority', disentagle_df=None):

    R , C = M.shape
    
    if 'r_' in method:
        M=-M
        disentagle_df=-disentagle_df

    if method=='mean_rank':
        grades=compute_rank_and_mean(disentagle_df)
        grades=grades.rank(ascending=False).values ##representing stocks' ranks Stambaugh 2015
        new_indices=np.argsort(grades)
        placements=[(ind, rank) for ind, rank in zip(new_indices,grades[new_indices])]
        sorted_M=None

    elif method=='zscore':
        grades=compute_zscores(disentagle_df)
        grades=pd.Series(grades).rank(ascending=False).values ##representing stocks' ranks Stambaugh 2015
        new_indices=np.argsort(grades)
        placements=[(ind, rank) for ind, rank in zip(new_indices,grades[new_indices])]
        sorted_M=None

    else:

        M = np.sort(M, axis=1, kind='mergesort')
        
        if hasattr(disentagle_df,'shape'):
            # conto le righe uguali, aggiunto un +0.01 a seconda del ranking indotto dai percentili. 
            # outcome sarà un vettor con 0 (se non ci sono pareggi) o i*0.01 se ci sono pareggi [0,0,0,0,0.1,0.,0.1,0.2,...]
            disentaglement=compute_disentanglement(M,
                                                disentagle_df).values
    
   
        new_indices = voting_ordering_indexing(M = M, 
                                            original_index = np.arange(R), 
                                            method = method)
        
        sorted_M=M[new_indices,:]
        grades=give_rank(sorted_M)
    
        if hasattr(disentagle_df,'shape'):
            ## IF PM matrix is provided, disentagle elements having equal rank
            ## grades = grades + disentanglement
            grades=grades+disentaglement[new_indices]
        
        placements=[(ind, rank) for ind, rank in zip(new_indices,grades)]
    
    grades=len(new_indices) - grades[np.argsort(new_indices)]
    
    
    return grades, placements, new_indices, sorted_M
    
 


def voting_ordering_indexing(M , original_index, method):
    '''
    M = [RxC] matrix to be ordered 
    original_index = array to be initilialized as np.range(R)
    method = ['majority','lex','dlex']
    
    '''

    R , C = M.shape
    
    ## 11 colonne vorremmo la sesta -> il sesto elemento ha indice 5 (C+1)/2 -1 -> (C-1)/2
    ## 10 colonne il majority ordinato dal più grande al più piccolo ci dice che 
    ## dovremmo voler prendere il sesto elemento. Avendo ordinato al contrario, vogliamo il quinto.
    ## pertanto il quinto elemento ha indice 4  (C/2-1) 
    
    if method=='majority' or method=='r_majority':
        if C % 2 == 0:
            selected_col = int(C/2)-1    
        else:
            selected_col = int((C-1)/2)
    elif method=='lex' or  method=='r_lex':
        selected_col = -1
    elif method=='dlex':
        selected_col=0
    elif method=='75q' or method=='r_75q':
        selected_col = np.quantile(np.arange(C),0.75, method='higher')
    elif method=='90q' or method=='r_90q':
        selected_col = np.quantile(np.arange(C),0.25, method='higher')

        

    c = M[:, selected_col]
    arg_sort_indices = (-c).argsort()
    sort_index = original_index[arg_sort_indices]
    
   
    if C ==1:
        return sort_index
    
    new_index=[]
    
    M = M[arg_sort_indices,:]
    
    vals , counts = np.unique(c[arg_sort_indices], return_counts=True)
    vals=vals[::-1]
    counts=counts[::-1]
  
    checked_rows=0
    for i,el in enumerate(counts):
        if el>1:
            check, sort_index = np.split(sort_index,[el])
            slice_M=np.delete(M,selected_col,axis=1)[checked_rows:checked_rows+el,:]
            permutation_check = voting_ordering_indexing(slice_M,check,method)
            new_index+=list(permutation_check)
            checked_rows+=el
            
        elif el==1:
            to_add, sort_index = np.split(sort_index,[el])
            new_index+=list(to_add)
            checked_rows+=1

    return new_index


def weigthed_voting(M, weights, method='majority', disentagle_df=None):
    
    col_list=np.split(M,M.shape[1],1)

    if hasattr(disentagle_df,'shape'):
        disentagle_df_list=np.split(disentagle_df,disentagle_df.shape[1],1)
        weighted_disentangle_df=[]
    
    assert len(col_list) == len(weights), "Grades and weights have different lengths"
    assert len(disentagle_df_list) == len(weights), "Disentaglement df and weights have different lengths"
    weighted_M=[]
    
    for i, w in enumerate(weights):
        weighted_M += w * col_list[i:i+1]
        if hasattr(disentagle_df,'shape'):
            weighted_disentangle_df += w * disentagle_df_list[i:i+1]
        
    weighted_M = np.concatenate(weighted_M,1)

    if hasattr(disentagle_df,'shape'):
        weighted_disentangle_df = pd.DataFrame(np.concatenate(weighted_disentangle_df,1))
    else:
        weighted_disentangle_df=None

    
    grades, placements, new_indices, sorted_weighted_M = voting_ordering(weighted_M, method=method, disentagle_df=weighted_disentangle_df)
    
    return grades, placements, new_indices, sorted_weighted_M
    

   
def select_votes(M, method='majority'):

    R , C = M.shape
    
    M = np.sort(M, axis=1, kind='mergesort')
    
    if method=='majority' or method=='r_majority':
        if C % 2 == 0:
            selected_col = int(C/2)-1    
        else:
            selected_col = int((C-1)/2)
    elif method=='lex' or  method=='r_lex':
        selected_col = -1
    elif method=='dlex':
        selected_col=0
    elif method=='75q' or method=='r_75q':
        selected_col = np.quantile(np.arange(C),0.75, method='higher')
    elif method=='90q' or method=='r_90q':
        selected_col = np.quantile(np.arange(C),0.25, method='higher')

        

    c = M[:, selected_col]
    
    return c