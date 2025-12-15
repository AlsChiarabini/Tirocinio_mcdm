import random
from MJP.ordering_MJ_lex_dlex import *
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr, ttest_1samp, wilcoxon
import statsmodels.api as sm
import copy
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from joblib import Parallel, delayed
pd.set_option('mode.chained_assignment', None)


def custom_qcut(series, num_cat):
    """
    Custom qcut function to handle coinciding bin edges by collapsing bins and assigning average labels.
    
    Parameters:
    - series: A pandas Series to be binned.
    - num_cat: The number of quantile categories to create.
    
    Returns:
    - A pandas Series with adjusted bin labels.
    """
    # Step 1: Get initial quantile bins with pd.qcut
    try:
        bins, bin_edges = pd.qcut(series, num_cat, labels=False, retbins=True, duplicates='drop')
    except ValueError as e:
        print(f"Error: {e}")
        return pd.Series(np.nan, index=series.index)

    # Step 2: Identify unique bin edges and handle coinciding edges
    unique_bin_edges = np.unique(bin_edges)
    
    # Step 3: Calculate the number of bins after collapsing
    num_unique_bins = len(unique_bin_edges) - 1

    # Step 4: Handle collapsing of bins if needed
    if num_unique_bins < num_cat:
        # Create a mapping to average the bin labels where edges coincide
        bin_mapping = {}
        cumulative_sum = 0
        current_label = 1
        
        for i in range(num_unique_bins):
            if bin_edges[i] == bin_edges[i + 1]:
                bin_mapping[i] = current_label - 0.5
            else:
                bin_mapping[i] = current_label
                current_label += 1

        # Assigning the last bin
        bin_mapping[num_unique_bins] = current_label - 0.5 if bin_edges[-2] == bin_edges[-1] else current_label
        
        # Apply the mapping to the bins
        bins = bins.map(bin_mapping)

    else:
        bins = bins + 1  # If no collapsing needed, increment all bins by 1 for 1-based index

    return bins


def select_voters_and_signs(data,
                            voting_window,
                            default_voters,
                            portfolios,
                            alpha=0.05,
                            min_voters=3):
    
    l_window=data - pd.offsets.MonthEnd(voting_window)
    u_window=data

    '''
    voters=[]
    signs=[]
    for factor in default_voters: 
        mask_rolling=(portfolios[factor].index>l_window)&(portfolios[factor].index<=u_window) 

        #ret_vals=portfolios[factor].loc[mask_rolling,[f'port{i}' for i in range(1,11)]].mean()
        #X = sm.add_constant(np.arange(1,11))
        #model = sm.OLS(ret_vals,
        #            X)
        #results = model.fit()
        #stat , pval = results.params['x1'], results.pvalues['x1']

        #ret_vals=portfolios[factor].loc[mask_rolling,[f'port{i}' for i in range(1,11)]].mean()
        #stat_c , pval_c = spearmanr(np.arange(1,11),ret_vals)
        
        #ret_vals=portfolios[factor].loc[mask_rolling,'long_short'].values
        #stat,pval = ttest_1samp(ret_vals, popmean=0.0, nan_policy='omit') 

        #if pval<alpha:
        #    print(factor,stat,pval)
        #    signs.append(np.sign(stat))  
        #    voters.append(factor)  
         
        ret_vals=portfolios[factor].loc[mask_rolling,'long_short'].values
        _,pval_gtr = wilcoxon(ret_vals, alternative='greater',  nan_policy='omit')
        _,pval_less = wilcoxon(ret_vals, alternative='less',  nan_policy='omit')

        if pval_gtr<alpha:
            print(factor,pval_gtr,+1)
            signs.append(1.)  
            voters.append(factor)  
        elif pval_less<alpha:
            print(factor,pval_less,-1)
            signs.append(-1.)  
            voters.append(factor)  '''
        
    sign_list=[]
    pvals_list=[]
    for factor in default_voters: 
        mask_rolling=(portfolios[factor].index>l_window)&(portfolios[factor].index<=u_window) 
        ret_vals=portfolios[factor].loc[mask_rolling,'long_short'].values
        _,pval_gtr = wilcoxon(ret_vals, alternative='greater',  nan_policy='omit')

        sign_list.append(1 if pval_gtr<1-pval_gtr else -1)
        pvals_list.append(np.minimum(pval_gtr,1-pval_gtr))

    # Define the condition (example: elements greater than 4)
    condition = lambda x: x < alpha

    # Find elements in list1 that satisfy the condition
    satisfying_elements = [(i, sign_list[i]) for i in range(len(pvals_list)) if condition(pvals_list[i])]


    if len(satisfying_elements) >= min_voters:
        print('at least min_voters')
        # More than 3 elements satisfy the condition
        selected_indices = [index for index, value in satisfying_elements]
    else:
        print('less than min_voters')
        # fewer than 3 elements satisfy the condition
        # Find the 3 smallest elements from list1
        smallest_elements = sorted(range(len(pvals_list)), key=lambda i: pvals_list[i])[:min_voters]
        selected_indices = smallest_elements

    voters=[default_voters[i] for i in selected_indices]
    signs=[sign_list[i] for i in selected_indices]
    return voters, signs


# Define function to split rows into deciles and aggregate PERMNO values
def split_into_deciles(group, factor_column,num_port,remove_outliers,outliers):
    # Sort values by the specified factor column
    group_sorted = group.sort_values(by=factor_column)
    # Drop NaN values in the specified factor column
    group_sorted = group_sorted.dropna(subset=[factor_column])
    ## remove outliers if requested
    if remove_outliers:
        inf,sup = outliers
        if not sup:
            group_sorted = group_sorted[group_sorted["me_lag"] >= group_sorted[inf]]
        elif not inf:
            group_sorted = group_sorted[group_sorted["me_lag"] <= group_sorted[sup]]
        else:
            group_sorted = group_sorted[np.logical_and(group_sorted["me_lag"] >= group_sorted[inf], 
                                                        group_sorted["me_lag"] <= group_sorted[sup])]
        
    # Split rows into 10 equally sized bins
    group_sorted['portfolio'] = pd.qcut(group_sorted[factor_column], q=num_port+1, labels=False)
    # Group by 'decile' and aggregate PERMNO values
    #decile_permnos = group_sorted.groupby('portfolio')['PERMNO'].apply(list).reset_index()
    #decile_permnos['reallocation_date'] = group['medate'].iloc[0]  # Using iloc[0] since all rows have the same date
    
    group_sorted.pop(factor_column)
    return group_sorted


# Define function to split rows into deciles and aggregate PERMNO values
def split_into_deciles_inclusive(group, factor_column,num_port,remove_outliers,outliers):
    # Sort values by the specified factor column
    group_sorted = group.sort_values(by=factor_column)
    # Drop NaN values in the specified factor column
    group_sorted = group_sorted.dropna(subset=[factor_column])
    ## remove outliers if requested
    if remove_outliers:
        inf,sup = outliers
        if not sup:
            group_sorted = group_sorted[group_sorted["me_lag"] >= group_sorted[inf]]
        elif not inf:
            group_sorted = group_sorted[group_sorted["me_lag"] <= group_sorted[sup]]
        else:
            group_sorted = group_sorted[np.logical_and(group_sorted["me_lag"] >= group_sorted[inf], 
                                                        group_sorted["me_lag"] <= group_sorted[sup])]
    # Split rows into 10 equally sized bins
    cuts=np.arange(0, 101, num_port)
    percentiles = np.percentile(group_sorted[factor_column], cuts)
    concat_fitered_data = []

    # Iterate through consecutive pairs of deciles
    for i in range(len(percentiles) - 1):
        # Find data within the range of the current pair of deciles
        filtered_data = group_sorted[(group_sorted[factor_column] >= percentiles[i]) & (group_sorted[factor_column] <= percentiles[i+1])]
        filtered_data['portfolio']=i+1
        concat_fitered_data.append(filtered_data)
        # Store the index values associated with the filtered data in the dictionary
        #index_dict[i+1] = [filtered_data['PERMNO'].tolist(),filtered_data['medate'].iloc[0]]

    #percentiles_permnos=pd.DataFrame(index_dict).T.reset_index()
    #percentiles_permnos.columns=['portfolio','PERMNO','reallocation_date']
    concat_fitered_data=pd.concat(concat_fitered_data,axis=0).reset_index(drop=True)
    concat_fitered_data.pop(factor_column)
    return concat_fitered_data


def portfolio_formation(df, 
                           factor, 
                           holding_periods, 
                           num_port, 
                           lag=0, 
                           reverse=False, 
                           remove_outliers=False,
                           outliers=[],
                           inclusive=False):
    '''
    1) definisci date di riallocazione partendo da lag e distanziandole di K mesi
    2) definisci i periodi di investimento che sono shiftati di un mese rispetto alle date di riallocazione 
    3) aggiungi colonna df['investment_periods'] categorica
    4) per ogni periodo, identifica la data di riallocazione
              se mj: identifica il numero di rank individuati da mj e fai la divisione in percentili basandoti sul RANK. 
                     (caso alternativo non sensato)
                     dividi in 10 portafogli
              altrimenti: taglia in 10 portafogli
              per ogni periodo:
                  lista stocks nei portafogli
                  identifica temp_df: sotto-df contente le stock di un portafoglio nelle date del periodo
                  calcolo rendimenti del portafoglio nel periodo
                  calcolo rendimenti pesati del portafoglio nel periodo
    5) concatena i portafogli
    6) calcola strategie long-short e rendimenti cumulati
                  
    '''
    print(factor)
    
    
    df.loc[:,'medate'] = df.loc[:,'date'].copy() + pd.offsets.MonthEnd(0)
        
        
    portfolios_returns={}
    portfolios_weighted_returns={}
    for i in range(1,num_port+1):
        portfolios_returns[i]=[]
        portfolios_weighted_returns[i]=[]

    data_reallocation_min=df['medate'].min()+pd.offsets.MonthEnd(lag)
    data_reallocation_max=df['medate'].max()
    data_reallocation=[data_reallocation_min]
    while data_reallocation[-1] + pd.offsets.MonthEnd(holding_periods)<data_reallocation_max:
        data_reallocation.append(data_reallocation[-1] + pd.offsets.MonthEnd(holding_periods))

    add_cols=[]
    if remove_outliers:
        inf,sup = outliers
        if inf:
            add_cols.append(inf)
        if sup:
            add_cols.append(sup)

    df_reallocation = df.loc[df['medate'].isin(data_reallocation),['PERMNO','medate','me_lag',factor]+add_cols]



    # Group by 'date'
    # Apply the function to each group and concatenate the results
    # Apply the modified function to each group and concatenate the results
    if inclusive:
        fn=split_into_deciles_inclusive
    else:
        fn=split_into_deciles
    port_data = df_reallocation.groupby('medate').apply(fn, 
                                                        factor_column=factor, 
                                                        num_port = num_port,
                                                        remove_outliers=remove_outliers,
                                                        outliers=outliers).reset_index(drop=True)

    percentiles_permnos = port_data.groupby(['medate','portfolio'])['PERMNO'].apply(list).reset_index()

    reallocation_rates=[]
    for port_num in range(1,num_port+1):

        group_df = percentiles_permnos.loc[percentiles_permnos['portfolio']==port_num]
        # Shift the 'PERMNO' column by one row to get the previous values
        group_df.loc[:,'previous_PERMNO'] = group_df['PERMNO'].shift(1)

        group_df=group_df
        # Compute the set difference between consecutive rows of 'PERMNO' and store in a new column
        group_df.loc[:,'B'] = group_df.apply(lambda row: list(set(row['PERMNO']) - set(row['previous_PERMNO'])) if row['previous_PERMNO'] else list([]) , axis=1)
        group_df.loc[:,'S'] = group_df.apply(lambda row: list(set(row['previous_PERMNO']) - set(row['PERMNO'])) if row['previous_PERMNO'] else list([]), axis=1)
        group_df.loc[:,'EW_turnover']= group_df.apply(
            lambda row: 0.5*(len(row['B'])/len(row['PERMNO'])+ len(row['S'])/len(row['previous_PERMNO'])) if (row['B'] and row['S']) else np.nan, axis=1)

        def weighted_turnover(group):
            date= group.name
            current_permnos = group_df.loc[group_df['medate']==date]['PERMNO'].tolist()[0]
            previous_permnos = group_df.loc[group_df['medate']==date]['previous_PERMNO'].tolist()[0]
            B_permnos = group_df.loc[group_df['medate']==date]['B'].tolist()[0]
            S_permnos = group_df.loc[group_df['medate']==date]['S'].tolist()[0]
            
            wS = group.loc[group['PERMNO'].isin(S_permnos)]['me_lag'].sum()  if S_permnos else np.nan
            wB = group.loc[group['PERMNO'].isin(B_permnos)]['me_lag'].sum() if B_permnos else np.nan
            w = group.loc[group['PERMNO'].isin(current_permnos)]['me_lag'].sum() if current_permnos else 1
            w_previous = group.loc[group['PERMNO'].isin(previous_permnos)]['me_lag'].sum() if previous_permnos else 1
            group.loc[:,'VW_turnover'] = 0.5*(wS/w_previous + wB/w)
            return group

        decile_permnos_mk=port_data.loc[port_data['portfolio']==port_num]
        vw_turnover = decile_permnos_mk.groupby('medate').apply(weighted_turnover).reset_index(drop=True)[['medate','VW_turnover']].drop_duplicates()
        vw_turnover.loc[:,'portfolio']=port_num
        reallocation_rates.append(group_df.merge(vw_turnover,on=['portfolio','medate'],how='outer')[['portfolio','EW_turnover','VW_turnover']].mean())
    
    reallocation_rates = pd.concat(reallocation_rates,axis=1).T
    for date in data_reallocation:
        temp_df=df.loc[(df['medate']>date) & (df['medate']<= date+pd.offsets.MonthEnd(holding_periods))]
        for portf in range(1,num_port+1):
            list_stocks=percentiles_permnos.loc[(percentiles_permnos['medate']==date)& (percentiles_permnos['portfolio']==portf)]['PERMNO'].tolist()[0]
            port_df=temp_df.loc[(temp_df['PERMNO'].isin(list_stocks))]

            portfolios_returns[portf].append(port_df.groupby('medate')['RET_RF'].mean())
            if holding_periods!=1 and len(temp_df['medate'].unique())!=1:
                portfolios_weighted_returns[portf].append(
                    port_df.groupby('medate').apply(
                        lambda x: x.assign
                        (ret=np.nanmean(x["RET_RF"] * x["me_lag"]) / np.nanmean(x["me_lag"]))
                    ).reset_index(drop=True)[['ret','medate']].drop_duplicates().set_index('medate').squeeze())
            else:
                if holding_periods!=1:
                    portfolios_weighted_returns[portf].append(
                        port_df.groupby('medate').apply(
                            lambda x: x.assign
                            (ret=np.nanmean(x["RET_RF"] * x["me_lag"]) / np.nanmean(x["me_lag"]))
                        ).reset_index(drop=True)[['ret','medate']].drop_duplicates().set_index('medate')['ret'])
                else:
                    portfolios_weighted_returns[portf].append(
                        port_df.groupby('medate').apply(
                            lambda x: x.assign
                            (ret=np.nanmean(x["RET_RF"] * x["me_lag"]) / np.nanmean(x["me_lag"]))
                        ).reset_index(drop=True)[['ret','medate']].drop_duplicates().set_index('medate'))
    
    
    for i in range(1,num_port+1):
        portfolios_returns[i]=pd.concat(portfolios_returns[i])
        portfolios_weighted_returns[i]=pd.concat(portfolios_weighted_returns[i])
        if holding_periods==1:
            portfolios_weighted_returns[i]=portfolios_weighted_returns[i].squeeze()


    # portfolios_stocks[i][j]   i=numero portafoglio.  j=numero periodo
    # portfolios_weighted_returns[j] returns del portafoglio j su tutta la serie storica

    portfolios_weighted_returns=pd.DataFrame(portfolios_weighted_returns)
    portfolios_returns=pd.DataFrame(portfolios_returns)


    #==================================================
    # Long-Short Portfolio Returns  
    #==================================================
    # Add prefix port in front of each column
    portfolios_returns.columns = ['port' + str(col) for col in portfolios_returns.columns]
    portfolios_weighted_returns.columns = ['port' + str(col) for col in portfolios_weighted_returns.columns]


    # Evaluate long-short returns
    if reverse == 0:
        portfolios_returns['long_short'] = portfolios_returns[f'port{num_port}'] - portfolios_returns['port1']   
        portfolios_weighted_returns['long_short'] = portfolios_weighted_returns[f'port{num_port}'] - portfolios_weighted_returns['port1']
    else:
        portfolios_returns['long_short'] = portfolios_returns['port1'] - portfolios_returns[f'port{num_port}']  
        portfolios_weighted_returns['long_short'] = portfolios_weighted_returns['port1'] - portfolios_weighted_returns[f'port{num_port}']

    return portfolios_returns, portfolios_weighted_returns, reallocation_rates


def invert_signs_in_df(df,factors,signs):
    df0=df[factors].copy()
    for id, factor in enumerate(factors):
        if type(signs) is list:
            df0.loc[:,factor]=signs[id]* df0.loc[:,factor]
        elif type(signs) is dict:
            df0.loc[:,factor]=signs[factor]* df0.loc[:,factor]          
    return df0

'''
def compute_categorical_votes_qcut(subdf,voters,signs,num_cat):
    df=subdf.copy()
    df=invert_signs_in_df(df,voters,signs)
    df=df.apply(lambda x: pd.qcut(x, num_cat, labels=False), axis=0)+1
    return df'''


def compute_categorical_votes_qcut(subdf,voters,signs,num_cat):
    df=subdf.copy()
    df=invert_signs_in_df(df,voters,signs)
    try:
        df=df.apply(lambda x: pd.qcut(x, num_cat, labels=False), axis=0)+1
    except ValueError:
        # Apply the custom qcut function to each column in the DataFrame
        df = df.apply(lambda x: custom_qcut(x, num_cat), axis=0)
    return df



def map_list(lst):
    return pd.Series(lst)


def flatten(xss):
    return [x for xs in xss for x in xs]


def fill_nan_with_median(row):
    median = row.median(skipna=True)
    return row.fillna(median)


def fill_nan_with_min(row):
    minim = row.min(skipna=True)
    return row.fillna(minim)


def fill_nan_with_mean(row):
    mean = row.mean(skipna=True)
    return row.fillna(mean)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def compute_cat_votes(voter_values,num_cat,sign):
    cat_voter_values=np.array(voter_values)
    notna_indices=np.where(~np.isnan(cat_voter_values))[0]
    not_na_voter_values = cat_voter_values[notna_indices]  

    if sign>0:
        list_index=list(split(np.argsort(not_na_voter_values).tolist(), num_cat)) 
    else:
        list_index=list(split(np.argsort(not_na_voter_values)[::-1].tolist(), num_cat)) 

    vote_values=flatten([len(l)*[i+1] for i,l in enumerate(list_index)])
    list_index=flatten(list_index)

    df_vote=pd.DataFrame(np.array([list_index,vote_values]).T)
    vote_values = df_vote.sort_values(by=0)[1].to_numpy()
    cat_voter_values[notna_indices]=vote_values
    return cat_voter_values


def treat_nan_values_MJ(M,treat_na_mj):
    if treat_na_mj=='drop':
        return M.dropna()
    elif treat_na_mj=='median':
        return M.apply(fill_nan_with_median, axis=1)
    elif treat_na_mj=='min':
        return M.apply(fill_nan_with_min, axis=1)
    elif treat_na_mj=='mean':
        return M.apply(fill_nan_with_mean, axis=1)


def compute_ranking(M,treat_na_mj,method,disentagle_df=None):  
    M=treat_nan_values_MJ(M,treat_na_mj)

    _, placementsMJ, _, _ = voting_ordering(M = M, 
                                            method = method,
                                            disentagle_df=disentagle_df)
    idx=np.array([el[0] for el in placementsMJ])
    rank=np.array([el[1] for el in placementsMJ])
    mjfactor=rank[np.argsort(idx)]
    return mjfactor


def compute_categorical_votes(subdf,voters,signs,num_cat):
    votes=[]
    for i,voter in enumerate(voters): 
        voter_values=subdf[voter].to_list()
        c_voter=compute_cat_votes(voter_values,num_cat,signs[i])
        votes.append(c_voter)
    return np.stack(votes).T


def mask_rolling_window(df,reallocation_date,mj_window,date_col='medate'):
    l_window=reallocation_date - pd.offsets.MonthEnd(mj_window)
    u_window=reallocation_date
    mask_rolling=(df[date_col]>l_window)&(df[date_col]<=u_window) 
    return mask_rolling


def rank_rolling(df,
                 data_reallocation,
                 configuration):
    
    basic_portfolios=configuration['basic_portfolios']
    basic_weighted_portfolios=configuration['basic_weighted_portfolios']

    default_voters=configuration['default_voters']
    default_signs=configuration['default_signs']
    default_wvoters=configuration['default_wvoters']
    default_wsigns=configuration['default_wsigns']

    mj_window=configuration['mj_window']
    voting_window=configuration['voting_window']
    
    verbose=configuration['verbose']
    method=configuration['method']
    num_cat=configuration['num_cat']
    treat_na_mj=configuration['treat_na_mj']
    alpha=configuration['p_threshold']
    min_voters=configuration['min_voters']

    mj_voters={
        'date':[],
        'mj':[],
        'wmj':[]}
    df.loc[:,[f'r_{v}' for v in default_voters]]=np.nan
    df.loc[:,[f'wr_{v}' for v in default_wvoters]]=np.nan

    voters=default_voters
    wvoters=default_wvoters
    signs=default_signs
    wsigns=default_wsigns

    for data in data_reallocation:

        new_voters, new_signs = select_voters_and_signs(data,
                                                    voting_window,
                                                    default_voters,
                                                    basic_portfolios,
                                                    alpha=alpha,
                                                    min_voters=min_voters)
        new_wvoters, new_wsigns = select_voters_and_signs(data,
                                                        voting_window,
                                                        default_wvoters,
                                                        basic_weighted_portfolios,
                                                        alpha=alpha,
                                                        min_voters=min_voters)
        print(f'new voters/wvoters: {new_voters}/{new_wvoters}')

        if len(new_voters)>0:
            voters=new_voters
            signs=new_signs

        if len(new_wvoters)>0:
            wvoters=new_wvoters
            wsigns=new_wsigns

        mask_rolling=mask_rolling_window(df,data,mj_window)
        subdf=df.loc[mask_rolling]

        list_permno_on_reallocation_date=subdf.loc[subdf['medate']==data]['PERMNO'].unique().tolist()
        subdf=subdf.loc[subdf['PERMNO'].isin(list_permno_on_reallocation_date)]
        
        if verbose:
            print(f'date: {data}, voters= {len(voters)}, wvoters={len(wvoters)}')
        mj_voters['date'].append(data)
        mj_voters['mj'].append(len(voters))
        mj_voters['wmj'].append(len(wvoters))

        #remove rows having nans in the factors cells at reallocation dates
        #mask_factor_nans_at_reallocation_dates = ~(df['medate'].isin(data_reallocation) & df[factors].isna().any(axis=1))

        subdf.loc[:,[f'r_{v}' for v in voters]]=compute_categorical_votes_qcut(subdf,voters,signs,num_cat).values
        subdf.loc[:,[f'wr_{v}' for v in wvoters]]=compute_categorical_votes_qcut(subdf,wvoters,wsigns,num_cat).values
        
        for date in subdf['medate'].unique().tolist():
            
            date_df=subdf.loc[subdf['medate']==date]
            
            M=copy.copy(date_df[[f'r_{v}' for v in voters]])
            disentagle_df=copy.copy(date_df[[f'{v}' for v in voters]])
            disentagle_df=invert_signs_in_df(disentagle_df,voters,signs)
            mjfactor=compute_ranking(M,
                                     treat_na_mj,
                                     method,
                                     disentagle_df=disentagle_df)
            df.loc[df.index.isin(M.index),'mj']=mjfactor/max(mjfactor)

            wM=copy.copy(date_df[[f'wr_{v}' for v in wvoters]])
            disentagle_df=copy.copy(date_df[[f'{v}' for v in wvoters]])
            disentagle_df=invert_signs_in_df(disentagle_df,wvoters,wsigns)
            mjfactor=compute_ranking(wM,
                                     treat_na_mj,
                                     method,
                                     disentagle_df=disentagle_df)
            df.loc[df.index.isin(wM.index),'wmj']=mjfactor/max(mjfactor)

            
    df.loc[:,'mj']=-df.groupby(['PERMNO'])['mj'].rolling(mj_window, min_periods=1).mean().reset_index(level=0)['mj']
    df.loc[:,'wmj']=-df.groupby(['PERMNO'])['wmj'].rolling(mj_window, min_periods=1).mean().reset_index(level=0)['wmj']
    return df, mj_voters


def profile_rolling(df,
                 data_reallocation,
                 configuration):
    
    basic_portfolios=configuration['basic_portfolios']
    basic_weighted_portfolios=configuration['basic_weighted_portfolios']

    default_voters=configuration['default_voters']
    default_signs=configuration['default_signs']
    default_wvoters=configuration['default_wvoters']
    default_wsigns=configuration['default_wsigns']

    mj_window=configuration['mj_window']
    voting_window=configuration['voting_window']
    
    verbose=configuration['verbose']
    method=configuration['method']
    num_cat=configuration['num_cat']
    treat_na_mj=configuration['treat_na_mj']
    alpha=configuration['p_threshold']
    min_voters=configuration['min_voters']

    voters=default_voters
    wvoters=default_wvoters
    signs=default_signs
    wsigns=default_wsigns

    mj_voters={
        'date':[],
        'mj':[],
        'wmj':[]}
    df.loc[:,[f'av{f}' for f in default_voters]]=np.nan
    df.loc[:,[f'r_{v}' for v in voters]]=np.nan
    df.loc[:,[f'wr_{v}' for v in wvoters]]=np.nan
    df['mj']=np.nan
    df['wmj']=np.nan
    
    for data in data_reallocation:
        new_voters, new_signs = select_voters_and_signs(data,
                                                    voting_window,
                                                    default_voters,
                                                    basic_portfolios,
                                                    alpha=alpha,
                                                    min_voters=min_voters)
        new_wvoters, new_wsigns = select_voters_and_signs(data,
                                                        voting_window,
                                                        default_wvoters,
                                                        basic_weighted_portfolios,
                                                        alpha=alpha,
                                                        min_voters=min_voters)
        print(f'new voters/wvoters: {new_voters}/{new_wvoters}')

        if len(new_voters)>0:
            voters=new_voters
            signs=new_signs

        if len(new_wvoters)>0:
            wvoters=new_wvoters
            wsigns=new_wsigns

        mask_rolling=mask_rolling_window(df,data,mj_window)
        
        df_rolling=df.loc[mask_rolling]
        
        list_permno_on_reallocation_date=df_rolling.loc[df_rolling['medate']==data]['PERMNO'].unique().tolist()
        #compute rolling values of real-valued factor values
        mean_factors=df_rolling.loc[df_rolling['PERMNO'].isin(list_permno_on_reallocation_date)][['PERMNO']+default_voters].groupby('PERMNO').mean().reset_index()
        mask_reallocation_data_and_PERMNO=(df['medate']==data)&(df['PERMNO'].isin(list_permno_on_reallocation_date))
        df.loc[mask_reallocation_data_and_PERMNO,[f'av{f}' for f in default_voters]]=mean_factors[default_voters].values

        #print(df_rolling['medate'].unique().tolist())

        if verbose:
            print(f'date: {data}, voters= {len(voters)}, wvoters={len(wvoters)}')
        mj_voters['date'].append(data)
        mj_voters['mj'].append(len(voters))
        mj_voters['wmj'].append(len(wvoters))
        

        df.loc[mask_reallocation_data_and_PERMNO,[f'r_{v}' for v in voters]]=compute_categorical_votes_qcut(df.loc[mask_reallocation_data_and_PERMNO],
                                                                                                       [f'av{v}' for v in voters],
                                                                                                       signs,
                                                                                                       num_cat).values
        df.loc[mask_reallocation_data_and_PERMNO,[f'wr_{v}' for v in wvoters]]=compute_categorical_votes_qcut(df.loc[mask_reallocation_data_and_PERMNO],
                                                                                                         [f'av{wv}' for wv in wvoters],
                                                                                                         wsigns,
                                                                                                         num_cat).values
        
            
        date_df=df.loc[mask_reallocation_data_and_PERMNO]
        
        M=copy.copy(date_df[[f'r_{v}' for v in voters]])
        disentagle_df=copy.copy(date_df[[f'{v}' for v in voters]])
        disentagle_df=invert_signs_in_df(disentagle_df,voters,signs)
        mjfactor=compute_ranking(M,
                                 treat_na_mj,
                                 method,
                                 disentagle_df=disentagle_df)
        df.loc[df.index.isin(M.index),'mj']=-mjfactor
        
        wM=copy.copy(date_df[[f'wr_{v}' for v in wvoters]])
        disentagle_df=copy.copy(date_df[[f'{v}' for v in wvoters]])
        disentagle_df=invert_signs_in_df(disentagle_df,wvoters,wsigns)
        mjfactor=compute_ranking(wM,
                                 treat_na_mj,
                                 method,
                                 disentagle_df=disentagle_df)
        df.loc[df.index.isin(wM.index),'wmj']=-mjfactor
    return df, mj_voters


def vote_rolling(df,
                 data_reallocation,
                 configuration):
    
    basic_portfolios=configuration['basic_portfolios']
    basic_weighted_portfolios=configuration['basic_weighted_portfolios']

    default_voters=configuration['default_voters']
    default_signs=configuration['default_signs']
    default_wvoters=configuration['default_wvoters']
    default_wsigns=configuration['default_wsigns']

    mj_window=configuration['mj_window']
    voting_window=configuration['voting_window']
    
    verbose=configuration['verbose']
    method=configuration['method']
    num_cat=configuration['num_cat']
    treat_na_mj=configuration['treat_na_mj']
    alpha=configuration['p_threshold']
    min_voters=configuration['min_voters']

    voters=default_voters
    wvoters=default_wvoters
    signs=default_signs
    wsigns=default_wsigns


    mj_voters={
        'date':[],
        'mj':[],
        'wmj':[]}
    df['mj_vote']=np.nan
    df['wmj_vote']=np.nan
    df.loc[:,[f'r_{v}' for v in voters]]=np.nan
    df.loc[:,[f'wr_{v}' for v in wvoters]]=np.nan
    
    
    for data in data_reallocation:
        new_voters, new_signs = select_voters_and_signs(data,
                                                    voting_window,
                                                    default_voters,
                                                    basic_portfolios,
                                                    alpha=alpha,
                                                    min_voters=min_voters)
        new_wvoters, new_wsigns = select_voters_and_signs(data,
                                                        voting_window,
                                                        default_wvoters,
                                                        basic_weighted_portfolios,
                                                        alpha=alpha,
                                                        min_voters=min_voters)
        print(f'new voters/wvoters: {new_voters}/{new_wvoters}')

        if len(new_voters)>0:
            voters=new_voters
            signs=new_signs

        if len(new_wvoters)>0:
            wvoters=new_wvoters
            wsigns=new_wsigns

        mask_rolling=mask_rolling_window(df,data,mj_window)
        subdf=df.loc[mask_rolling]

        list_permno_on_reallocation_date=subdf.loc[subdf['medate']==data]['PERMNO'].unique().tolist()
        mask_reallocation_data_and_PERMNO=(df['medate']==data)&(df['PERMNO'].isin(list_permno_on_reallocation_date))
        
        #print(subdf['medate'].unique().tolist())

        if verbose:
            print(f'date: {data}, voters= {len(voters)}, wvoters={len(wvoters)}')
        mj_voters['date'].append(data)
        mj_voters['mj'].append(len(voters))
        mj_voters['wmj'].append(len(wvoters))

        subdf.loc[:,[f'r_{v}' for v in voters]]=compute_categorical_votes_qcut(subdf,voters,signs,num_cat).values
        subdf.loc[:,[f'wr_{v}' for v in wvoters]]=compute_categorical_votes_qcut(subdf,wvoters,wsigns,num_cat).values
        

        for date in subdf['medate'].unique().tolist():

            date_df=subdf.loc[subdf['medate']==date]
            M=copy.copy(date_df[[f'r_{v}' for v in voters]])
            M=treat_nan_values_MJ(M,treat_na_mj)
            df.loc[df.index.isin(M.index),'mj_vote']=select_votes(M)

            wM=copy.copy(date_df[[f'wr_{v}' for v in wvoters]])
            wM=treat_nan_values_MJ(wM,treat_na_mj)
            df.loc[df.index.isin(wM.index),'wmj_vote']=select_votes(wM)


        M_rolling=df.loc[mask_rolling & df['PERMNO'].isin(list_permno_on_reallocation_date)][['date','PERMNO','mj_vote']].groupby('PERMNO')['mj_vote'].agg(lambda x: x.tolist())
        M_rolling=M_rolling.apply(map_list)
        disentagle_df=copy.copy(date_df[[f'{v}' for v in voters]])
        disentagle_df=invert_signs_in_df(disentagle_df,voters,signs)
        df.loc[mask_reallocation_data_and_PERMNO,'mj']=-compute_ranking(M_rolling,
                                                                        treat_na_mj,
                                                                        method,
                                                                        disentagle_df=disentagle_df)

        wM_rolling=df.loc[mask_rolling & df['PERMNO'].isin(list_permno_on_reallocation_date)][['date','PERMNO','wmj_vote']].groupby('PERMNO')['wmj_vote'].agg(lambda x: x.tolist())
        wM_rolling=wM_rolling.apply(map_list)
        disentagle_df=copy.copy(date_df[[f'{v}' for v in wvoters]])
        disentagle_df=invert_signs_in_df(disentagle_df,wvoters,wsigns)
        df.loc[mask_reallocation_data_and_PERMNO,'wmj']=-compute_ranking(wM_rolling,
                                                                         treat_na_mj,
                                                                         method,
                                                                         disentagle_df=disentagle_df)
    return df, mj_voters


def profile_vote_rolling(df,
                 data_reallocation,
                 configuration):
    
    basic_portfolios=configuration['basic_portfolios']
    basic_weighted_portfolios=configuration['basic_weighted_portfolios']

    default_voters=configuration['default_voters']
    default_signs=configuration['default_signs']
    default_wvoters=configuration['default_wvoters']
    default_wsigns=configuration['default_wsigns']

    mj_window=configuration['mj_window']
    voting_window=configuration['voting_window']
    
    verbose=configuration['verbose']
    method=configuration['method']
    num_cat=configuration['num_cat']
    treat_na_mj=configuration['treat_na_mj']
    alpha=configuration['p_threshold']
    min_voters=configuration['min_voters']

    voters=default_voters
    wvoters=default_wvoters
    signs=default_signs
    wsigns=default_wsigns

    mj_voters={
        'date':[],
        'mj':[],
        'wmj':[]}
    df['mj_vote']=np.nan
    df['wmj_vote']=np.nan
    df.loc[:,[f'r_{v}' for v in voters]]=np.nan
    df.loc[:,[f'wr_{v}' for v in wvoters]]=np.nan
    df.loc[:,[f'{f}_median' for f in default_voters]]=np.nan
    df.loc[:,[f'w{f}_median' for f in default_wvoters]]=np.nan
       
    for data in data_reallocation:
        new_voters, new_signs = select_voters_and_signs(data,
                                                    voting_window,
                                                    default_voters,
                                                    basic_portfolios,
                                                    alpha=alpha,
                                                    min_voters=min_voters)
        new_wvoters, new_wsigns = select_voters_and_signs(data,
                                                        voting_window,
                                                        default_wvoters,
                                                        basic_weighted_portfolios,
                                                        alpha=alpha,
                                                        min_voters=min_voters)
        print(f'new voters/wvoters: {new_voters}/{new_wvoters}')

        if len(new_voters)>0:
            voters=new_voters
            signs=new_signs

        if len(new_wvoters)>0:
            wvoters=new_wvoters
            wsigns=new_wsigns

        mask_rolling=mask_rolling_window(df,data,mj_window)
        list_permno_on_reallocation_date=df.loc[df['medate']==data]['PERMNO'].unique().tolist()
        mask_reallocation_data_and_PERMNO=(df['medate']==data)&(df['PERMNO'].isin(list_permno_on_reallocation_date))
        
        subdf=df.loc[(mask_rolling)& (df['PERMNO'].isin(list_permno_on_reallocation_date))]

        #print(subdf['medate'].unique().tolist())

        if verbose:
            print(f'date: {data}, voters= {len(voters)}, wvoters={len(wvoters)}')
        mj_voters['date'].append(data)
        mj_voters['mj'].append(len(voters))
        mj_voters['wmj'].append(len(wvoters))


        subdf.loc[:,[f'r_{v}' for v in voters]]=compute_categorical_votes_qcut(subdf,voters,signs,num_cat).values
        subdf.loc[:,[f'wr_{v}' for v in wvoters]]=compute_categorical_votes_qcut(subdf,wvoters,wsigns,num_cat).values
        
        median_votes=subdf[['PERMNO']+[f'r_{v}' for v in voters]].groupby('PERMNO').median().reset_index()
        median_wvotes=subdf[['PERMNO']+[f'wr_{v}' for v in wvoters]].groupby('PERMNO').median().reset_index()
        
        
        df.loc[mask_reallocation_data_and_PERMNO,[f'{v}_median' for v in voters]]=median_votes[[f'r_{v}' for v in voters]].values
        df.loc[mask_reallocation_data_and_PERMNO,[f'w{v}_median' for v in wvoters]]=median_wvotes[[f'wr_{v}' for v in wvoters]].values
    
        date_df=df.loc[mask_reallocation_data_and_PERMNO]
        
        M=copy.copy(date_df[[f'{v}_median' for v in voters]])
        disentagle_df=copy.copy(date_df[[f'{v}' for v in voters]])
        disentagle_df=invert_signs_in_df(disentagle_df,voters,signs)
        mjfactor=compute_ranking(M,
                                 treat_na_mj,
                                 method,
                                 disentagle_df=disentagle_df)
        df.loc[df.index.isin(M.index),'mj']=-mjfactor
        
        wM=copy.copy(date_df[[f'w{v}_median' for v in wvoters]])
        disentagle_df=copy.copy(date_df[[f'{v}' for v in wvoters]])
        disentagle_df=invert_signs_in_df(disentagle_df,wvoters,wsigns)
        mjfactor=compute_ranking(wM,
                                 treat_na_mj,
                                 method,
                                 disentagle_df=disentagle_df)
        df.loc[df.index.isin(wM.index),'wmj']=-mjfactor

        
    return df, mj_voters


def compute_factor_strategies(df:pd.DataFrame,
                                configuration:dict
                                ):
    
    factors=configuration['factors']
    K=configuration['K']
    lag=configuration['lag']
    num_port=configuration['num_port']
    n_jobs=configuration['n_jobs']
    remove_outliers=configuration['remove_outliers']
    inclusive=configuration['inclusive']


    if remove_outliers:
        outliers=configuration['outliers']
    else:
        outliers=[]
    
    
    df['medate'] = df['date'] + pd.offsets.MonthEnd(0)
    df['wRET']=df["RET_RF"] * df["me_lag"]
    
    ### compute reallocation date 
    data_reallocation_min=df['medate'].min()+pd.offsets.MonthEnd(lag)
    data_reallocation_max=df['medate'].max()
    data_reallocation=[data_reallocation_min]
    while data_reallocation[-1] + pd.offsets.MonthEnd(K)<data_reallocation_max:
        data_reallocation.append(data_reallocation[-1] + pd.offsets.MonthEnd(K))

    
    #remove rows having nans in the factors cells at reallocation dates
    nan_dict={}
    for f in factors:
        nan_dict[f]=~(df['medate'].isin(data_reallocation) & df[f].isna())
    ###


    portfolios={}
    weighted_portfolios={}
    portfolios_stock_reallocation={}
    

    ## compute single portfolio returns
    results_factors = Parallel(n_jobs=n_jobs)(
        delayed(portfolio_formation)(df.loc[nan_dict[f]], 
                                      factor, 
                                      K, 
                                      num_port,
                                      lag,
                                      reverse=False,
                                      remove_outliers=remove_outliers,
                                      outliers=outliers,
                                      inclusive=inclusive) for factor in factors)
    
    for i,res in enumerate(results_factors):
        portfolios[factors[i]], weighted_portfolios[factors[i]], portfolios_stock_reallocation[factors[i]] = res

    return portfolios, weighted_portfolios, portfolios_stock_reallocation


def compute_long_short_quantile_stretegy(df:pd.DataFrame,
                                         configuration:dict):
    
    long_portfolios, long_weighted_portfolios, _, long_portfolios_stock_reallocation = compute_MJ_portfolio_strategy(df,configuration)
    configuration['method']='r_'+configuration['method']

    short_portfolios, short_weighted_portfolios, _, short_portfolios_stock_reallocation = compute_MJ_portfolio_strategy(df,configuration)

    num_port=configuration['num_port']
    ls_portforlio=long_portfolios['mj'][[f'port{num_port}']]-short_portfolios['mj'][[f'port{num_port}']]
    ls_weighted_portfolio=long_weighted_portfolios['wmj'][[f'port{num_port}']]-short_weighted_portfolios['wmj'][[f'port{num_port}']]
    
    return long_portfolios, long_weighted_portfolios, short_portfolios, short_weighted_portfolios, ls_portforlio, ls_weighted_portfolio


#change investment period to mj_rolling in the mj computation
def compute_MJ_portfolio_strategy(df:pd.DataFrame,
                                    configuration:dict
                                    ):

    K=configuration['K']
    lag=configuration['lag']
    num_port=configuration['num_port']
    fix_signs=configuration['fix_signs']
    remove_outliers=configuration['remove_outliers']
    rolling_method=configuration['rolling_method']
    inclusive=configuration['inclusive']

    
    if remove_outliers:
        outliers=configuration['outliers']
    else:
        outliers=[]
    
    
    df['medate'] = df['date'] + pd.offsets.MonthEnd(0)
    df['wRET']=df["RET_RF"] * df["me_lag"]
    df['mj']=np.nan
    df['wmj']=np.nan
    
    

    ### compute reallocation date 
    data_reallocation_min=df['medate'].min()+pd.offsets.MonthEnd(lag)
    data_reallocation_max=df['medate'].max()
    data_reallocation=[data_reallocation_min]
    while data_reallocation[-1] + pd.offsets.MonthEnd(K)<data_reallocation_max:
        data_reallocation.append(data_reallocation[-1] + pd.offsets.MonthEnd(K))

    
    #remove rows having nans in the factors cells at reallocation dates
    #mask_factor_nans_at_reallocation_dates=~(df['medate'].isin(data_reallocation) & df[factors].isna().any(axis=1))
    
    
    ###
    portfolios={}
    weighted_portfolios={}
    portfolios_stock_reallocation={}
    

    if rolling_method=='rank':
        print('Using rank rolling\n')
        df,mj_voters= rank_rolling(df,
                                   data_reallocation,
                                   configuration)
    elif rolling_method=='vote':
        print('Using vote rolling\n')
        df,mj_voters= vote_rolling(df,
                                   data_reallocation,
                                   configuration)
    elif rolling_method=='profile':
        print('Using profile rolling\n')
        df,mj_voters=profile_rolling(df,
                                   data_reallocation,
                                   configuration)
    elif rolling_method=='profile_vote':
        print('Using profile rolling\n')
        df,mj_voters=profile_vote_rolling(df,
                                   data_reallocation,
                                   configuration)
    if fix_signs:
        lag_mj=lag
    else:
        lag_mj=lag+K
                    
    factor='mj'
    portfolios[factor], _ , \
    portfolios_stock_reallocation[factor] = portfolio_formation(df,
                                                      factor, 
                                                      K, 
                                                      num_port, 
                                                      lag_mj,
                                                      remove_outliers=remove_outliers,
                                                      outliers=outliers,
                                                      inclusive=inclusive)
    factor='wmj'
    _ , weighted_portfolios[factor], \
    portfolios_stock_reallocation[factor] = portfolio_formation(df,
                                                      factor, 
                                                      K, 
                                                      num_port, 
                                                      lag_mj,
                                                      remove_outliers=remove_outliers,
                                                      outliers=outliers,
                                                      inclusive=inclusive)
    
            
    for key in ['mj']:
        portfolios[key] = portfolios[key][portfolios[key].index >= portfolios['mj'].index[0]]
    for key in ['wmj']:
        weighted_portfolios[key] = weighted_portfolios[key][weighted_portfolios[key].index >= weighted_portfolios['wmj'].index[0]]

    
    return portfolios, weighted_portfolios, mj_voters, portfolios_stock_reallocation



#change investment period to mj_rolling in the mj computation
def compute_stratified_MJ_portfolio_strategy(df:pd.DataFrame,
                                    configuration:dict
                                    ):

    factors=configuration['factors']
    K=configuration['K']
    lag=configuration['lag']
    method=configuration['method']
    num_port=configuration['num_port']
    num_cat=configuration['num_cat']
    verbose=configuration['verbose']
    remove_outliers=configuration['remove_outliers']
    treat_na_mj=configuration['treat_na_mj']
    mj_window=configuration['mj_window']
    fix_signs=configuration['fix_signs']
    rolling_method=configuration['rolling_method']
    inclusive=configuration['inclusive']

    if remove_outliers==True:
        print('we do not remove outliers in stratified MJ strategy, use standard method.')
    
    
    
    voters=configuration['default_voters']
    signs=configuration['default_signs']
    wvoters=configuration['default_wvoters']
    wsigns=configuration['default_wsigns']
    
    
    if not configuration['cut']:
        print('you did not specified any market size class: defaulting to micro cap')
        cut='micro'
    else:
        cut = configuration['cut']
    remove_outliers=True
            
    
    
    df['medate'] = df['date'] + pd.offsets.MonthEnd(0)
    df['wRET']=df["RET_RF"] * df["me_lag"]
    df['mj']=np.nan
    df['wmj']=np.nan


    
    ### compute reallocation date 
    data_reallocation_min=df['medate'].min()+pd.offsets.MonthEnd(lag)
    data_reallocation_max=df['medate'].max()
    data_reallocation=[data_reallocation_min]
    while data_reallocation[-1] + pd.offsets.MonthEnd(K)<data_reallocation_max:
        data_reallocation.append(data_reallocation[-1] + pd.offsets.MonthEnd(K))

    
    #remove rows having nans in the factors cells at reallocation dates
    mask_factor_nans_at_reallocation_dates=~(df['medate'].isin(data_reallocation) & df[factors].isna().any(axis=1))

    ## remove outliers if requested
    if cut:
        '''
        if configuration['cut']=='nano':     # <50million
        elif configuration['cut']=='micro':  #between $50 million and $300 million
        elif configuration['cut']=='small':  #between $300 million and $2000 million
        elif configuration['cut']=='medium': #between $2000 million and $10000 million
        elif configuration['cut']=='large': #between $10 billion and $200 billion
        elif configuration['cut']=='mega':   # >$200 billion
        '''
        if cut=='micro':
            inf=None
            sup='20th'
        elif cut=='small':
            inf='20th'
            sup='50th'
        elif cut=='medium':
            inf='50th'
            sup='75th'
        elif cut=='large':
            inf='75th'
            sup=None

        outliers=[inf,sup]


    ###
    portfolios={}
    weighted_portfolios={}
    portfolios_stock_reallocation={}

    

    if rolling_method=='rank':
        print('Using rank rolling\n')
        df,mj_voters= rank_rolling(df,
                                    data_reallocation,
                                    voters,
                                    wvoters,
                                    signs,
                                    wsigns,
                                    mj_window,
                                    method,
                                    num_cat,
                                    treat_na_mj,
                                    verbose)
    elif rolling_method=='vote':
        print('Using vote rolling\n')
        df,mj_voters= vote_rolling(df,
                                    factors,
                                    data_reallocation,
                                    voters,
                                    wvoters,
                                    signs,
                                    wsigns,
                                    mj_window,
                                    method,
                                    num_cat,
                                    treat_na_mj,
                                    verbose)
    elif rolling_method=='profile':
        print('Using profile rolling\n')
        df,mj_voters=profile_rolling(df,
                                    factors,
                                    data_reallocation,
                                    voters,
                                    wvoters,
                                    signs,
                                    wsigns,
                                    mj_window,
                                    method,
                                    num_cat,
                                    treat_na_mj,
                                    verbose)
    if fix_signs:
        lag_mj=lag
    else:
        lag_mj=lag+K
                    
    factor='mj'
    portfolios[factor], _ , \
    portfolios_stock_reallocation[factor] = portfolio_formation(df.loc[mask_factor_nans_at_reallocation_dates],
                                                      factor, 
                                                      K, 
                                                      num_port, 
                                                      lag_mj,
                                                      remove_outliers=remove_outliers,
                                                      outliers=outliers,
                                                      inclusive=inclusive)
    factor='wmj'
    _ , weighted_portfolios[factor], \
    portfolios_stock_reallocation[factor] = portfolio_formation(df.loc[mask_factor_nans_at_reallocation_dates],
                                                      factor, 
                                                      K, 
                                                      num_port, 
                                                      lag_mj,
                                                      remove_outliers=remove_outliers,
                                                      outliers=outliers,
                                                      inclusive=inclusive)
    
    
    
    return portfolios, weighted_portfolios, mj_voters, portfolios_stock_reallocation


def drop_correlated_voters(df_corr_window:pd.DataFrame,
                          voters:list,
                          signs:list,
                          corr_voters=[],
                          drop_corr_thresh=0.75):
    import networkx as nx

    def flatten(xss):
        return [x for xs in xss for x in xs]
    
    if corr_voters: 
        df_voters=pd.DataFrame(np.array([corr_voters,voters,signs]).T,columns=['score','factor','sign'])
        df_voters['sign']=df_voters['sign'].astype('float')
    else: 
        df_voters=pd.DataFrame(np.array([voters,signs]).T,columns=['factor','sign'])
        df_voters['sign']=df_voters['sign'].astype('float')

    corr_df=df_corr_window[voters].corr(method='spearman').abs()
    mask_keep = np.triu(np.ones(corr_df.shape), k=1).astype(bool).reshape(corr_df.size)
    # melt (unpivot) the dataframe and apply mask
    sr = corr_df.stack()[mask_keep]
    # filter and get names
    edges = sr[sr > drop_corr_thresh].reset_index().values[:, :2]

    g = nx.from_edgelist(edges)
    ls_cliques = []
    for clique in nx.algorithms.find_cliques(g):
        ls_cliques.append(clique)
    

    if corr_voters: 
        voting_cliques=[]
        for clique in ls_cliques:
            clique_ls=[]
            for el in clique:
                if el in voters:
                    clique_ls.append(el)
            voting_cliques.append(clique_ls)
        
        to_keep=[]
        for clique in voting_cliques:
            if clique:
                temp_df=df_voters.loc[df_voters['factor'].isin(clique)]
                to_keep.append(temp_df.loc[temp_df['score']==temp_df['score'].max()]['factor'].tolist())

        to_remove=list(set(flatten(voting_cliques))-set(flatten(to_keep)))

    else:
        to_keep=[np.random.choice(clique_voters) for clique_voters in ls_cliques]
        to_remove=list(set(flatten(ls_cliques))-set(to_keep))

    df_voters=df_voters.loc[df_voters['factor'].isin(list(set(voters)-set(to_remove)))]
    voters=df_voters['factor'].tolist()
    signs=df_voters['sign'].tolist()

    if corr_voters:
        corr_voters=df_voters['score'].tolist()
    


    return voters, signs, corr_voters


def single_GS_point(df,configuration,point):
    
    K, lag, rolling_method, mj_window, remove_outliers = point
        
    if remove_outliers:
        date_initial=configuration['date_initial']
        date_final=configuration['date_final']
        dataFF = pd.read_csv("./new_dataset_creation/original_data/ME_Breakpoints.csv") # data are in millions
        dataFF['date'] = pd.to_datetime(dataFF['date'],format='%Y%m')+ MonthEnd(0)
        ### consider the same lag as in the market equity values
        dataFF['date'] = dataFF["date"] + pd.offsets.MonthEnd(1)

        dataFF = dataFF[(dataFF.date >= date_initial) & (dataFF.date <= date_final)].reset_index(drop = True)
        inf = '5th'
        sup =  None 

        if not sup:        
            suffix = '_'+inf #suffix to add to the plot suffix  
            df = pd.merge(df, dataFF[['date',inf]], on='date',how='left')
        else:
            suffix = '_'+inf+sup #suffix to add to the plot suffix   
            df = pd.merge(df, dataFF[['date',inf,sup]], on='date',how='left')
    
    
    
    configuration['K']=K 
    configuration['lag']=lag 
    configuration['mj_window']=mj_window 
    configuration['remove_outliers']=remove_outliers
    configuration['rolling_method']=rolling_method
    if configuration['remove_outliers']:
        configuration['outliers']=[inf,sup]
    
    
    num_port = configuration['num_port']
    factors = configuration['factors']
    
    portfolios, \
    weighted_portfolios, \
    mj_voters, \
    portfolios_stock_reallocation = compute_MJ_portfolio_strategy(df=df, configuration=configuration)

    
    sharpe_ratios={}
    sharpe_ratios['even']=[]
    sharpe_ratios['weighted']=[]

    volatilities={}
    volatilities['even']=[]
    volatilities['weighted']=[]

    returns={}
    returns['even']=[]
    returns['weighted']=[]


    for factor in ['mj']:
        sharpe_ratio_factor=np.sqrt(12)*(portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']]).mean()\
                                        /portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].std()
        volatility_factor=np.sqrt(12)*(portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].std())
        return_factor=12*(portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].mean())
        for i in range(len(sharpe_ratio_factor.index)):
            if i!=len(sharpe_ratio_factor.index)-1: 
                sharpe_ratios['even'].append([factor,i+1,sharpe_ratio_factor.iloc[i]])
                volatilities['even'].append([factor,i+1,volatility_factor.iloc[i]])
                returns['even'].append([factor,i+1,return_factor.iloc[i]])
            else:
                sharpe_ratios['even'].append([factor,'long_short',sharpe_ratio_factor.iloc[i]])
                volatilities['even'].append([factor,'long_short',volatility_factor.iloc[i]])
                returns['even'].append([factor,'long_short',return_factor.iloc[i]])

    sharpe_ratios['even']=pd.DataFrame(sharpe_ratios['even'])
    sharpe_ratios['even'].columns=['factor','portfolio','sharpe ratio']

    volatilities['even']=pd.DataFrame(volatilities['even'])
    volatilities['even'].columns=['factor','portfolio','volatility']

    returns['even']=pd.DataFrame(returns['even'])
    returns['even'].columns=['factor','portfolio','returns']
    
    for factor in ['wmj']:
        sharpe_ratio_factor=np.sqrt(12)*(weighted_portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']]).mean()\
                                        /weighted_portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].std()
        volatility_factor=np.sqrt(12)*(weighted_portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].std())
        return_factor=12*(weighted_portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].mean())
        for i in range(len(sharpe_ratio_factor.index)):
            if i!=len(sharpe_ratio_factor.index)-1: 
                sharpe_ratios['weighted'].append([factor,i+1,sharpe_ratio_factor.iloc[i]])
                volatilities['weighted'].append([factor,i+1,volatility_factor.iloc[i]])
                returns['weighted'].append([factor,i+1,return_factor.iloc[i]])
            else:
                sharpe_ratios['weighted'].append([factor,'long_short',sharpe_ratio_factor.iloc[i]])
                volatilities['weighted'].append([factor,'long_short',volatility_factor.iloc[i]])
                returns['weighted'].append([factor,'long_short',return_factor.iloc[i]])

    sharpe_ratios['weighted']=pd.DataFrame(sharpe_ratios['weighted'])
    sharpe_ratios['weighted'].columns=['factor','portfolio','sharpe ratio']


    volatilities['weighted']=pd.DataFrame(volatilities['weighted'])
    volatilities['weighted'].columns=['factor','portfolio','volatility']

    returns['weighted']=pd.DataFrame(returns['weighted'])
    returns['weighted'].columns=['factor','portfolio','returns']

    return tuple(np.array(point)), portfolios_stock_reallocation, returns, volatilities, sharpe_ratios


def single_GS_longshort_point(df,configuration,point):
    
    K, lag, rolling_method, mj_window, remove_outliers = point
        
    if remove_outliers:
        date_initial=configuration['date_initial']
        date_final=configuration['date_final']
        dataFF = pd.read_csv("./new_dataset_creation/original_data/ME_Breakpoints.csv") # data are in millions
        dataFF['date'] = pd.to_datetime(dataFF['date'],format='%Y%m')+ MonthEnd(0)
        ### consider the same lag as in the market equity values
        dataFF['date'] = dataFF["date"] + pd.offsets.MonthEnd(1)

        dataFF = dataFF[(dataFF.date >= date_initial) & (dataFF.date <= date_final)].reset_index(drop = True)
        inf = '5th'
        sup =  None 

        if not sup:        
            suffix = '_'+inf #suffix to add to the plot suffix  
            df = pd.merge(df, dataFF[['date',inf]], on='date',how='left')
        else:
            suffix = '_'+inf+sup #suffix to add to the plot suffix   
            df = pd.merge(df, dataFF[['date',inf,sup]], on='date',how='left')
    
    
    
    configuration['K']=K 
    configuration['lag']=lag 
    configuration['mj_window']=mj_window 
    configuration['remove_outliers']=remove_outliers
    configuration['rolling_method']=rolling_method
    if configuration['remove_outliers']:
        configuration['outliers']=[inf,sup]
    
    
    num_port = configuration['num_port']
    factors = configuration['factors']

    long_portfolios,  long_weighted_portfolios,  \
    short_portfolios, short_weighted_portfolios, \
    ls_portfolios, ls_weighted_portfolios = compute_long_short_quantile_stretegy(df,configuration)

    portfolios=pd.concat([long_portfolios['mj']['port10'],
            short_portfolios['mj']['port10'],
            ls_portfolios
            ],axis=1)

    weighted_portfolios=pd.concat([
            long_weighted_portfolios['wmj']['port10'],
            short_weighted_portfolios['wmj']['port10'],
            ls_weighted_portfolios
            ],axis=1)

    portfolios.columns=['long','short','long_short']
    weighted_portfolios.columns=['long','short','long_short']

    
    
    return tuple(np.array(point)), portfolios, weighted_portfolios
    
   
def single_GS_point_complete(df,configuration,point):
    
    K, lag, rolling_method, mj_window, remove_outliers = point
        
    if remove_outliers:
        date_initial=configuration['date_initial']
        date_final=configuration['date_final']
        dataFF = pd.read_csv("./dataset_creation/original_data/ME_Breakpoints.csv") # data are in millions
        dataFF['date'] = pd.to_datetime(dataFF['date'],format='%Y%m')+ MonthEnd(0)
        ### consider the same lag as in the market equity values
        dataFF['date'] = dataFF["date"] + pd.offsets.MonthEnd(1)

        dataFF = dataFF[(dataFF.date >= date_initial) & (dataFF.date <= date_final)].reset_index(drop = True)
        inf = '5th'
        sup =  None 

        if not sup:        
            suffix = '_'+inf #suffix to add to the plot suffix  
            df = pd.merge(df, dataFF[['date',inf]], on='date',how='left')
        else:
            suffix = '_'+inf+sup #suffix to add to the plot suffix   
            df = pd.merge(df, dataFF[['date',inf,sup]], on='date',how='left')
    
    
    
    configuration['K']=K 
    configuration['lag']=lag 
    configuration['mj_window']=mj_window 
    configuration['remove_outliers']=remove_outliers
    configuration['rolling_method']=rolling_method
    if configuration['remove_outliers']:
        configuration['outliers']=[inf,sup]
    
    
    num_port = configuration['num_port']
    factors = configuration['factors']
    
    portfolios, \
    weighted_portfolios, \
    mj_voters, \
    portfolios_stock_reallocation = compute_MJ_portfolio_strategy(df=df, configuration=configuration)

    
    sharpe_ratios={}
    sharpe_ratios['even']=[]
    sharpe_ratios['weighted']=[]

    volatilities={}
    volatilities['even']=[]
    volatilities['weighted']=[]

    returns={}
    returns['even']=[]
    returns['weighted']=[]


    for factor in ['mj']:
        sharpe_ratio_factor=np.sqrt(12)*(portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']]).mean()\
                                        /portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].std()
        volatility_factor=np.sqrt(12)*(portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].std())
        return_factor=12*(portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].mean())
        for i in range(len(sharpe_ratio_factor.index)):
            if i!=len(sharpe_ratio_factor.index)-1: 
                sharpe_ratios['even'].append([factor,i+1,sharpe_ratio_factor.iloc[i]])
                volatilities['even'].append([factor,i+1,volatility_factor.iloc[i]])
                returns['even'].append([factor,i+1,return_factor.iloc[i]])
            else:
                sharpe_ratios['even'].append([factor,'long_short',sharpe_ratio_factor.iloc[i]])
                volatilities['even'].append([factor,'long_short',volatility_factor.iloc[i]])
                returns['even'].append([factor,'long_short',return_factor.iloc[i]])

    sharpe_ratios['even']=pd.DataFrame(sharpe_ratios['even'])
    sharpe_ratios['even'].columns=['factor','portfolio','sharpe ratio']

    volatilities['even']=pd.DataFrame(volatilities['even'])
    volatilities['even'].columns=['factor','portfolio','volatility']

    returns['even']=pd.DataFrame(returns['even'])
    returns['even'].columns=['factor','portfolio','returns']
    
    for factor in ['wmj']:
        sharpe_ratio_factor=np.sqrt(12)*(weighted_portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']]).mean()\
                                        /weighted_portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].std()
        volatility_factor=np.sqrt(12)*(weighted_portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].std())
        return_factor=12*(weighted_portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].mean())
        for i in range(len(sharpe_ratio_factor.index)):
            if i!=len(sharpe_ratio_factor.index)-1: 
                sharpe_ratios['weighted'].append([factor,i+1,sharpe_ratio_factor.iloc[i]])
                volatilities['weighted'].append([factor,i+1,volatility_factor.iloc[i]])
                returns['weighted'].append([factor,i+1,return_factor.iloc[i]])
            else:
                sharpe_ratios['weighted'].append([factor,'long_short',sharpe_ratio_factor.iloc[i]])
                volatilities['weighted'].append([factor,'long_short',volatility_factor.iloc[i]])
                returns['weighted'].append([factor,'long_short',return_factor.iloc[i]])

    sharpe_ratios['weighted']=pd.DataFrame(sharpe_ratios['weighted'])
    sharpe_ratios['weighted'].columns=['factor','portfolio','sharpe ratio']


    volatilities['weighted']=pd.DataFrame(volatilities['weighted'])
    volatilities['weighted'].columns=['factor','portfolio','volatility']

    returns['weighted']=pd.DataFrame(returns['weighted'])
    returns['weighted'].columns=['factor','portfolio','returns']

    return tuple(np.array(point)), portfolios_stock_reallocation, returns, volatilities, sharpe_ratios, portfolios, weighted_portfolios
  

def single_GS_point_robustness(df,configuration_original,point):
    
    configuration=configuration_original.copy()

    configuration['remove_outliers']=point[0]
    voters_index=list(point[1:])
        
    if configuration['remove_outliers']:
        date_initial=configuration['date_initial']
        date_final=configuration['date_final']
        dataFF = pd.read_csv("./new_dataset_creation/original_data/ME_Breakpoints.csv") # data are in millions
        dataFF['date'] = pd.to_datetime(dataFF['date'],format='%Y%m')+ MonthEnd(0)
        ### consider the same lag as in the market equity values
        dataFF['date'] = dataFF["date"] + pd.offsets.MonthEnd(1)

        dataFF = dataFF[(dataFF.date >= date_initial) & (dataFF.date <= date_final)].reset_index(drop = True)
        inf = '5th'
        sup =  None 

        if not sup:        
            df = pd.merge(df, dataFF[['date',inf]], on='date',how='left')
        else: 
            df = pd.merge(df, dataFF[['date',inf,sup]], on='date',how='left')
    
    if configuration['remove_outliers']:
        configuration['outliers']=[inf,sup]
    
    num_port = configuration['num_port']

    for default_quant in ['default_voters',
                      'default_signs',
                      'default_scores',
                      'default_wvoters',
                      'default_wsigns',
                      'default_wscores']:
    
        configuration[default_quant]=[configuration_original[default_quant][i] for i in voters_index]

    
    portfolios, \
    weighted_portfolios, \
    _, portfolios_stock_reallocation = compute_MJ_portfolio_strategy(df=df, configuration=configuration)

    
    sharpe_ratios={}
    sharpe_ratios['even']=[]
    sharpe_ratios['weighted']=[]

    volatilities={}
    volatilities['even']=[]
    volatilities['weighted']=[]

    returns={}
    returns['even']=[]
    returns['weighted']=[]


    for factor in ['mj']:
        sharpe_ratio_factor=np.sqrt(12)*(portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']]).mean()\
                                        /portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].std()
        volatility_factor=np.sqrt(12)*(portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].std())
        return_factor=12*(portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].mean())
        for i in range(len(sharpe_ratio_factor.index)):
            if i!=len(sharpe_ratio_factor.index)-1: 
                sharpe_ratios['even'].append([factor,i+1,sharpe_ratio_factor.iloc[i]])
                volatilities['even'].append([factor,i+1,volatility_factor.iloc[i]])
                returns['even'].append([factor,i+1,return_factor.iloc[i]])
            else:
                sharpe_ratios['even'].append([factor,'long_short',sharpe_ratio_factor.iloc[i]])
                volatilities['even'].append([factor,'long_short',volatility_factor.iloc[i]])
                returns['even'].append([factor,'long_short',return_factor.iloc[i]])

    sharpe_ratios['even']=pd.DataFrame(sharpe_ratios['even'])
    sharpe_ratios['even'].columns=['factor','portfolio','sharpe ratio']

    volatilities['even']=pd.DataFrame(volatilities['even'])
    volatilities['even'].columns=['factor','portfolio','volatility']

    returns['even']=pd.DataFrame(returns['even'])
    returns['even'].columns=['factor','portfolio','returns']
    
    for factor in ['wmj']:
        sharpe_ratio_factor=np.sqrt(12)*(weighted_portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']]).mean()\
                                        /weighted_portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].std()
        volatility_factor=np.sqrt(12)*(weighted_portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].std())
        return_factor=12*(weighted_portfolios[factor][[f'port{i}' for i in range(1,num_port+1)]+['long_short']].mean())
        for i in range(len(sharpe_ratio_factor.index)):
            if i!=len(sharpe_ratio_factor.index)-1: 
                sharpe_ratios['weighted'].append([factor,i+1,sharpe_ratio_factor.iloc[i]])
                volatilities['weighted'].append([factor,i+1,volatility_factor.iloc[i]])
                returns['weighted'].append([factor,i+1,return_factor.iloc[i]])
            else:
                sharpe_ratios['weighted'].append([factor,'long_short',sharpe_ratio_factor.iloc[i]])
                volatilities['weighted'].append([factor,'long_short',volatility_factor.iloc[i]])
                returns['weighted'].append([factor,'long_short',return_factor.iloc[i]])

    sharpe_ratios['weighted']=pd.DataFrame(sharpe_ratios['weighted'])
    sharpe_ratios['weighted'].columns=['factor','portfolio','sharpe ratio']


    volatilities['weighted']=pd.DataFrame(volatilities['weighted'])
    volatilities['weighted'].columns=['factor','portfolio','volatility']

    returns['weighted']=pd.DataFrame(returns['weighted'])
    returns['weighted'].columns=['factor','portfolio','returns']

    return tuple(np.array(point)), portfolios_stock_reallocation, returns, volatilities, sharpe_ratios
  

def compute_performance_matrix(crsp_m):

    # Stock performance evaluations
    # Mean
    av_ret_rf = crsp_m.groupby("PERMNO")["RET_RF"].mean() 
    vol_ret_rf= crsp_m.groupby("PERMNO")["RET_RF"].std()

    #av_ret = crsp_m.groupby("PERMNO")["RET"].mean() 
    av_mktret = crsp_m.groupby("PERMNO")["Mkt_RF"].mean() 
    av_me = crsp_m.groupby("PERMNO")["me"].mean() 

    # Volatility
    #vol_ret = crsp_m.groupby("PERMNO")["RET"].std() 

    # Sharpe ratio
    sharpe_ratio = (av_ret_rf / vol_ret_rf)

    # Treynor ratio
    variance = crsp_m.groupby("PERMNO")["Mkt_RF"].var() 
    covariance = crsp_m.groupby('PERMNO').apply(lambda x: x['RET_RF'].cov(x['Mkt_RF']))
    beta = covariance/variance
    Treynor_Ratio = ((av_ret_rf)/beta)


    # Information ratio
    crsp_m['ret_mkt'] = crsp_m['RET_RF']-crsp_m['Mkt_RF']
    avg_ret_mkt = crsp_m.groupby("PERMNO")["ret_mkt"].mean() 
    std_ret_mkt = crsp_m.groupby("PERMNO")["ret_mkt"].std()  
    information_ratio = (avg_ret_mkt / std_ret_mkt) 

    # Jansen's alpha
    alpha = av_ret_rf - beta * av_mktret 

    # Performance dataframe - we annualized the values
    performance = pd.DataFrame({'Av_ret': av_ret_rf*12, 'Vol_ret': vol_ret_rf*np.sqrt(12), 'Sharpe_ratio':sharpe_ratio*np.sqrt(12), 
                                'Treynor_ratio': Treynor_Ratio*(12),'Information_ratio': information_ratio*np.sqrt(12),
                                'alpha': alpha*12, 'Market_cap': av_me}).reset_index()
    return performance


def define_grades_based_on_NYSE(performance,
                                performances_list,
                                performance_signs,
                                NYSE_ID,
                                id25,
                                df_names,
                                percentiles=[0, 20, 40 ,60, 80, 100],
                                verbose=False,
                                train=True):
    # Define prformance metrics with categorical values
    import string
    performance25 = performance[(performance['PERMNO'].isin(id25))]
    performance25 = pd.merge(df_names.set_index('PERMNO'), performance25.set_index('PERMNO'), left_index=True, right_index=True).reset_index()
    if train:
        performance25 = performance25.drop_duplicates(subset=['PERMNO'], keep='last')
    else:
        performance25 = performance25.drop_duplicates(subset=['PERMNO'], keep='first')
    performance25 = performance25.reset_index(drop=True)

    NYSE_performance=performance.loc[performance['PERMNO'].isin(NYSE_ID)]
    #NYSE_performance=NYSE_performance[(np.abs(stats.zscore(NYSE_performance)) < 3).all(axis=1)]

    valuation = {}
    valuation_grades ={}

    for perf in performances_list:
        print(perf)
        bins = np.nanpercentile(NYSE_performance[perf], percentiles , axis=0)
        C=len(percentiles)-1 #number of categories
        bins[0]=bins[0]-(0.00001) ##MIND THE LINE: REMOVING OUTLIERS MAY CAUSE NAN RESULT
        bins[-1]=bins[-1]+(0.0001) ##MIND THE LINE: REMOVING OUTLIERS MAY CAUSE NAN RESULT
        if performance_signs[perf]<0:
            cat_values = (len(percentiles)-2)-pd.cut(performance25[perf], bins, labels=False, retbins=False, right=False)
        else:
            cat_values = pd.cut(performance25[perf], bins, labels=False, retbins=False, right=False)
        valuation[perf]=cat_values.to_list()
        cperc_grades=cat_values.map(dict([(C-1-i,string.ascii_uppercase[i]) for i in range(C)]))
        
        valuation_grades[perf]=cperc_grades.to_list()
        
        if verbose:
            ## PLOT HISTOGRAM
            performance25[perf].hist()
            for b in bins:
                plt.vlines(b, 0, 4, colors='red')
            plt.title(perf)
            plt.show()
        
    valuation_grades=pd.DataFrame(valuation_grades)
    valuation_grades.index=performance25['TICKER'].to_list()
    valuation=pd.DataFrame(valuation)
    valuation.index=performance25['TICKER'].to_list()
    return valuation, valuation_grades, performance25


def best_portfolio_performance(list_port,crsp_m_prediction,performances):
    sharpe_rf=(pd.concat(list_port,axis=1).mean()/pd.concat(list_port,axis=1).std())
    ret_rf=pd.concat(list_port,axis=1).mean()
    std_rf=pd.concat(list_port,axis=1).std()

    best_and_mk=pd.concat(list_port,
                        axis=1).merge(crsp_m_prediction[['date',
                                                        'Mkt_RF']].drop_duplicates('date').set_index('date'),
                                                        left_index=True,
                                                        right_index=True)

    ret_mk=best_and_mk[performances+['mj']].sub(best_and_mk['Mkt_RF'], axis=0)

    variance = best_and_mk['Mkt_RF'].var()

    covariance={}
    for perf in performances+['mj']:
        covariance[perf]=best_and_mk[perf].cov(best_and_mk['Mkt_RF'])

    covariance=pd.Series(covariance,name='cov')
    beta = covariance/variance
    Treynor_Ratio = ret_rf.div(beta)
    information_ratio = ret_mk.mean().div(ret_mk.std())
    alpha = ret_rf-beta*best_and_mk['Mkt_RF'].mean()

    performance_pred = pd.DataFrame({'Av_ret': ret_rf*12, 'Vol_ret': std_rf*np.sqrt(12), 'Sharpe_ratio':sharpe_rf*np.sqrt(12), 
                                    'Treynor_ratio': Treynor_Ratio*(12),'Information_ratio': information_ratio*np.sqrt(12),
                                    'alpha': alpha*12}).reset_index(drop=True)
    
    performance_pred.index=[f'best portfolio {perf}' for perf in performances+['mj']]
    return performance_pred


def compute_df_rankings(placements,df):
    perf_col={'mj':[],
            'TICKER':[]}
    for index,place in placements:
        perf_col['mj'].append(place)
        perf_col['TICKER'].append(df.TICKER[index])

    results=pd.DataFrame(perf_col).sort_values('mj')
    results['tiles']=((results['mj'] - results['mj'].astype(int))*100)+1
    results['mj']=results['mj'].astype(int)
    results = results[['mj','tiles','TICKER']]
    return results


def portfolio_rets_and_plots(aggr_train_test_performance,crsp_m_prediction,perf_metric):

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    if perf_metric=='mj' or perf_metric=='Vol_ret':
        ascending=False
    else:
        ascending=True

    aggr_train_test_performance=aggr_train_test_performance.sort_values(by=f'{perf_metric}_train',
                                                                        ascending=ascending).reset_index(drop=True)
    port_1=aggr_train_test_performance.iloc[:15]['PERMNO'].to_list()
    port_2=aggr_train_test_performance.iloc[15:30]['PERMNO'].to_list()
    port_3=aggr_train_test_performance.iloc[20:30]['PERMNO'].to_list()

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    for i, port in enumerate([port_1,port_2,port_3]):
        mean_return_by_date=crsp_m_prediction.loc[crsp_m_prediction['PERMNO'].isin(port)][['date','RET_RF']].groupby('date').mean()
        plt.plot(mean_return_by_date.index, ((1+mean_return_by_date).cumprod() - 1).values,label=f'port {i+1}')

    plt.xticks(rotation=45)
    # Display only month and year values
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.legend()
    plt.title(perf_metric)
    plt.show()

    mean_return_by_date.rename(columns={'RET_RF':perf},inplace=True)

    return mean_return_by_date


#change investment period to mj_rolling in the mj computation
def compute_portfolio_strategies(df:pd.DataFrame,
                                 configuration:dict
                                 ):

    portfolios, weighted_portfolios, portfolios_stock_reallocation = compute_factor_strategies(df,configuration)

    conf_basic = configuration.copy()
    conf_basic['K']=1

    basic_portfolios, basic_weighted_portfolios, _ = compute_factor_strategies(df,conf_basic)

    configuration['basic_portfolios']=basic_portfolios
    configuration['basic_weighted_portfolios']=basic_weighted_portfolios

    MJ_portfolios, MJ_weighted_portfolios, \
        mj_voters, MJ_portfolios_stock_reallocation = compute_MJ_portfolio_strategy(df,
                                                                                configuration)
    
    portfolios.update(MJ_portfolios)
    weighted_portfolios.update(MJ_weighted_portfolios)
    portfolios_stock_reallocation.update(MJ_portfolios_stock_reallocation)

    return portfolios, weighted_portfolios, \
              mj_voters, portfolios_stock_reallocation

    


    




