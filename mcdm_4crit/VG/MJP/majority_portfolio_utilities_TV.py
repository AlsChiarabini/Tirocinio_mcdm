
import itertools
from MJP.ordering_MJ_lex_dlex import *
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon
import statsmodels.api as sm
import copy
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from joblib import Parallel, delayed
pd.set_option('mode.chained_assignment', None)
# Suppress RuntimeWarnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from pymcdm.methods import TOPSIS, VIKOR, PROMETHEE_II
from pymcdm.weights import equal_weights

import ahpy

def compute_mcdm_rolling_strategy(df: pd.DataFrame, method_name: str, configuration: dict, weighting: bool = False):
    df = df.copy()
    df['medate'] = df['date'] + MonthEnd(0)

    K = configuration['K']
    lag = configuration['lag']
    num_port = configuration['num_port']

    # Importante: ordina e riassegna i fattori, anche i segni devono essere allineati
    factors = sorted(configuration['factors'])
    types = {}
    if 'default_signs' in configuration:
        signs = configuration['default_signs']
        # Assumiamo dict con segni per fattore
        types = {k: signs[i] for i,k in enumerate(configuration['factors'])}
        # Ricomponi types ordinato come factors
        types = [types[f] for f in factors]
    else:
        # fallback segni uguali
        types = [1]*len(factors)

    verbose = configuration.get('verbose', False)

    method_map = {
        'topsis': TOPSIS(),
        'vikor': VIKOR(v=0.5),
        'promethee': PROMETHEE_II(preference_function='usual'),
        'ahp': None
    }

    # Creo dizionario pairwise con valori 1 per tutte le coppie completate e ordinate
    factors = sorted(factors)
    criteria_pairwise = {}
    for i in range(len(factors)):
        for j in range(i+1, len(factors)):
            criteria_pairwise[(factors[i], factors[j])] = 1

    # Se AHP, calcolo pesi con ahpy e preparo vettore pesi
    if method_name == 'ahp':
        try:
            criteria_compare = ahpy.Compare('Criteria', criteria_pairwise, precision=3)
            weights_dict = criteria_compare.target_weights
            weights_vector = np.array([weights_dict[f] for f in factors])
            ascending = False
            if verbose:
                print("AHP Pesi criteri:", weights_dict)
                print("AHP Consistency Ratio:", criteria_compare.consistency_ratio)
        except AssertionError as e:
            raise ValueError(f"E' avvenuto un errore nella costruzione AHP: {e}")
    else:
        assert method_name in method_map, f"Metodo {method_name} non supportato"
        model = method_map[method_name]
        ascending = True if method_name == 'vikor' else False

    reallocation_dates = []
    date_start = df['medate'].min() + MonthEnd(lag)
    date_end = df['medate'].max()
    while date_start + MonthEnd(K) <= date_end:
        reallocation_dates.append(date_start)
        date_start += MonthEnd(K)

    returns_dict = {f'port{i+1}': [] for i in range(num_port)}
    returns_dict['long_short'] = []

    turnover_values = {f'port{i+1}': [] for i in range(num_port)}
    reallocation = {f'port{i+1}': {} for i in range(num_port)}
    prev_portfolios = {f'port{i+1}': None for i in range(num_port)}

    for date in reallocation_dates:
        df_now = df[df['medate'] == date].dropna(subset=factors).copy()
        if df_now.empty or len(df_now) < num_port:
            continue

        matrix = df_now[factors].values

        # Controllo disallineamento dimensioni
        if method_name == 'ahp':
            assert matrix.shape[1] == len(weights_vector), "Dimensione pesi AHP differente dalla matrice fattori"
            scores = matrix.dot(weights_vector)
        else:
            # Equal weights con pymcdm come fallback
            weights = equal_weights(matrix)
            scores = model(matrix, weights, types)

        df_now['score'] = scores
        df_now = df_now.sort_values('score', ascending=ascending).reset_index(drop=True)

        labels = [f'port{i+1}' for i in reversed(range(num_port))]
        df_now['portfolio'] = pd.qcut(df_now.index, q=num_port, labels=labels)

        period_end = date + MonthEnd(K)
        df_hold = df[(df['medate'] > date) & (df['medate'] <= period_end)]

        port_permnos = {}
        for port in labels:
            tickers = df_now[df_now['portfolio'] == port]['PERMNO'].tolist()
            port_permnos[port] = tickers
            reallocation[port][date] = tickers
            df_port = df_hold[df_hold['PERMNO'].isin(tickers)].copy()

            if weighting:
                df_port['wret'] = df_port['RET_RF'] * df_port['me_lag']
                ret = df_port.groupby('medate').apply(lambda x: x['wret'].sum() / x['me_lag'].sum())
            else:
                ret = df_port.groupby('medate')['RET_RF'].mean()

            returns_dict[port].append(ret)

        for port in labels:
            curr_holdings = set(port_permnos[port])
            if prev_portfolios[port] is not None:
                prev_holdings = set(prev_portfolios[port])
                intersection = len(prev_holdings & curr_holdings)
                total_unique = max(len(prev_holdings), len(curr_holdings))
                turnover_rate = 1 - (intersection / total_unique) if total_unique > 0 else 0.0
            else:
                turnover_rate = np.nan
            turnover_values[port].append(turnover_rate)
            prev_portfolios[port] = port_permnos[port]

        long = df_hold[df_hold['PERMNO'].isin(port_permnos['port10'])].copy()
        short = df_hold[df_hold['PERMNO'].isin(port_permnos['port1'])].copy()

        if weighting:
            long['wret'] = long['RET_RF'] * long['me_lag']
            short['wret'] = short['RET_RF'] * short['me_lag']
            ret_long = long.groupby('medate').apply(lambda x: x['wret'].sum() / x['me_lag'].sum())
            ret_short = short.groupby('medate').apply(lambda x: x['wret'].sum() / x['me_lag'].sum())
        else:
            ret_long = long.groupby('medate')['RET_RF'].mean()
            ret_short = short.groupby('medate')['RET_RF'].mean()

        ls_ret = ret_long - ret_short
        returns_dict['long_short'].append(ls_ret)

    for key in returns_dict:
        returns_dict[key] = pd.concat(returns_dict[key]).sort_index()

    portfolios_df = pd.DataFrame(returns_dict).dropna()

    avg_turnover = {
        port: np.nanmean(values) if values else np.nan
        for port, values in turnover_values.items()
    }

    turnover_key = 'VW_turnover' if weighting else 'EW_turnover'
    turnover_df = pd.DataFrame({
        'portfolio': [int(p.replace('port', '')) for p in avg_turnover.keys()],
        turnover_key: list(avg_turnover.values())
    })

    portfolios_stock_reallocation = {
        turnover_key: turnover_df,
        'reallocation': reallocation
    }

    return portfolios_df, portfolios_stock_reallocation

def create_df(filename,factors,date_initial,date_final,remove_outliers=False,inf=None,sup=None):
    default_cols=['PERMNO', 'date', 'RET',  'me', 'RF', 'Mkt_RF', 'RET_RF', 'me_lag']
    df=pd.read_csv(filename,usecols=default_cols+factors)
    df['date'] = pd.to_datetime(df['date'])

    df=df.loc[(df.date >= date_initial) & (df.date <= date_final)]
    if remove_outliers:
        dataFF = pd.read_csv("./dataset_creation/original_data/ME_Breakpoints.csv") # data are in millions
        dataFF['date'] = pd.to_datetime(dataFF['date'],format='%Y%m')+ pd.offsets.MonthEnd(0)
        ### consider the same lag as in the market equity values
        dataFF['date'] = dataFF["date"] + pd.offsets.MonthEnd(1)
        
        dataFF = dataFF[(dataFF.date >= date_initial) & (dataFF.date <= date_final)].reset_index(drop = True)
        
        if not sup:          
            df = pd.merge(df, dataFF[['date',inf]], on='date',how='left')
        elif not inf: 
            df = pd.merge(df, dataFF[['date',sup]], on='date',how='left')
        else:
            df = pd.merge(df, dataFF[['date',inf,sup]], on='date',how='left')

    return df

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

def portfolio_formation_single(df, 
                           factor, 
                           holding_periods, 
                           num_port, 
                           lag=0, 
                           reverse=False, 
                           remove_outliers=False,
                           outliers=[],
                           inclusive=False,
                           weighting=False,
                           gamma=1.,
                           rank_weight = False):
    
    
    df.loc[:,'medate'] = df.loc[:,'date'].copy() + pd.offsets.MonthEnd(0)
         
    portfolios_returns={}
    for i in range(1,num_port+1):
        portfolios_returns[i]=[]
    
    
    #calcolo date riallocazione
    data_reallocation_min=df['medate'].min()+pd.offsets.MonthEnd(lag)
    data_reallocation_max=df['medate'].max()
    data_reallocation=[data_reallocation_min]
    while data_reallocation[-1] + pd.offsets.MonthEnd(holding_periods)<data_reallocation_max:
        data_reallocation.append(data_reallocation[-1] + pd.offsets.MonthEnd(holding_periods))

    #gestione di cut ed outliers
    add_cols=[]
    if remove_outliers:
        inf,sup = outliers
        if inf:
            add_cols.append(inf)
        if sup:
            add_cols.append(sup)

    #tengo solo le date di riallocazione
    df_reallocation = df.loc[df['medate'].isin(data_reallocation),['PERMNO','medate','me_lag',factor]+add_cols]

    
    # gestione del metodo di divisione in percentili e indivisuazione dei portafogli sulle date di riallocazione
    if inclusive:
        fn=split_into_deciles_inclusive
    else:
        fn=split_into_deciles_inclusive
    
    # -------------------------
    # 1) Compute port_data
    # -------------------------
    port_data = (
        df_reallocation
        .groupby('medate')
        .apply(
            fn,
            factor_column=factor,
            num_port=num_port,
            remove_outliers=remove_outliers,
            outliers=outliers
        )
        .reset_index(drop=True)
    )

    # -------------------------
    # 2) Create a DF of PERMNO lists grouped by medate & portfolio
    # -------------------------
    percentiles_permnos = (
        port_data
        .groupby(['medate', 'portfolio'])['PERMNO']
        .apply(list)
        .reset_index()
    )

    # -------------------------
    # 3) Loop through each portfolio
    # -------------------------
    reallocation_rates = []

    for port_num in range(1, num_port + 1):
        # Filter data for the current portfolio
        group_df = percentiles_permnos.query("portfolio == @port_num").copy()
        
        # Shift PERMNO list to get previous values
        group_df["previous_PERMNO"] = group_df["PERMNO"].shift()
        
        # -------------------------
        # 3a) Compute B, S
        # -------------------------
        def compute_differences(row):
            """Return sets for 'B' and 'S' (the difference in PERMNO sets)."""
            curr = set(row["PERMNO"]) if isinstance(row["PERMNO"], list) else set()
            prev = set(row["previous_PERMNO"]) if isinstance(row["previous_PERMNO"], list) else set()
            return pd.Series({
                "B": list(curr - prev),
                "S": list(prev - curr)
            })
        
        group_df[["B", "S"]] = group_df.apply(compute_differences, axis=1)

        if not weighting: 
            def compute_ew_turnover(row):
                """Compute equal-weighted turnover given current/previous PERMNO sets."""
                if not isinstance(row["previous_PERMNO"], list) or not row["previous_PERMNO"]:
                    return np.nan  # No previous row or empty list
                return 0.5 * (
                    len(row["B"]) / len(row["PERMNO"]) +
                    len(row["S"]) / len(row["previous_PERMNO"])
                )
            
            group_df["EW_turnover"] = group_df.apply(compute_ew_turnover, axis=1)
            reallocation_rates.append(group_df[['portfolio','EW_turnover']].mean())
        else: 
            # -------------------------
            # 3b) Compute VW_turnover (value-weighted) by applying a custom function
            # -------------------------
            def weighted_turnover(subgroup):
                """Assign VW_turnover to each row based on me_lag and sets B, S."""
                # subgroup is the slice of decile_permnos_mk for a given 'medate'
                date = subgroup.name
                
                # Pull the row from group_df with info for this medate
                # (We assume exactly one matching row in group_df per medate.)
                row = group_df.loc[group_df["medate"] == date].squeeze()
                
                current_permnos = row["PERMNO"] or []
                previous_permnos = row["previous_PERMNO"] or []
                b_permnos = row["B"] or []
                s_permnos = row["S"] or []
                
                wS = subgroup.loc[subgroup["PERMNO"].isin(s_permnos), "me_lag"].sum() if b_permnos else np.nan
                wB = subgroup.loc[subgroup["PERMNO"].isin(b_permnos), "me_lag"].sum() if s_permnos else np.nan
                
                w_curr = subgroup.loc[subgroup["PERMNO"].isin(current_permnos), "me_lag"].sum() or 1
                w_prev = subgroup.loc[subgroup["PERMNO"].isin(previous_permnos), "me_lag"].sum() or 1
                
                subgroup["VW_turnover"] = 0.5 * (wS / w_prev + wB / w_curr)
                return subgroup
            
            # Filter from port_data again to get all rows for this portfolio
            decile_permnos_mk = port_data.query("portfolio == @port_num")
            
            # Apply the weighted turnover function by medate
            vw_turnover = (
                decile_permnos_mk
                .groupby("medate", group_keys=False)
                .apply(weighted_turnover)[["medate", "VW_turnover"]]
                .drop_duplicates()
                .assign(portfolio=port_num)
            )

            reallocation_rates.append(vw_turnover[['portfolio','VW_turnover']].mean())
        

    # -------------------------
    # 4) Combine results
    # -------------------------
    reallocation_rates = pd.concat(reallocation_rates, axis=1).T
    
    for date in data_reallocation:
        temp_df=df.loc[(df['medate']>date) & (df['medate']<= date+pd.offsets.MonthEnd(holding_periods))]
        if weighting and rank_weight:
            ascending = True if not reverse else False 
            overall_weights= df.loc[(df['medate'] == date)]
            overall_weights['rank'] = overall_weights[factor].rank(ascending=ascending)
            overall_weights['rank'] = overall_weights['rank']-overall_weights['rank'].median()

        for portf in range(1,num_port+1):
            list_stocks=percentiles_permnos.loc[(percentiles_permnos['medate']==date)& (percentiles_permnos['portfolio']==portf)]['PERMNO'].tolist()[0]
            port_df=temp_df.loc[(temp_df['PERMNO'].isin(list_stocks))]

            if weighting:
                if rank_weight:
                    weights = overall_weights.loc[(overall_weights['PERMNO'].isin(list_stocks)), ['PERMNO', 'rank', 'me_lag']]
                    weights['weight'] = (weights['rank'])* (weights['me_lag']**gamma)
                else: 
                    weights = df.loc[(df['medate'] == date) & (df['PERMNO'].isin(list_stocks)), ['PERMNO', 'me_lag']]
                    weights['weight'] = (weights['me_lag']**gamma)

                weights['weight'] = weights['weight']/ weights['weight'].sum()
                # Apply these weights to each date in the holding period
                vw_df = port_df.merge(weights[['PERMNO', 'weight']], on='PERMNO')[['RET_RF','weight','medate']]

                wret=vw_df.groupby('medate').apply(
                    lambda x: x.assign
                            (ret=(x['RET_RF'] * x['weight']).sum()) ).reset_index(drop=True)[['ret','medate']].drop_duplicates().set_index('medate')
                if holding_periods!=1 and len(temp_df['medate'].unique())!=1:
                    wret=wret.squeeze()    
                else:
                    wret=wret['ret']
       
                portfolios_returns[portf].append(wret)
            else: 
                portfolios_returns[portf].append(port_df.groupby('medate')['RET_RF'].mean())
    
    for i in range(1,num_port+1):
        portfolios_returns[i]=pd.concat(portfolios_returns[i])
    if weighting and holding_periods==1:
        portfolios_returns[i]=portfolios_returns[i].squeeze()
   
    # portfolios_stocks[i][j]   i=numero portafoglio.  j=numero periodo
    # portfolios_weighted_returns[j] returns del portafoglio j su tutta la serie storica
    portfolios_returns=pd.DataFrame(portfolios_returns)

    # Add prefix port in front of each column
    portfolios_returns.columns = ['port' + str(col) for col in portfolios_returns.columns]

    #==================================================
    # Long-Short Portfolio Returns  
    #==================================================
   
    long_port=f'port{num_port}' if not reverse else f'port1'
    short_port=  'port1' if not reverse else f'port{num_port}'
    short_port_nr=  1 if not reverse else num_port

    # Evaluate long-short returns
    if not (weighting and rank_weight):
        portfolios_returns['long_short'] = portfolios_returns[long_port] - portfolios_returns[short_port]   
    else: 
        short_portfolio=[]
        for date in data_reallocation:
            temp_df=df.loc[(df['medate']>date) & (df['medate']<= date+pd.offsets.MonthEnd(holding_periods))]

            ascending = False if not reverse else True 
            overall_weights= df.loc[(df['medate'] == date)]
            overall_weights['rank'] = overall_weights[factor].rank(ascending=ascending)
            overall_weights['rank'] = overall_weights['rank']-overall_weights['rank'].median()

            list_stocks=percentiles_permnos.loc[(percentiles_permnos['medate']==date)& (percentiles_permnos['portfolio']==short_port_nr)]['PERMNO'].tolist()[0]
            port_df=temp_df.loc[(temp_df['PERMNO'].isin(list_stocks))]

            weights = overall_weights.loc[(overall_weights['PERMNO'].isin(list_stocks)), ['PERMNO', 'rank', 'me_lag']]
            weights['weight'] = (weights['rank'])*(weights['me_lag']**gamma)
            weights['weight'] = weights['weight']/weights['weight'].sum()
                    
            # Apply these weights to each date in the holding period
            vw_df = port_df.merge(weights[['PERMNO', 'weight']], on='PERMNO')[['RET_RF','weight','medate']]

            wret=vw_df.groupby('medate').apply(
                lambda x: x.assign
                        (ret=(x['RET_RF'] * x['weight']).sum()) ).reset_index(drop=True)[['ret','medate']].drop_duplicates().set_index('medate')
            if holding_periods!=1 and len(temp_df['medate'].unique())!=1:
                wret=wret.squeeze()    
            else:
                wret=wret['ret']

            short_portfolio.append(wret)
        short_portfolio = pd.concat(short_portfolio)
    
        
        portfolios_returns['long_short_BAB'] = portfolios_returns[long_port] - short_portfolio 
        
               

    return portfolios_returns, reallocation_rates

def calculate_weighted_return(x):
    # Debug: print the group
    print(f"Processing group: {x['medate'].unique()}")
    # Debug: check NaNs
    print(f"NaNs in RET_RF: {x['RET_RF'].isna().sum()}, NaNs in me_lag: {x['me_lag'].isna().sum()}")
    
    # Handle NaN cases: check if all values are NaN before calculating
    if not x["RET_RF"].isna().all() and not x["me_lag"].isna().all():
        ret = np.nanmean(x["RET_RF"] * x["me_lag"]) / np.nanmean(x["me_lag"])
    else:
        ret = 0  # or another value as needed

    return x.assign(ret=ret)

def invert_signs_in_df(df,factors,signs):
    df0=df[factors].copy()
    for id, factor in enumerate(factors):
        if type(signs) is list:
            df0.loc[:,factor]=signs[id]* df0.loc[:,factor]
        elif type(signs) is dict:
            df0.loc[:,factor]=signs[factor]* df0.loc[:,factor]          
    return df0

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

def static_rank_rolling_single(df,
                    data_reallocation,
                    configuration):
    
    voters=configuration['default_voters']
    signs=configuration['default_signs']
    
    mj_window=configuration['mj_window']
    weighting=configuration['weighting']
    
    method=configuration['method']
    num_cat=configuration['num_cat']
    treat_na_mj=configuration['treat_na_mj']
    
    if weighting:
        factor='wmj'
    else:
        factor='mj'
        
    df.loc[:,[f'r_{v}' for v in voters]]=np.nan
        
    for data in data_reallocation:

        mask_rolling=mask_rolling_window(df,data,mj_window)
        subdf=df.loc[mask_rolling]

        list_permno_on_reallocation_date=subdf.loc[subdf['medate']==data]['PERMNO'].unique().tolist()
        subdf=subdf.loc[subdf['PERMNO'].isin(list_permno_on_reallocation_date)]
       
        subdf.loc[:,[f'r_{v}' for v in voters]]=compute_categorical_votes_qcut(subdf,voters,signs,num_cat).values
        
        for date in subdf['medate'].unique().tolist():
            
            date_df=subdf.loc[subdf['medate']==date]
            
            M=copy.copy(date_df[[f'r_{v}' for v in voters]])
            disentagle_df=copy.copy(date_df[[f'{v}' for v in voters]])
            disentagle_df=invert_signs_in_df(disentagle_df,voters,signs)
            mjfactor=compute_ranking(M,
                                     treat_na_mj,
                                     method,
                                     disentagle_df=disentagle_df)
            df.loc[df.index.isin(M.index),factor]=mjfactor/max(mjfactor)
    
    df.loc[:,factor]=-df.groupby(['PERMNO'])[factor].rolling(mj_window, min_periods=1).mean().reset_index(level=0)[factor]
    return df, None

def rank_rolling_single_shapley(df,
                                data_reallocation,
                                configuration):
        
    default_voters=configuration['default_voters']

    mj_window=configuration['mj_window']    
    method=configuration['method']
    num_cat=configuration['num_cat']
    treat_na_mj=configuration['treat_na_mj']

    verbose=configuration['verbose']
    weighting=configuration['weighting']

    if weighting:
        factor='wmj'
    else:
        factor='mj'
        
    mj_voters={
        'date':[],
        factor:[]}
    df.loc[:,[f'r_{v}' for v in default_voters]]=np.nan


    for data in data_reallocation:
        print('reallocation date MJ', data)
        voters, signs = CSA_voters_selection_single(df, data, default_voters, configuration)
        
        if verbose:        
            print(f'date: {data}, n. voters= {len(voters)}')
            print(f'new voters: {voters} and signs {signs}')
        mj_voters['date'].append(data)
        mj_voters[factor].append(voters)


        mask_rolling=mask_rolling_window(df,data,mj_window)
        subdf=df.loc[mask_rolling]

        list_permno_on_reallocation_date=subdf.loc[subdf['medate']==data]['PERMNO'].unique().tolist()
        subdf=subdf.loc[subdf['PERMNO'].isin(list_permno_on_reallocation_date)]
        
        
        subdf.loc[:,[f'r_{v}' for v in voters]]=compute_categorical_votes_qcut(subdf,voters,signs,num_cat).values
        
        for date in subdf['medate'].unique().tolist():
            
            date_df=subdf.loc[subdf['medate']==date]
            
            M=copy.copy(date_df[[f'r_{v}' for v in voters]])
            disentagle_df=copy.copy(date_df[[f'{v}' for v in voters]])
            disentagle_df=invert_signs_in_df(disentagle_df,voters,signs)
            mjfactor=compute_ranking(M,
                                     treat_na_mj,
                                     method,
                                     disentagle_df=disentagle_df)
            df.loc[df.index.isin(M.index),factor]=mjfactor/max(mjfactor)

            
    df.loc[:,factor]=-df.groupby(['PERMNO'])[factor].rolling(mj_window, min_periods=1).mean().reset_index(level=0)[factor]
    return df, mj_voters

def static_profile_rolling_single(df,
                 data_reallocation,
                 configuration):
    
    voters=configuration['default_voters']
    signs=configuration['default_signs']

    mj_window=configuration['mj_window']
    weighting=configuration['weighting']
    
    method=configuration['method']
    num_cat=configuration['num_cat']
    treat_na_mj=configuration['treat_na_mj']
    
    if weighting:
        factor='wmj'
    else:
        factor='mj'
    
    
    df.loc[:,[f'av{f}' for f in voters]]=np.nan
    df.loc[:,[f'r_{v}' for v in voters]]=np.nan
    df[factor]=np.nan
    
    for data in data_reallocation:
        
        mask_rolling=mask_rolling_window(df,data,mj_window)
        df_rolling=df.loc[mask_rolling]
        
        list_permno_on_reallocation_date=df_rolling.loc[df_rolling['medate']==data]['PERMNO'].unique().tolist()
        #compute rolling values of real-valued factor values
        mean_factors=df_rolling.loc[df_rolling['PERMNO'].isin(list_permno_on_reallocation_date)][['PERMNO']+voters].groupby('PERMNO').mean().reset_index()
        mask_reallocation_data_and_PERMNO=(df['medate']==data)&(df['PERMNO'].isin(list_permno_on_reallocation_date))
        df.loc[mask_reallocation_data_and_PERMNO,[f'av{f}' for f in voters]]=mean_factors[voters].values

            
        df.loc[mask_reallocation_data_and_PERMNO,[f'r_{v}' for v in voters]]=compute_categorical_votes_qcut(df.loc[mask_reallocation_data_and_PERMNO],
                                                                                                       [f'av{v}' for v in voters],
                                                                                                       signs,
                                                                                                       num_cat).values
        
        date_df=df.loc[mask_reallocation_data_and_PERMNO]
        
        M=copy.copy(date_df[[f'r_{v}' for v in voters]])
        disentagle_df=copy.copy(date_df[[f'{v}' for v in voters]])
        disentagle_df=invert_signs_in_df(disentagle_df,voters,signs)
        mjfactor=compute_ranking(M,
                                 treat_na_mj,
                                 method,
                                 disentagle_df=disentagle_df)
        df.loc[df.index.isin(M.index),factor]=-mjfactor

    return df, None

def profile_rolling_single_shapley(df,
                 data_reallocation,
                 configuration):
    
    
    default_voters=configuration['default_voters']

    mj_window=configuration['mj_window']
    method=configuration['method']
    num_cat=configuration['num_cat']
    treat_na_mj=configuration['treat_na_mj']

    verbose=configuration['verbose']
    weighting=configuration['weighting']

    if weighting:
        factor='wmj'
    else:
        factor='mj'

    
    mj_voters={
        'date':[],
        factor:[]}
    
    df.loc[:,[f'av{f}' for f in default_voters]]=np.nan
    df.loc[:,[f'r_{v}' for v in default_voters]]=np.nan
    df[factor]=np.nan
    
    for data in data_reallocation:
        
        voters, signs = CSA_voters_selection_single(df, data, default_voters, configuration)
        
        if verbose:
            print(f'new voters: {voters} and signs {signs}')
            print(f'date: {data}, n. voters= {len(voters)}')
        mj_voters['date'].append(data)
        mj_voters[factor].append(voters)

    
        mask_rolling=mask_rolling_window(df,data,mj_window)
        df_rolling=df.loc[mask_rolling]
        
        list_permno_on_reallocation_date=df_rolling.loc[df_rolling['medate']==data]['PERMNO'].unique().tolist()
        #compute rolling values of real-valued factor values
        mean_factors=df_rolling.loc[df_rolling['PERMNO'].isin(list_permno_on_reallocation_date)][['PERMNO']+default_voters].groupby('PERMNO').mean().reset_index()
        mask_reallocation_data_and_PERMNO=(df['medate']==data)&(df['PERMNO'].isin(list_permno_on_reallocation_date))
        df.loc[mask_reallocation_data_and_PERMNO,[f'av{f}' for f in default_voters]]=mean_factors[default_voters].values

        

        df.loc[mask_reallocation_data_and_PERMNO,[f'r_{v}' for v in voters]]=compute_categorical_votes_qcut(df.loc[mask_reallocation_data_and_PERMNO],
                                                                                                       [f'av{v}' for v in voters],
                                                                                                       signs,
                                                                                                       num_cat).values
        
        date_df=df.loc[mask_reallocation_data_and_PERMNO]
        
        M=copy.copy(date_df[[f'r_{v}' for v in voters]])
        disentagle_df=copy.copy(date_df[[f'{v}' for v in voters]])
        disentagle_df=invert_signs_in_df(disentagle_df,voters,signs)
        mjfactor=compute_ranking(M,
                                 treat_na_mj,
                                 method,
                                 disentagle_df=disentagle_df)
        df.loc[df.index.isin(M.index),factor]=-mjfactor
        
    return df, mj_voters

def static_vote_rolling_single(df,
                 data_reallocation,
                 configuration):
    
    voters=configuration['default_voters']
    signs=configuration['default_signs']

    mj_window=configuration['mj_window']
    weighting=configuration['weighting']
    
    method=configuration['method']
    num_cat=configuration['num_cat']
    treat_na_mj=configuration['treat_na_mj']
    
    if weighting:
        factor='wmj'
    else:
        factor='mj'

    
    df['mj_vote']=np.nan
    df.loc[:,[f'r_{v}' for v in voters]]=np.nan
    
    
    for data in data_reallocation:
        
        mask_rolling=mask_rolling_window(df,data,mj_window)
        subdf=df.loc[mask_rolling]

        list_permno_on_reallocation_date=subdf.loc[subdf['medate']==data]['PERMNO'].unique().tolist()
        mask_reallocation_data_and_PERMNO=(df['medate']==data)&(df['PERMNO'].isin(list_permno_on_reallocation_date))
        
        subdf.loc[:,[f'r_{v}' for v in voters]]=compute_categorical_votes_qcut(subdf,voters,signs,num_cat).values
        
        for date in subdf['medate'].unique().tolist():

            date_df=subdf.loc[subdf['medate']==date]
            M=copy.copy(date_df[[f'r_{v}' for v in voters]])
            M=treat_nan_values_MJ(M,treat_na_mj)
            df.loc[df.index.isin(M.index),'mj_vote']=select_votes(M)

        M_rolling=df.loc[mask_rolling & df['PERMNO'].isin(list_permno_on_reallocation_date)][['date','PERMNO','mj_vote']].groupby('PERMNO')['mj_vote'].agg(lambda x: x.tolist())
        M_rolling=M_rolling.apply(map_list)
        disentagle_df=copy.copy(date_df[[f'{v}' for v in voters]])
        disentagle_df=invert_signs_in_df(disentagle_df,voters,signs)
        df.loc[mask_reallocation_data_and_PERMNO,factor]=-compute_ranking(M_rolling,
                                                                        treat_na_mj,
                                                                        method,
                                                                        disentagle_df=disentagle_df)
    return df, None

def vote_rolling_single_shapley(df,
                                data_reallocation,
                                configuration):
    
    
    default_voters=configuration['default_voters']

    mj_window=configuration['mj_window']
    method=configuration['method']
    num_cat=configuration['num_cat']
    treat_na_mj=configuration['treat_na_mj']

    verbose=configuration['verbose']
    weighting=configuration['weighting']

    if weighting:
        factor='wmj'
    else:
        factor='mj'
        

    mj_voters={
        'date':[],
        factor:[]}
    df['mj_vote']=np.nan
    df.loc[:,[f'r_{v}' for v in voters]]=np.nan
    
    
    for data in data_reallocation:

        voters, signs = CSA_voters_selection_single(df, data, default_voters, configuration)
        
        if verbose:
            print(f'date: {data}, n. voters= {len(voters)}')
            print(f'new voters: {voters} and signs {signs}')
      
        mj_voters['date'].append(data)
        mj_voters[factor].append(voters)


        mask_rolling=mask_rolling_window(df,data,mj_window)
        subdf=df.loc[mask_rolling]

        list_permno_on_reallocation_date=subdf.loc[subdf['medate']==data]['PERMNO'].unique().tolist()
        mask_reallocation_data_and_PERMNO=(df['medate']==data)&(df['PERMNO'].isin(list_permno_on_reallocation_date))

        


        subdf.loc[:,[f'r_{v}' for v in voters]]=compute_categorical_votes_qcut(subdf,voters,signs,num_cat).values
        
        for date in subdf['medate'].unique().tolist():

            date_df=subdf.loc[subdf['medate']==date]
            M=copy.copy(date_df[[f'r_{v}' for v in voters]])
            M=treat_nan_values_MJ(M,treat_na_mj)
            df.loc[df.index.isin(M.index),'mj_vote']=select_votes(M)


        M_rolling=df.loc[mask_rolling & df['PERMNO'].isin(list_permno_on_reallocation_date)][['date','PERMNO','mj_vote']].groupby('PERMNO')['mj_vote'].agg(lambda x: x.tolist())
        M_rolling=M_rolling.apply(map_list)
        disentagle_df=copy.copy(date_df[[f'{v}' for v in voters]])
        disentagle_df=invert_signs_in_df(disentagle_df,voters,signs)
        df.loc[mask_reallocation_data_and_PERMNO,factor]=-compute_ranking(M_rolling,
                                                                        treat_na_mj,
                                                                        method,
                                                                        disentagle_df=disentagle_df)

    return df, mj_voters
 
def static_profile_vote_rolling_single(df,
                 data_reallocation,
                 configuration):
    
    
    voters=configuration['default_voters']
    signs=configuration['default_signs']
    
    mj_window=configuration['mj_window']
    weighting=configuration['weighting']
    
    method=configuration['method']
    num_cat=configuration['num_cat']
    treat_na_mj=configuration['treat_na_mj']
    
    if weighting:
        factor='wmj'
    else:
        factor='mj'

    
    df['mj_vote']=np.nan
    df.loc[:,[f'r_{v}' for v in voters]]=np.nan
    df.loc[:,[f'{f}_median' for f in voters]]=np.nan
       
    for data in data_reallocation:
        
        mask_rolling=mask_rolling_window(df,data,mj_window)
        list_permno_on_reallocation_date=df.loc[df['medate']==data]['PERMNO'].unique().tolist()
        mask_reallocation_data_and_PERMNO=(df['medate']==data)&(df['PERMNO'].isin(list_permno_on_reallocation_date))
        
        subdf=df.loc[(mask_rolling)& (df['PERMNO'].isin(list_permno_on_reallocation_date))]

        subdf.loc[:,[f'r_{v}' for v in voters]]=compute_categorical_votes_qcut(subdf,voters,signs,num_cat).values
        
        median_votes=subdf[['PERMNO']+[f'r_{v}' for v in voters]].groupby('PERMNO').median().reset_index()
        
        df.loc[mask_reallocation_data_and_PERMNO,[f'{v}_median' for v in voters]]=median_votes[[f'r_{v}' for v in voters]].values
    
        date_df=df.loc[mask_reallocation_data_and_PERMNO]
        
        M=copy.copy(date_df[[f'{v}_median' for v in voters]])
        disentagle_df=copy.copy(date_df[[f'{v}' for v in voters]])
        disentagle_df=invert_signs_in_df(disentagle_df,voters,signs)
        mjfactor=compute_ranking(M,
                                 treat_na_mj,
                                 method,
                                 disentagle_df=disentagle_df)
        df.loc[df.index.isin(M.index),factor]=-mjfactor
        

    return df, None

def profile_vote_rolling_single_shapley(df,
                 data_reallocation,
                 configuration):
    
    
    default_voters=configuration['default_voters']
 
    mj_window=configuration['mj_window']
    method=configuration['method']
    num_cat=configuration['num_cat']
    treat_na_mj=configuration['treat_na_mj']

    verbose=configuration['verbose']
    weighting=configuration['weighting']
    
    if weighting:
        factor='wmj'
    else:
        factor='mj'

    mj_voters={
        'date':[],
        factor:[]}
    
    df['mj_vote']=np.nan
    df.loc[:,[f'r_{v}' for v in voters]]=np.nan
    df.loc[:,[f'{f}_median' for f in voters]]=np.nan
       
    for data in data_reallocation:

        voters, signs = CSA_voters_selection_single(df, data, default_voters, configuration)
        
        if verbose:
            print(f'date: {data}, n. voters= {len(voters)}')
            print(f'new voters: {voters} and signs {signs}')

        mj_voters['date'].append(data)
        mj_voters[factor].append(voters)

        
        mask_rolling=mask_rolling_window(df,data,mj_window)
        list_permno_on_reallocation_date=df.loc[df['medate']==data]['PERMNO'].unique().tolist()
        mask_reallocation_data_and_PERMNO=(df['medate']==data)&(df['PERMNO'].isin(list_permno_on_reallocation_date))
        
        subdf=df.loc[(mask_rolling)& (df['PERMNO'].isin(list_permno_on_reallocation_date))]

        subdf.loc[:,[f'r_{v}' for v in voters]]=compute_categorical_votes_qcut(subdf,voters,signs,num_cat).values
        
        median_votes=subdf[['PERMNO']+[f'r_{v}' for v in voters]].groupby('PERMNO').median().reset_index()
        
        df.loc[mask_reallocation_data_and_PERMNO,[f'{v}_median' for v in voters]]=median_votes[[f'r_{v}' for v in voters]].values
    
        date_df=df.loc[mask_reallocation_data_and_PERMNO]
        
        M=copy.copy(date_df[[f'{v}_median' for v in voters]])
        disentagle_df=copy.copy(date_df[[f'{v}' for v in voters]])
        disentagle_df=invert_signs_in_df(disentagle_df,voters,signs)
        mjfactor=compute_ranking(M,
                                 treat_na_mj,
                                 method,
                                 disentagle_df=disentagle_df)
        df.loc[df.index.isin(M.index),factor]=-mjfactor
        

    return df, mj_voters

def compute_factor_strategies(df:pd.DataFrame,
                                configuration:dict):
    
    factors=configuration['factors']
    K=configuration['K']
    lag=configuration['lag']
    num_port=configuration['num_port']
    n_jobs=configuration['n_jobs']
    remove_outliers=configuration['remove_outliers']
    inclusive=configuration['inclusive']
    all_voters_not_nan_on_reallocation=configuration['all_voters_not_nan_on_reallocation']


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
    #print('factor reallocation dates:', data_reallocation)
    
    if all_voters_not_nan_on_reallocation: 
        nan_dict={}
        for f in factors:
            #remove rows having nans in any of the factors cells at reallocation dates
            nan_dict[f]=~(df['medate'].isin(data_reallocation) & df[factors].isna().any(axis=1))
    else:
        #remove rows having nans in the factors cells at reallocation dates
        nan_dict={}
        for f in factors:
            nan_dict[f]=~(df['medate'].isin(data_reallocation) & df[f].isna())
        ###


    portfolios={}
    weighted_portfolios={}
    portfolios_stock_reallocation={}
    

    reverse_list=[el == -1 for el in configuration['default_signs']]
    
    ## compute single portfolio returns
    results_factors = Parallel(n_jobs=n_jobs)(
        delayed(portfolio_formation_single)(df.loc[nan_dict[factor]], 
                                      factor, 
                                      K, 
                                      num_port,
                                      lag,
                                      reverse=reverse,
                                      remove_outliers=remove_outliers,
                                      outliers=outliers,
                                      inclusive=inclusive,
                                      weighting=False) for factor, reverse in zip(factors,reverse_list))
    
    for i,res in enumerate(results_factors):
        portfolios_stock_reallocation[factors[i]]={}
        portfolios[factors[i]], portfolios_stock_reallocation[factors[i]]["EW_turnover"] = res


    ## compute single portfolio returns
    results_factors = Parallel(n_jobs=n_jobs)(
        delayed(portfolio_formation_single)(df.loc[nan_dict[factor]], 
                                      factor, 
                                      K, 
                                      num_port,
                                      lag,
                                      reverse=reverse,
                                      remove_outliers=remove_outliers,
                                      outliers=outliers,
                                      inclusive=inclusive,
                                      weighting=True) for factor, reverse in zip(factors,reverse_list))
    
    for i,res in enumerate(results_factors):
        weighted_portfolios[factors[i]], portfolios_stock_reallocation[factors[i]]["VW_turnover"] = res

    #print('factor portfolio dates:', portfolios[factors[0]]['port1'].index)
    return portfolios, weighted_portfolios, portfolios_stock_reallocation

#change investment period to mj_rolling in the mj computation
def compute_MJ_portfolio_strategy(df:pd.DataFrame,
                                  configuration_default:dict,
                                  gamma = 1.,
                                  rank_weight = False):

    configuration=configuration_default.copy()

    K=configuration['K']
    lag=configuration['lag']
    num_port=configuration['num_port']
    remove_outliers=configuration['remove_outliers']
    rolling_method=configuration['rolling_method']
    inclusive=configuration['inclusive']
    fix_signs=configuration['fix_signs']
    all_voters_not_nan_on_reallocation=configuration['all_voters_not_nan_on_reallocation']
    weighting=configuration['weighting']
    
    if weighting == True:
        factor='wmj'
    elif weighting == False:
        factor = 'mj'
        

    if remove_outliers:
        outliers=configuration['outliers']
    else:
        outliers=[]
    
    
    df['medate'] = df['date'] + pd.offsets.MonthEnd(0)
    df['wRET']=df["RET_RF"] * df["me_lag"]
    df['mj']=np.nan
    df['wmj']=np.nan
    
    if fix_signs:
        fns_mjrolling={'rank':static_rank_rolling_single,
                       'vote':static_vote_rolling_single,
                       'profile':static_profile_rolling_single,
                       'profile_vote':static_profile_vote_rolling_single}
    else:
        fns_mjrolling={'rank':rank_rolling_single_shapley,
                       'vote':vote_rolling_single_shapley,
                       'profile':profile_rolling_single_shapley,
                       'profile_vote':profile_vote_rolling_single_shapley}
        voting_window=configuration['voting_window']
        lag=lag+voting_window

    ### compute reallocation date 
    data_reallocation_min=df['medate'].min()+pd.offsets.MonthEnd(lag)
    data_reallocation_max=df['medate'].max()
    data_reallocation=[data_reallocation_min]
    while data_reallocation[-1] + pd.offsets.MonthEnd(K)<data_reallocation_max:
        data_reallocation.append(data_reallocation[-1] + pd.offsets.MonthEnd(K))

    #print('mjreallocation dates', data_reallocation)
    #print('fix_signs', fix_signs)
    
    if fix_signs and all_voters_not_nan_on_reallocation:
        #remove rows having nans in the factors cells at reallocation dates
        factors=configuration['factors']
        mask_factor_nans_at_reallocation_dates=~(df['medate'].isin(data_reallocation) & df[factors].isna().any(axis=1))

    
    ###
    portfolios={}
    portfolios_stock_reallocation={}

    df,mj_voters= fns_mjrolling[rolling_method](df,
                                                data_reallocation,
                                                configuration)
    

    if fix_signs and all_voters_not_nan_on_reallocation:
        df=df.loc[mask_factor_nans_at_reallocation_dates]  
    

      
    portfolios[factor], portfolios_stock_reallocation[factor] = portfolio_formation_single(df,
                                                                                            factor, 
                                                                                            K, 
                                                                                            num_port, 
                                                                                            lag,
                                                                                            remove_outliers=remove_outliers,
                                                                                            outliers=outliers,
                                                                                            inclusive=inclusive,
                                                                                            weighting=weighting,
                                                                                            gamma=gamma,
                                                                                            rank_weight=rank_weight)
    
    return portfolios, mj_voters, portfolios_stock_reallocation


### SHAPLEY ### 

def calculate_shapley_value(values, players, num_players):
    # Initialize Shapley values for each player
    shapley_values = {player: 0 for player in players}
    
    # Iterate over each player to calculate their Shapley value
    for player in players:
        # Iterate over all possible coalitions (subsets of players)
        for coalition in itertools.chain.from_iterable(itertools.combinations(players, r) for r in range(num_players+1)):
            if player not in coalition:
                # Compute the marginal contribution of the player to the coalition
                coalition_with_player = tuple(sorted(coalition + (player,)))
                v_without = values.get(coalition, 0)
                v_with = values.get(coalition_with_player, 0)
                marginal_contribution = v_with - v_without
                
                # Weight the marginal contribution based on the coalition size
                coalition_size = len(coalition)
                weight = (factorial(coalition_size) * factorial(num_players - coalition_size - 1)) / factorial(num_players)
                shapley_values[player] += weight * marginal_contribution
    
    return shapley_values

def calculate_approximate_shapley_value(values, players, size, small):
    # Initialize Shapley values for each player
    shapley_values = {player: 0 for player in players}
    num_players=len(players)

    if small:
        coalitions=get_combinations_up_to_size(players, max_size=size)
    else:
        coalitions=get_combinations_from_min_size(players, min_size=size)

    # Iterate over each player to calculate their Shapley value
    for player in players:
        # Iterate over all possible coalitions (subsets of players)
        for coalition in coalitions:
            if player not in coalition:
                # Compute the marginal contribution of the player to the coalition
                coalition_with_player = tuple(sorted(coalition + (player,)))
                v_without = values.get(coalition, 0)
                v_with = values.get(coalition_with_player, 0)
                marginal_contribution = v_with - v_without
                
                # Weight the marginal contribution based on the coalition size
                coalition_size = len(coalition)
                weight = (factorial(coalition_size) * factorial(num_players - coalition_size - 1)) / factorial(num_players)
                shapley_values[player] += weight * marginal_contribution
    
    return shapley_values

def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def get_coalitions(voters):
    coalitions = []
    for r in range(1, len(voters) + 1):
        coalitions.extend(itertools.combinations(voters, r))

    # Convert tuples to lists if needed
    coalitions = [list(comb) for comb in coalitions]
    return coalitions

# ----------------------------- Greedy Coalition Building -----------------------------

def compute_full_reward_and_Shapley_values(df, 
                                           voters_df, 
                                           MJ_configuration):
    players=voters_df['factors'].tolist()
    coalitions=get_coalitions(players)

    ## compute single portfolio returns
    results_reward = Parallel(n_jobs=MJ_configuration['n_jobs'], verbose=10)(
        delayed(process_coalition)(coalition, 
                                  df,
                                  voters_df,
                                  MJ_configuration) for coalition in coalitions)

    reward_values = { (): 0, }
    for el in results_reward:
        coalition, reward = el
        reward_values[tuple(coalition)]=reward
    reward_values = sort_coalition_keys(reward_values)
    # Compute Shapley values
    shapley_values = calculate_shapley_value(reward_values, players, len(players))
    return shapley_values, reward_values

def greedy_coalition_building(players, characteristic_function, desired_team_size, verbose=False):
    """
    Greedy Coalition Building based on marginal contributions.

    Parameters:
    - players: list of players (e.g., ['ag', 'beta', 'bm', ...])
    - characteristic_function: function that takes a sorted tuple of players and returns the coalition's value
    - desired_team_size: int, number of players to select

    Returns:
    - selected_coalition: list of selected players in the order they were added
    """
    selected_coalition = []
    remaining_players = set(players)
    current_coalition = tuple(sorted(selected_coalition))
    current_value = characteristic_function(current_coalition)

    if verbose:
        print("Starting Greedy Coalition Building...")
        print(f"Initial coalition: {selected_coalition} with value {current_value}\n")

    for step in range(1, desired_team_size + 1):
        best_player = None
        best_marginal_contribution = float('-inf')

        for player in remaining_players:
            # Create a new coalition with the player added
            new_coalition = tuple(sorted(selected_coalition + [player]))
            # Compute the value of the new coalition
            new_value = characteristic_function(new_coalition)
            # Compute marginal contribution
            marginal_contribution = new_value - current_value

            # Debug: Print marginal contribution for each player
            if verbose:
                print(f"Evaluating player '{player}':")
                print(f"  Coalition without player: {tuple(sorted(selected_coalition))}")
                print(f"  Coalition with player: {new_coalition}")
                print(f"  Marginal Contribution: {marginal_contribution}\n")

            # Select the player with the highest marginal contribution
            if marginal_contribution > best_marginal_contribution:
                best_marginal_contribution = marginal_contribution
                best_player = player
        if verbose:
            print('Best marginal contribution:', best_marginal_contribution)

        if best_player is not None:
            # Add the best player to the coalition
            selected_coalition.append(best_player)
            remaining_players.remove(best_player)
            # Update current coalition and value
            current_coalition = tuple(sorted(selected_coalition))
            current_value = characteristic_function(current_coalition)

            if verbose:
                print(f"Step {step}: Selected '{best_player}' with marginal contribution {best_marginal_contribution}")
                print(f"  Updated coalition: {selected_coalition} with value {current_value}\n")
        else:
            if verbose:
                print("No improvement possible. Ending selection.")
            break  # No improvement, end early

    return selected_coalition

# ----------------------------- Characteristic Function -----------------------------

def characteristic_function_factory(reward_values):
    """
    Creates a characteristic function that retrieves the coalition's reward from the reward_values dictionary.

    Parameters:
    - reward_values (dict): Dictionary mapping sorted tuples of players to their rewards.

    Returns:
    - characteristic_function (function): Function that takes a sorted tuple of players and returns the reward.
    """
    def characteristic_function(coalition):
        """
        Retrieves the reward for a given coalition.

        Parameters:
        - coalition (tuple): Sorted tuple of players.

        Returns:
        - float: Reward associated with the coalition.
        """
        return reward_values.get(tuple(sorted(coalition)), 0.0)
    
    return characteristic_function

def select_top_k_Shapley_values(shapley_values,k,players):
    res = sorted(list(range(len(shapley_values.values()))), key = lambda sub: list(shapley_values.values())[sub])[-k:]
    return [players[i] for i in res]

# ---------------------- CSA ---------------------------

def sort_coalition_keys(existing_values):
    """
    Sorts the keys of the existing_reward_values dictionary to ensure consistency.
    
    Parameters:
    - existing_values (dict): Original reward values with unsorted coalition keys.
    
    Returns:
    - sorted_values (dict): Reward values with sorted tuple keys.
    """
    sorted_values = {}
    for coalition, value in existing_values.items():
        sorted_coalition = tuple(sorted(coalition))
        sorted_values[sorted_coalition] = value
    return sorted_values

def get_combinations_up_to_size(players, max_size=5):
    """
    Generates all possible coalitions up to a specified size.

    Parameters:
    - players (list): List of all players.
    - max_size (int): Maximum size of coalitions to include.

    Returns:
    - List of tuples representing coalitions.
    """
    combinations = []
    for r in range(1, min(max_size, len(players)) + 1):
        combinations.extend(itertools.combinations(players, r))
    return combinations

def get_combinations_from_min_size(players, min_size=5):
    """
    Generates all possible coalitions up to a specified size.

    Parameters:
    - players (list): List of all players.
    - max_size (int): Maximum size of coalitions to include.

    Returns:
    - List of tuples representing coalitions.
    """
    combinations = []
    for r in range(max(min_size,1), len(players)):
        combinations.extend(itertools.combinations(players, r))
    return combinations

def filter_df(df,date,window):
    date=pd.to_datetime(date)
    reduced_df=df.copy()
    l_window=date - pd.offsets.MonthEnd(window)
    u_window=date
    reduced_df=reduced_df.loc[(reduced_df.date >= l_window) & (reduced_df.date <= u_window)]
    return reduced_df

def process_coalition(coalition,
                      df, 
                      voters_df, 
                      MJ_configuration):
    
    MJ_configuration_coalition=MJ_configuration.copy()
    MJ_configuration_coalition['default_voters']=coalition
    MJ_configuration_coalition['default_signs']=[voters_df.loc[voters_df['factors']== factor,'signs'].iloc[0] for factor in coalition]
    MJ_configuration_coalition['verbose']=False
    MJ_configuration_coalition['lag']=0
    MJ_configuration_coalition['K']=1
    MJ_configuration_coalition['fix_signs']=True
    
    MJ_portfolios, _, _ = compute_MJ_portfolio_strategy(df,
                                                    MJ_configuration_coalition)
    
    weighting = MJ_configuration_coalition['weighting']
    if weighting == True:
        factor='wmj'
    elif weighting == False:
        factor = 'mj'

    reward=MJ_portfolios[factor]['long_short'].mean()

    return coalition, reward
    
def select_statistically_meaningful_factors(df,MJ_configuration_default,players,reallocation_date,alpha=0.5,min_voters=5): ## DA MIGLIORARE
    
    window = MJ_configuration_default['sign_voting_window']
    reallocation_date=pd.to_datetime(reallocation_date)+pd.offsets.MonthEnd(0)
    reduced_df=filter_df(df,reallocation_date,window)

    #annullo il lag qua se no non usa tutti i valori nella window
    #da notare che in questo modo le strategie dei fattori vengono riallocate ogni mese (K=1) invece che ogni K mesi
    MJ_configuration=MJ_configuration_default.copy()
    MJ_configuration['verbose']=False
    MJ_configuration['lag']=0
    MJ_configuration['K']=1

    
    portfolios, weighted_portfolios, _ = compute_factor_strategies(reduced_df,MJ_configuration)

    factors_df={'factors':[],
                'signs':[],
                'pval':[],
                'tval':[]}
    wfactors_df={'factors':[],
                'signs':[],
                'pval':[],
                'tval':[]}

    '''
    from scipy import stats
    for player in players:
        res = stats.wilcoxon(portfolios[player]['long_short'])
        factors_df['factors'].append(player)
        factors_df['signs'].append(np.sign(portfolios[player]['long_short'].mean()))
        factors_df['pval'].append(res.pvalue)

        res = stats.wilcoxon(weighted_portfolios[player]['long_short'])
        wfactors_df['factors'].append(player)
        wfactors_df['signs'].append(np.sign(portfolios[player]['long_short'].mean()))
        wfactors_df['pval'].append(res.pvalue)
    '''
    
    for player in players:
        series = portfolios[player]['long_short']
        X = np.ones((len(series), 1))  # single column of ones
        model = sm.OLS(series, X)
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags":5})
        factors_df['factors'].append(player)
        factors_df['signs'].append(np.sign(results.params[0]))
        factors_df['pval'].append(results.pvalues[0])
        factors_df['tval'].append(results.tvalues[0])


        series = weighted_portfolios[player]['long_short']
        X = np.ones((len(series), 1))  # single column of ones
        model = sm.OLS(series, X)
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags":5})
        wfactors_df['factors'].append(player)
        wfactors_df['signs'].append(np.sign(results.params[0]))
        wfactors_df['pval'].append(results.pvalues[0])
        wfactors_df['tval'].append(results.pvalues[0])
    
    
    factors_df=pd.DataFrame(factors_df)
    wfactors_df=pd.DataFrame(wfactors_df)

    if len(factors_df.loc[factors_df['pval']<alpha])<min_voters:
        factors_df = factors_df.sort_values(by='pval').iloc[:min_voters]
    else:
        factors_df = factors_df.loc[factors_df['pval']<alpha]

    if len(wfactors_df.loc[wfactors_df['pval']<alpha])<min_voters:
        wfactors_df = wfactors_df.sort_values(by='pval').iloc[:min_voters]
    else:
        wfactors_df = wfactors_df.loc[wfactors_df['pval']<alpha]
    
    return factors_df, wfactors_df

def get_reduced_shapley_values(df,players,voters_df,MJ_configuration,size, small=True,n_jobs=1):
    

    if small:
        combinations=get_combinations_up_to_size(players, max_size=size)
        print(len(combinations))
    else:
        combinations=get_combinations_from_min_size(players, min_size=size)
        print(len(combinations))

    print('processing coalitions:')
    ## compute single portfolio returns
    reward_values={}
    reduced_reward = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_coalition)(list(coalition), 
                                   df,
                                   voters_df,
                                   MJ_configuration) for coalition in combinations)
    
    

    for el in reduced_reward:
        coalition, reward = el
        reward_values[tuple(coalition)]=reward

    reward_values = sort_coalition_keys(reward_values)

    shapley_values = calculate_shapley_value(reward_values, players, len(players))
    return shapley_values

def get_reduced_reward_values(df,
                               players,
                               voters_df,
                               MJ_configuration,
                               size, 
                               small=True, 
                               n_jobs=-1):
    

    if small:
        combinations=get_combinations_up_to_size(players, max_size=size)
        print(len(combinations))
    else:
        combinations=get_combinations_from_min_size(players, min_size=size)
        print(len(combinations))

    print('processing coalitions:')
    ## compute single portfolio returns
    reward_values={}
    reduced_reward = Parallel(n_jobs=n_jobs, verbose=MJ_configuration['verbose'])(
        delayed(process_coalition)(list(coalition), 
                                   df,
                                   voters_df,
                                   MJ_configuration) for coalition in combinations)

    for el in reduced_reward:
        coalition, reward = el
        reward_values[tuple(coalition)]=reward

    reward_values = sort_coalition_keys(reward_values)
    return reward_values

def find_players_to_remove(shapley_values,e,delta):
    
    sorted_shapley_dict={k: v for k, v in sorted(shapley_values.items(), key=lambda x: x[1],reverse=False)}
    sorted_values=list(sorted_shapley_dict.values())
    sorted_keys=list(sorted_shapley_dict.keys())

    to_remove=[]
    i=0
    terminated=False
    while len(to_remove)<e and (not terminated):
        if sorted_values[i]<delta:
            to_remove.append(sorted_keys[i])
            i+=1
        else:
            terminated=True
    return to_remove

def coalition_selection_algorithm_single(df,voters_df,MJ_configuration,size,delta,e,small=True):
    
    selected_players=voters_df['factors'].to_list()
    reward_values = get_reduced_reward_values(df,
                                            selected_players,
                                            voters_df,
                                            MJ_configuration,
                                            size, 
                                            small,  
                                            n_jobs=-1)

    min_shapley_value=-np.inf

    while min_shapley_value<delta:
        
        shapley_values = calculate_approximate_shapley_value(reward_values, selected_players, size, small)
        min_shapley_value = min(list(shapley_values.values()))
        to_remove=find_players_to_remove(shapley_values,e,delta)
        selected_players = list(set(selected_players)-set(to_remove))

        if MJ_configuration['verbose']:
            print('min shapley value: ', min_shapley_value)
            print('to be removed: ', to_remove)
            print('remaining players: ', selected_players)
        
    return selected_players

def CSA_voters_selection_single(df, reallocation_date, players, configuration):
    
    alpha = configuration['p_threshold']
    min_voters = configuration['min_voters']
    
    small = configuration['small']
    delta = configuration['delta_utility']
    e = configuration['eliminations']
    size = configuration['players_batch_size']

    weighting = configuration['weighting']
    if not weighting:
        voters_df, _=select_statistically_meaningful_factors(df,configuration,players,reallocation_date,alpha,min_voters)
    else:
        _, voters_df=select_statistically_meaningful_factors(df,configuration,players,reallocation_date,alpha,min_voters)

    #for f in voters_df['factors'].tolist():
    #    voters_df.loc[voters_df['factors']==f,'signs']=configuration['default_signs'][int(np.where(np.array(configuration['default_voters'])==f)[0])]


    window_shapley = configuration['voting_window']
    shapley_df=filter_df(df,reallocation_date,window_shapley)

    ### Perform CSA
    selected_players_CSA = coalition_selection_algorithm_single(shapley_df,voters_df,configuration,size=size,delta=delta,e=e,small=small)
    selected_players_signs_CSA = [voters_df.loc[voters_df['factors']== factor,'signs'].iloc[0] for factor in selected_players_CSA]

    return selected_players_CSA, selected_players_signs_CSA

