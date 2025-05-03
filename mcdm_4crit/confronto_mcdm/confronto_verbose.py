import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
import yfinance as yf
import requests
import os
from io import StringIO
import time

from pymcdm.methods import TOPSIS, VIKOR, PROMETHEE_II
from grafici import plot_heatmap_corr, plot_scatter_comparison, plot_sensitivity_barplot, plot_execution_times, plot_top10_radar

filename = "sp500_data.csv"
TOP_N = 20

# Questa la uso solo se voglio usare come criterio i valori predetti a 6 mesi dopo
# aver usato RF_mcdm.py 
def importaDataset_pred(filename="sp500_pred.csv"): 
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ File {filename} non trovato. Devi eseguire prima lo script di previsione.")
    
    df = pd.read_csv(filename, index_col=0)

    # Rinomina la colonna per compatibilità con i metodi MCDM
    if "Return_6m_Pred" in df.columns:
        df = df.rename(columns={"Return_6m_Pred": "Return_6m"})

    # Seleziona solo le colonne utili per MCDM
    colonne_utili = ["MarketCap", "Momentum_6m", "Volatility", "Return_6m"]
    colonne_presenti = [col for col in colonne_utili if col in df.columns]
    df = df[colonne_presenti]

    print(f"✅ Dataset predittivo caricato da '{filename}' con colonne: {colonne_presenti}")
    return df


def importaDataset(filename):
    if os.path.exists(filename):
        print(f"File {filename} trovato, non lo scarico di nuovo.")
    else:
        print("\U0001F504 File non trovato, scarico i dati...")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(StringIO(requests.get(url).text))
        sp500 = tables[0]
        tickers = [symbol.replace(".", "-") for symbol in sp500["Symbol"].tolist()]
        valid_tickers = []
        data = {}

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='1y')
                if hist.empty:
                    print(f"⚠️ No data for {ticker}, skipping...")
                    continue
                info = stock.info
                data[ticker] = {
                    "MarketCap": info.get("marketCap", np.nan),
                    "Momentum_6m": hist["Close"].pct_change(min(len(hist), 126)).iloc[-1] if len(hist) > 126 else np.nan,
                    "Volatility": hist["Close"].pct_change().std() * (252 ** 0.5),
                    "Return_6m": hist["Close"].pct_change(min(len(hist), 126)).iloc[-1] if len(hist) > 126 else np.nan,
                }
                valid_tickers.append(ticker)
            except Exception:
                continue

        df = pd.DataFrame.from_dict(data, orient="index")
        df.dropna(inplace=True)
        df.to_csv(filename)
        print(f"\u2705 Dati salvati in '{filename}' per {len(valid_tickers)} aziende.")
    return pd.read_csv(filename, index_col=0)

def normalizza_matrice(df):
    matrix = df.values
    norm_matrix = df / np.linalg.norm(matrix, axis=0) # Normalizzazione della matrice dei valori attraverso la norma L2
    return norm_matrix

def applica_metodi_mcdm(norm_matrix, weights, types, df_index):
    topsis = TOPSIS() # metodo basato sulla distanza dall'alternativa ideale
    vikor = VIKOR(v=0.5) # metodo con parametro v=0.5 (equilibrio tra utilità di gruppo e rammarico individuale)
    promethee = PROMETHEE_II(preference_function='usual') # metodo con funzione di preferenza 'usual' (qualsiasi differenza è preferenza totale)

    pref_topsis = topsis(norm_matrix, weights, types)
    rank_topsis = topsis.rank(pref_topsis)

    pref_vikor = vikor(norm_matrix, weights, types) # VIKOR è meglio con punteggi bassi
    rank_vikor = vikor.rank(pref_vikor)

    pref_promethee = promethee(norm_matrix.astype(float), weights.astype(float), types.astype(float))
    rank_promethee = promethee.rank(pref_promethee)

    results = pd.DataFrame({
        'Alternative': df_index,
        'Score_TOPSIS': pref_topsis,
        'Rank_TOPSIS': rank_topsis,
        'Score_VIKOR': pref_vikor,
        'Rank_VIKOR': rank_vikor,
        'Score_PROMETHEE': pref_promethee,
        'Rank_PROMETHEE': rank_promethee
    })
    print("--- Applicazione Metodi MCDM Completata ---")
    return results, topsis, vikor, promethee

def stampa_top_n(results, top_n=10):
    for metodo in ['TOPSIS', 'VIKOR', 'PROMETHEE']:
        print(f"\n{metodo}:")
        print(results.sort_values(f'Rank_{metodo}').head(top_n)[['Alternative', f'Rank_{metodo}', f'Score_{metodo}']])
    print("-" * 50)

def analizza_overlap_top10(results):
    top10 = {metodo: set(results.sort_values(f'Rank_{metodo}').head(10)['Alternative']) for metodo in ['TOPSIS', 'VIKOR', 'PROMETHEE']}
    comuni = top10['TOPSIS'] & top10['VIKOR'] & top10['PROMETHEE']
    print(f"Elementi comuni in tutti e tre i metodi: {len(comuni)}")
    for metodo1 in top10:
        for metodo2 in top10:
            if metodo1 < metodo2:
                print(f"Overlap {metodo1}-{metodo2}: {len(top10[metodo1] & top10[metodo2])}")
    if comuni:
        print("\nAlternative comuni:")
        for alt in comuni:
            ranks = [f"{m} #{results.loc[results['Alternative'] == alt, f'Rank_{m}'].values[0]}" for m in top10]
            print(f"- {alt}: " + ", ".join(ranks))
    print("-" * 50)

def correlazione_ranking(results):
    metodi = ['TOPSIS', 'VIKOR', 'PROMETHEE']
    print("--- Correlazione tra Ranking (Spearman Rho & Kendall Tau) ---")
    correlation_summary = {}
    for i in range(len(metodi)):
        for j in range(i+1, len(metodi)):
            m1, m2 = metodi[i], metodi[j]
            rho, p_rho = spearmanr(results[f'Rank_{m1}'], results[f'Rank_{m2}'])
            tau, p_tau = kendalltau(results[f'Rank_{m1}'], results[f'Rank_{m2}'])
            print(f"{m1} vs {m2}: Spearman = {rho:.4f}, Kendall = {tau:.4f}")
            print(f"\tRisultati: Spearman = {rho:.4f}, Kendall = {tau:.4f}, p-value (Spearman): {p_rho:.4e}, p-value (Kendall): {p_tau:.4e}\n")
            correlation_summary[f"{m1}-{m2}"] = rho

    print("Interpretazione:")
    max_corr = max(correlation_summary.items(), key=lambda x: abs(x[1]))
    min_corr = min(correlation_summary.items(), key=lambda x: abs(x[1]))
    print(f"- La correlazione più forte è tra {max_corr[0]}, con un indice di correlazione di {max_corr[1]:.4f}")
    print(f"- La correlazione più debole è tra {min_corr[0]}, con un indice di correlazione di {min_corr[1]:.4f}")
    if all(val > 0.7 for val in correlation_summary.values()):
        print("- Tutti i metodi mostrano una forte correlazione positiva tra i ranking")
    elif all(val > 0.5 for val in correlation_summary.values()):
        print("- Tutti i metodi mostrano una correlazione moderata positiva tra i ranking")
    else:
        print("- I metodi mostrano correlazioni variabili, suggerendo differenze significative nei ranking")
    print("-" * 50)

def analisi_sensibilita(df, norm_matrix, results, topsis, vikor, promethee, types, weight_scenarios):
    sens = []
    original_ranks = {
        'TOPSIS': results['Rank_TOPSIS'],
        'VIKOR': results['Rank_VIKOR'],
        'PROMETHEE': results['Rank_PROMETHEE']
    }
    top5_originali = {m: set(results.sort_values(f'Rank_{m}').head(5)['Alternative']) for m in original_ranks}

    for s in weight_scenarios:
        print(f"\nScenario: {s['name']} --> {s['desc']}")
        print(f"Pesi: {s['weights']}")
        w = s['weights'] # Seleziono il campo 'weights' del dizionario
        ranks = {
            'TOPSIS': topsis.rank(topsis(norm_matrix, w, types)),
            'VIKOR': vikor.rank(vikor(norm_matrix, w, types)),
            'PROMETHEE': promethee.rank(promethee(norm_matrix.astype(float), w.astype(float), types.astype(float)))
        }
        stabilita = {m: spearmanr(original_ranks[m], ranks[m]).correlation for m in ranks}
        cambi = {m: len(top5_originali[m] - set(df.iloc[ranks[m].argsort()[:5]].index)) for m in ranks}
        for m in ranks:
            print(f"Stabilità {m}: {stabilita[m]:.4f} | Cambiamenti Top 5: {cambi[m]}/5")
        sens.append({
            'Scenario': s['name'],
            'Description': s['desc'],
            'Weights': str(w),
            **{f"{m}_Stability": stabilita[m] for m in ranks}
        })
    df_sens = pd.DataFrame(sens)
    print("\n--- Riassunto Analisi di Sensibilità ---")
    print(df_sens[['Scenario', 'TOPSIS_Stability', 'VIKOR_Stability', 'PROMETHEE_Stability']])
    return df_sens

def benchmark_metodi(topsis, vikor, promethee, norm_matrix, weights, types, sensitivity_df):
    def benchmark_method(method, matrix, weights, types, repeats=10):
        times = []
        for _ in range(repeats):
            start = time.time()
            _ = method(matrix, weights, types)
            times.append(time.time() - start)
        return np.mean(times) * 1000

    time_topsis = benchmark_method(topsis, norm_matrix, weights, types)
    time_vikor = benchmark_method(vikor, norm_matrix, weights, types)
    time_promethee = benchmark_method(promethee, norm_matrix.astype(float), weights.astype(float), types.astype(float))

    summary = pd.DataFrame({
        'Metodo': ['TOPSIS', 'VIKOR', 'PROMETHEE II'],
        'Tempo Medio (ms)': [time_topsis, time_vikor, time_promethee],
        'Spearman Sensibilità Media': [
            sensitivity_df['TOPSIS_Stability'].mean(),
            sensitivity_df['VIKOR_Stability'].mean(),
            sensitivity_df['PROMETHEE_Stability'].mean()
        ],
        'Stabilità Pesi': [
            "Alta" if sensitivity_df[f'{m}_Stability'].min() > 0.7 else "Media" if sensitivity_df[f'{m}_Stability'].min() > 0.5 else "Bassa"
            for m in ['TOPSIS', 'VIKOR', 'PROMETHEE']
        ],
        'Interpretabilità': [
            "Alta (distanza dall'ideale)",
            "Media (max utilità di gruppo)",
            "Alta (flussi di preferenza)"
        ]
    }).sort_values(by='Tempo Medio (ms)')

    print("\n--- Tabella Riassuntiva Prestazioni ---")
    print(summary)
    summary.to_csv("summary_table_mcdm.csv", index=False)
    return summary

def analisi_top10_caratteristiche(df, results):
    top10 = {
        m: results.sort_values(f'Rank_{m}').head(10)['Alternative'].tolist()
        for m in ['TOPSIS', 'VIKOR', 'PROMETHEE']
    }
    data = {m: df.loc[alts] for m, alts in top10.items()}
    means = pd.DataFrame({m: d.mean() for m, d in data.items()})
    print("\nMedia dei criteri nelle top 10 alternative:")
    print(means)
    plot_top10_radar(means)

# === MAIN ===
# df = importaDataset(filename)
df = importaDataset_pred("sp500_pred.csv") # Da commentare in base a quale criterio voglio usare
print("Matrice di valutazione, solo le prime 10 come esempio:")
print(df.head(10))
print('\n')

norm_matrix = normalizza_matrice(df)
print("Matrice di valutazione normalizzata, solo le prime 5 come esempio:")
print(pd.DataFrame(norm_matrix, index=df.index, columns=df.columns).head(5))
print('\n')

weights = np.array([0.1, 0.3, 0.4, 0.2])
types = np.array([1, 1, -1, 1])

results, topsis, vikor, promethee = applica_metodi_mcdm(norm_matrix, weights, types, df.index)

stampa_top_n(results, TOP_N)
print(f"--- Analisi delle overlap delle prime {TOP_N} aziende ---")
analizza_overlap_top10(results)
correlazione_ranking(results)

plot_heatmap_corr(results) # Crea matrice delle correlazioni (Spearman). Es.: azienda A prima per vikor, terza per topsis, seconda per promethee. 3x3 perché confronto 3 vettori da 500 aziende, ma considero il vettore nel suo insieme
plot_scatter_comparison(results)

scenari = [
    {"name": "Baseline", "weights": weights.copy(), "desc": "Pesi originali"},
    {"name": "MarketCap+", "weights": np.array([0.3, 0.2, 0.3, 0.2]), "desc": "+ importanza a MarketCap"},
    {"name": "Momentum+", "weights": np.array([0.1, 0.5, 0.3, 0.1]), "desc": "+ Momentum"},
    {"name": "Volatility+", "weights": np.array([0.1, 0.1, 0.7, 0.1]), "desc": "+ Volatility"},
    {"name": "Return+", "weights": np.array([0.1, 0.2, 0.3, 0.4]), "desc": "+ Return"},
]
sensitivity_df = analisi_sensibilita(df, norm_matrix, results, topsis, vikor, promethee, types, scenari)

plot_sensitivity_barplot(sensitivity_df)
summary_table = benchmark_metodi(topsis, vikor, promethee, norm_matrix, weights, types, sensitivity_df)
plot_execution_times(summary_table)
analisi_top10_caratteristiche(df, results)

print("-" * 30)
print("--- Analisi completata ---")