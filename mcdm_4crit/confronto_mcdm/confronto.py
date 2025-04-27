import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
import yfinance as yf
import requests
import os
from io import StringIO

# Importa i metodi MCDM da pymcdm
from pymcdm.methods import TOPSIS, VIKOR, PROMETHEE_II
# Importa helper per la normalizzazione (utile per confrontare scores)

filename = "sp500_data.csv"
if os.path.exists(filename):
    print(f"File {filename} trovato, non lo scarico di nuovo.")
else:
    print("🔄 File non trovato, scarico i dati...")

    # 📌 Scarichiamo la lista delle aziende S&P 500
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(StringIO(requests.get(url).text))
    sp500 = tables[0]  # Prima tabella con i simboli delle aziende
    tickers = [symbol.replace(".", "-") for symbol in sp500["Symbol"].tolist()]
    valid_tickers = []
    data = {}

    # Scarichiamo i dati storici e le informazioni delle aziende
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')  # Dati storici di un anno
            if hist.empty:
                print(f"⚠️ No data for {ticker}, skipping...")
                continue
            info = stock.info

            # Creazione delle feature --> feature engineering
            data[ticker] = {
                "MarketCap": info.get("marketCap", np.nan),
                "Momentum_6m": hist["Close"].pct_change(min(len(hist), 126)).iloc[-1] if len(hist) > 126 else np.nan,
                "Volatility": hist["Close"].pct_change().std() * (252 ** 0.5),
                "Return_6m": hist["Close"].pct_change(min(len(hist), 126)).iloc[-1] if len(hist) > 126 else np.nan,
            }
            valid_tickers.append(ticker)
        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}, skipping...")
            continue

    df = pd.DataFrame.from_dict(data, orient="index")
    df.dropna(inplace=True)
    df.to_csv(filename)
    print(f"✅ Dati salvati in '{filename}' per {len(valid_tickers)} aziende su {len(tickers)} disponibili.")

# STEP 1: Creo una matrice di valutazione --> dataframe
df = pd.read_csv(filename, index_col=0)
print("Matrice di valutazione, solo le prime 10 per esempio:")
print(df.head(10))
print("\n")

# Pesi dei criteri (somma a 1 per convenzione, ma non sempre necessario)
# Esempio: pesi leggermente diversi
weights = np.array([0.1, 0.3, 0.4, 0.2])

# Tipi di criteri: 1 per massimizzare (beneficio), -1 per minimizzare (costo)
# Esempio: [Max, Min, Max, Min]
types = np.array([1, 1, -1, 1])

# N per Top N ranking
TOP_N = 20

# -- 2. Applicazione Metodi MCDM --

# Inizializza i metodi
topsis = TOPSIS()
# VIKOR richiede il parametro 'v' (solitamente 0.5)
vikor = VIKOR(v=0.5)
# PROMETHEE II usa di default la 'usual preference function'
promethee = PROMETHEE_II(preference_function='usual') # Puoi specificare 'preference_function' per criterio se necessario

# Calcola i punteggi e i ranking
# Nota: alcuni metodi potrebbero richiedere la matrice già normalizzata,
# ma pymcdm solitamente gestisce la normalizzazione internamente.
# Controlla la documentazione specifica se hai dubbi.

matrix = df.values # Solo i valori perché lo richiede pyMCDM
norm_matrix = df / np.linalg.norm(matrix, axis=0)  # Norma L2, quindi la radice quafrata della somma dei quadrati

pref_topsis = topsis(norm_matrix, weights, types)
rank_topsis = topsis.rank(pref_topsis)

pref_vikor = vikor(norm_matrix, weights, types)
rank_vikor = vikor.rank(pref_vikor)

# PROMETHEE II potrebbe richiedere tipi float per i pesi e la matrice
pref_promethee = promethee(norm_matrix.astype(float), weights.astype(float), types.astype(float))
rank_promethee = promethee.rank(pref_promethee)

# Crea un DataFrame per gestire facilmente i risultati
results = pd.DataFrame({
    'Alternative': [f'Alt_{i+1}' for i in range(df.shape[0])],
    'Score_TOPSIS': pref_topsis,
    'Rank_TOPSIS': rank_topsis,
    'Score_VIKOR': pref_vikor,
    'Rank_VIKOR': rank_vikor,
    'Score_PROMETHEE': pref_promethee,
    'Rank_PROMETHEE': rank_promethee
})

print("--- Applicazione Metodi MCDM Completata ---")
print(results.head()) # Stampa le prime righe per controllo
print("-" * 30)

# -- 3. Top N Ranking --
print(f"--- Top {TOP_N} Ranking ---")

print("\nTOPSIS:")
print(results.sort_values('Rank_TOPSIS').head(TOP_N)[['Alternative', 'Rank_TOPSIS', 'Score_TOPSIS']])

print("\nVIKOR:")
# VIKOR rank basso = migliore
print(results.sort_values('Rank_VIKOR').head(TOP_N)[['Alternative', 'Rank_VIKOR', 'Score_VIKOR']])

print("\nPROMETHEE II:")
# PROMETHEE II score alto = migliore, quindi rank basso = migliore
print(results.sort_values('Rank_PROMETHEE').head(TOP_N)[['Alternative', 'Rank_PROMETHEE', 'Score_PROMETHEE']])
print("-" * 30)


# -- 4. Correlazione tra Ranking --
print("--- Correlazione tra Ranking (Spearman Rho & Kendall Tau) ---")

spearman_topsis_vikor, p_s_tv = spearmanr(results['Rank_TOPSIS'], results['Rank_VIKOR'])
kendall_topsis_vikor, p_k_tv = kendalltau(results['Rank_TOPSIS'], results['Rank_VIKOR'])
print(f"TOPSIS vs VIKOR: Spearman Rho = {spearman_topsis_vikor:.4f}, Kendall Tau = {kendall_topsis_vikor:.4f}")

spearman_topsis_promethee, p_s_tp = spearmanr(results['Rank_TOPSIS'], results['Rank_PROMETHEE'])
kendall_topsis_promethee, p_k_tp = kendalltau(results['Rank_TOPSIS'], results['Rank_PROMETHEE'])
print(f"TOPSIS vs PROMETHEE II: Spearman Rho = {spearman_topsis_promethee:.4f}, Kendall Tau = {kendall_topsis_promethee:.4f}")

spearman_vikor_promethee, p_s_vp = spearmanr(results['Rank_VIKOR'], results['Rank_PROMETHEE'])
kendall_vikor_promethee, p_k_vp = kendalltau(results['Rank_VIKOR'], results['Rank_PROMETHEE'])
print(f"VIKOR vs PROMETHEE II: Spearman Rho = {spearman_vikor_promethee:.4f}, Kendall Tau = {kendall_vikor_promethee:.4f}")
print("-" * 30)

# -- 4.5. Heatmap Correlazione --
import seaborn as sns

# Crea matrice delle correlazioni (Spearman)
corr_matrix = results[['Rank_TOPSIS', 'Rank_VIKOR', 'Rank_PROMETHEE']].corr(method='spearman')

# Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Spearman Correlation between Rankings')
plt.show()


# -- 5. Visualizzazione Comparativa (Scatter Plot Scores) --
# Normalizziamo i punteggi per renderli confrontabili (es. min-max scaling)
# Usiamo r_normalizations da pymcdm per coerenza, ma potremmo usare anche altro
# Nota: VIKOR score più basso è meglio, altri più alto è meglio. Invertiamo VIKOR per visualizzazione?
# O plottiamo direttamente i punteggi originali se i range sono simili.
# Per semplicità, plottiamo i punteggi originali. Considera la normalizzazione se i range sono molto diversi.

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(data=results, x='Score_TOPSIS', y='Score_VIKOR')
plt.title('TOPSIS vs VIKOR Scores')
plt.xlabel('TOPSIS Score (Higher is better)')
plt.ylabel('VIKOR Score (Lower is better!)') # Nota l'interpretazione!
plt.grid(True)

plt.subplot(1, 3, 2)
sns.scatterplot(data=results, x='Score_TOPSIS', y='Score_PROMETHEE')
plt.title('TOPSIS vs PROMETHEE II Scores')
plt.xlabel('TOPSIS Score (Higher is better)')
plt.ylabel('PROMETHEE II Score (Higher is better)')
plt.grid(True)

plt.subplot(1, 3, 3)
sns.scatterplot(data=results, x='Score_VIKOR', y='Score_PROMETHEE')
plt.title('VIKOR vs PROMETHEE II Scores')
plt.xlabel('VIKOR Score (Lower is better!)') # Nota l'interpretazione!
plt.ylabel('PROMETHEE II Score (Higher is better)')
plt.grid(True)

plt.tight_layout()
plt.show()
print("--- Visualizzazione Comparativa Generata ---")
print("-" * 30)

# -- 6. Analisi di Sensibilità di Base (Variazione Pesi) --
print("--- Analisi di Sensibilità (Variazione Pesi) ---")

# Scenario 1: Pesi Originali (già calcolati)
rank_topsis_orig = results['Rank_TOPSIS'].copy()
rank_vikor_orig = results['Rank_VIKOR'].copy()
rank_promethee_orig = results['Rank_PROMETHEE'].copy()

# Scenario 2: Aumenta leggermente il peso del primo criterio
weights_scen2 = weights.copy()
delta = 0.05 # Piccolo cambiamento
weights_scen2[0] += delta
weights_scen2[1] -= delta # Diminuisci un altro per mantenere somma ~1 (o normalizza)
weights_scen2 = np.clip(weights_scen2, 0.01, 0.99) # Evita pesi negativi/nulli
weights_scen2 = weights_scen2 / np.sum(weights_scen2) # Normalizza a somma 1

print(f"Scenario Sensibilità: Pesi = {weights_scen2}")

pref_topsis_s2 = topsis(matrix, weights_scen2, types)
rank_topsis_s2 = topsis.rank(pref_topsis_s2)

pref_vikor_s2 = vikor(matrix, weights_scen2, types)
rank_vikor_s2 = vikor.rank(pref_vikor_s2)

pref_promethee_s2 = promethee(matrix.astype(float), weights_scen2.astype(float), types.astype(float))
rank_promethee_s2 = promethee.rank(pref_promethee_s2)

# Calcola la stabilità come correlazione di Spearman tra ranking originale e nuovo
stab_topsis, _ = spearmanr(rank_topsis_orig, rank_topsis_s2)
stab_vikor, _ = spearmanr(rank_vikor_orig, rank_vikor_s2)
stab_promethee, _ = spearmanr(rank_promethee_orig, rank_promethee_s2)

print(f"\nStabilità Ranking (Spearman Rho con ranking originale):")
print(f"TOPSIS: {stab_topsis:.4f}")
print(f"VIKOR: {stab_vikor:.4f}")
print(f"PROMETHEE II: {stab_promethee:.4f}")
print("(Valori più vicini a 1 indicano maggiore stabilità alla variazione dei pesi)")

import time

# -- 7. Tabella Riassuntiva delle Prestazioni --

print("--- Calcolo Tempi di Esecuzione Medi ---")

# Funzione di benchmark semplice
def benchmark_method(method, matrix, weights, types, repeats=5):
    times = []
    for _ in range(repeats):
        start = time.time()
        _ = method(matrix, weights, types)
        end = time.time()
        times.append(end - start)
    return np.mean(times) * 1000  # Ritorna tempo medio in millisecondi

# Calcolo tempi medi di esecuzione per ogni metodo
time_topsis = benchmark_method(topsis, norm_matrix, weights, types)
time_vikor = benchmark_method(vikor, norm_matrix, weights, types)
time_promethee = benchmark_method(promethee, norm_matrix.astype(float), weights.astype(float), types.astype(float))

# Crea il DataFrame riassuntivo
summary_table = pd.DataFrame({
    'Metodo': ['TOPSIS', 'VIKOR', 'PROMETHEE II'],
    'Tempo Medio (ms)': [time_topsis, time_vikor, time_promethee],
    'Spearman Sensibilità (var pesi)': [stab_topsis, stab_vikor, stab_promethee],
})

# Ordina il DataFrame per tempo di esecuzione (facoltativo)
summary_table = summary_table.sort_values(by='Tempo Medio (ms)')

print("\n--- Tabella Riassuntiva Prestazioni ---")
print(summary_table)

# Salva anche su CSV se ti serve esportarla
summary_table.to_csv("summary_table_mcdm.csv", index=False)
print("✅ Tabella riassuntiva salvata in 'summary_table_mcdm.csv'")

# -- 8. Visualizzazione dei Tempi di Esecuzione (Bar Chart) --

plt.figure(figsize=(8, 5))

# Crea il barplot
ax = sns.barplot(
    x="Tempo Medio (ms)",
    y="Metodo",
    hue="Metodo",
    data=summary_table,
    palette="Blues_d",
    dodge=False,
    legend=False
)

# Aggiungi i numeri sopra le barre
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", padding=3, fontsize=9)

# Titoli e label
plt.title('Tempo Medio di Esecuzione per Metodo MCDM', fontsize=14)
plt.xlabel('Tempo Medio (millisecondi)', fontsize=12)
plt.ylabel('Metodo', fontsize=12)

# Migliora la leggibilità
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

plt.show()



print("-" * 30)
print("--- Analisi Completata ---")