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

# Importa i metodi MCDM da pymcdm
from pymcdm.methods import TOPSIS, VIKOR, PROMETHEE_II

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

# Spiegazione dei criteri
print("Descrizione dei criteri:")
print("- MarketCap: La capitalizzazione di mercato dell'azienda (max)")
print("- Momentum_6m: Performance degli ultimi 6 mesi (max)")
print("- Volatility: Volatilità annualizzata dei rendimenti (min)")
print("- Return_6m: Rendimento degli ultimi 6 mesi (max)")
print("\n")

# Pesi dei criteri (somma a 1 per convenzione, ma non sempre necessario)
# Giustificazione: diamo peso maggiore alla volatilità (0.4) perché in ottica di
# gestione del rischio è il criterio più importante, seguito dal momentum (0.3)
weights = np.array([0.1, 0.3, 0.4, 0.2])
print(f"Pesi dei criteri: {weights}")
print("Giustificazione: MarketCap (0.1), Momentum_6m (0.3), Volatility (0.4), Return_6m (0.2)")
print("La volatilità ha il peso maggiore per riflettere l'importanza della gestione del rischio")
print("\n")

# Tipi di criteri: 1 per massimizzare (beneficio), -1 per minimizzare (costo)
types = np.array([1, 1, -1, 1])
print("Tipi di criteri:")
print("MarketCap: massimizzare (+1)")
print("Momentum_6m: massimizzare (+1)")
print("Volatility: minimizzare (-1)")
print("Return_6m: massimizzare (+1)")
print("\n")

# N per Top N ranking
TOP_N = 20

# -- 2. Applicazione Metodi MCDM --

# Normalizzazione della matrice decisionale
matrix = df.values 
norm_matrix = df / np.linalg.norm(matrix, axis=0)  # Norma L2
print("Matrice decisionale normalizzata (prime 5 righe):")
print(pd.DataFrame(norm_matrix, index=df.index, columns=df.columns).head(5))
print("\n")

# Inizializza i metodi
topsis = TOPSIS()
# VIKOR richiede il parametro 'v' (solitamente 0.5)
vikor = VIKOR(v=0.5)
# PROMETHEE II usa di default la 'usual preference function'
promethee = PROMETHEE_II(preference_function='usual') 
print("Metodi MCDM inizializzati:")
print("- TOPSIS: metodo basato sulla distanza dall'alternativa ideale")
print("- VIKOR: metodo con parametro v=0.5 (equilibrio tra utilità di gruppo e rammarico individuale)")
print("- PROMETHEE II: metodo con funzione di preferenza 'usual' (qualsiasi differenza è preferenza totale)")
print("\n")

# Calcola i punteggi e i ranking
pref_topsis = topsis(norm_matrix, weights, types)
rank_topsis = topsis.rank(pref_topsis)

pref_vikor = vikor(norm_matrix, weights, types)
rank_vikor = vikor.rank(pref_vikor)
# Nota: in VIKOR, punteggi più bassi sono migliori, quindi rank corretti

pref_promethee = promethee(norm_matrix.astype(float), weights.astype(float), types.astype(float))
rank_promethee = promethee.rank(pref_promethee)

# Crea un DataFrame per gestire facilmente i risultati
results = pd.DataFrame({
    'Alternative': df.index,
    'Score_TOPSIS': pref_topsis,
    'Rank_TOPSIS': rank_topsis,
    'Score_VIKOR': pref_vikor,
    'Rank_VIKOR': rank_vikor,
    'Score_PROMETHEE': pref_promethee,
    'Rank_PROMETHEE': rank_promethee
})

print("--- Applicazione Metodi MCDM Completata ---")
print("Interpretazione dei punteggi:")
print("- TOPSIS: punteggi più alti sono migliori")
print("- VIKOR: punteggi più bassi sono migliori")
print("- PROMETHEE II: punteggi più alti sono migliori")
print("-" * 30)

# -- 3. Top N Ranking --
print(f"--- Top {TOP_N} Ranking ---")

print("\nTOPSIS:")
topsis_top = results.sort_values('Rank_TOPSIS').head(TOP_N)[['Alternative', 'Rank_TOPSIS', 'Score_TOPSIS']]
print(topsis_top)

print("\nVIKOR:")
vikor_top = results.sort_values('Rank_VIKOR').head(TOP_N)[['Alternative', 'Rank_VIKOR', 'Score_VIKOR']]
print(vikor_top)

print("\nPROMETHEE II:")
promethee_top = results.sort_values('Rank_PROMETHEE').head(TOP_N)[['Alternative', 'Rank_PROMETHEE', 'Score_PROMETHEE']]
print(promethee_top)
print("-" * 30)

# -- 4. Confronto tra i Top 10 ranking --
print("--- Analisi delle overlap nei Top 10 ranking ---")
top10_topsis = set(results.sort_values('Rank_TOPSIS').head(10)['Alternative'])
top10_vikor = set(results.sort_values('Rank_VIKOR').head(10)['Alternative'])
top10_promethee = set(results.sort_values('Rank_PROMETHEE').head(10)['Alternative'])

print(f"Overlap TOPSIS-VIKOR: {len(top10_topsis & top10_vikor)} elementi comuni nei top 10")
print(f"Overlap TOPSIS-PROMETHEE: {len(top10_topsis & top10_promethee)} elementi comuni nei top 10")
print(f"Overlap VIKOR-PROMETHEE: {len(top10_vikor & top10_promethee)} elementi comuni nei top 10")
print(f"Elementi comuni in tutti e tre i metodi: {len(top10_topsis & top10_vikor & top10_promethee)}")

common_alternatives = list(top10_topsis & top10_vikor & top10_promethee)
if common_alternatives:
    print("\nAlternative presenti nei Top 10 di tutti i metodi:")
    for alt in common_alternatives:
        topsis_rank = results[results['Alternative'] == alt]['Rank_TOPSIS'].values[0]
        vikor_rank = results[results['Alternative'] == alt]['Rank_VIKOR'].values[0]
        promethee_rank = results[results['Alternative'] == alt]['Rank_PROMETHEE'].values[0]
        print(f"- {alt}: TOPSIS #{topsis_rank}, VIKOR #{vikor_rank}, PROMETHEE #{promethee_rank}")
print("-" * 30)

# -- 5. Correlazione tra Ranking --
print("--- Correlazione tra Ranking (Spearman Rho & Kendall Tau) ---")

spearman_topsis_vikor, p_s_tv = spearmanr(results['Rank_TOPSIS'], results['Rank_VIKOR'])
kendall_topsis_vikor, p_k_tv = kendalltau(results['Rank_TOPSIS'], results['Rank_VIKOR'])
print(f"TOPSIS vs VIKOR: Spearman Rho = {spearman_topsis_vikor:.4f}, Kendall Tau = {kendall_topsis_vikor:.4f}")
print(f"p-value (Spearman): {p_s_tv:.4e}, p-value (Kendall): {p_k_tv:.4e}")

spearman_topsis_promethee, p_s_tp = spearmanr(results['Rank_TOPSIS'], results['Rank_PROMETHEE'])
kendall_topsis_promethee, p_k_tp = kendalltau(results['Rank_TOPSIS'], results['Rank_PROMETHEE'])
print(f"TOPSIS vs PROMETHEE II: Spearman Rho = {spearman_topsis_promethee:.4f}, Kendall Tau = {kendall_topsis_promethee:.4f}")
print(f"p-value (Spearman): {p_s_tp:.4e}, p-value (Kendall): {p_k_tp:.4e}")

spearman_vikor_promethee, p_s_vp = spearmanr(results['Rank_VIKOR'], results['Rank_PROMETHEE'])
kendall_vikor_promethee, p_k_vp = kendalltau(results['Rank_VIKOR'], results['Rank_PROMETHEE'])
print(f"VIKOR vs PROMETHEE II: Spearman Rho = {spearman_vikor_promethee:.4f}, Kendall Tau = {kendall_vikor_promethee:.4f}")
print(f"p-value (Spearman): {p_s_vp:.4e}, p-value (Kendall): {p_k_vp:.4e}")

# Interpretazione dei risultati di correlazione
print("\nInterpretazione:")
correlations = {
    "TOPSIS-VIKOR": spearman_topsis_vikor,
    "TOPSIS-PROMETHEE": spearman_topsis_promethee,
    "VIKOR-PROMETHEE": spearman_vikor_promethee
}
max_corr = max(correlations.items(), key=lambda x: abs(x[1]))
min_corr = min(correlations.items(), key=lambda x: abs(x[1]))

print(f"- La correlazione più forte è tra {max_corr[0]}: {max_corr[1]:.4f}")
print(f"- La correlazione più debole è tra {min_corr[0]}: {min_corr[1]:.4f}")
if all(corr > 0.7 for corr in correlations.values()):
    print("- Tutti i metodi mostrano una forte correlazione positiva tra i ranking")
elif all(corr > 0.5 for corr in correlations.values()):
    print("- Tutti i metodi mostrano una correlazione moderata positiva tra i ranking")
else:
    print("- I metodi mostrano correlazioni variabili, suggerendo differenze significative nei ranking")

print("-" * 30)

# -- 5.5. Heatmap Correlazione --
# Crea matrice delle correlazioni (Spearman)
corr_matrix = results[['Rank_TOPSIS', 'Rank_VIKOR', 'Rank_PROMETHEE']].corr(method='spearman')

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1)
plt.title('Correlazione di Spearman tra Rankings', fontsize=14)
plt.savefig('heatmap_rankings_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# -- 6. Visualizzazione Comparativa (Scatter Plot Scores) --
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(data=results, x='Score_TOPSIS', y='Score_VIKOR')
plt.title('TOPSIS vs VIKOR Scores', fontsize=12)
plt.xlabel('TOPSIS Score (Higher is better)', fontsize=10)
plt.ylabel('VIKOR Score (Lower is better!)', fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
sns.scatterplot(data=results, x='Score_TOPSIS', y='Score_PROMETHEE')
plt.title('TOPSIS vs PROMETHEE II Scores', fontsize=12)
plt.xlabel('TOPSIS Score (Higher is better)', fontsize=10)
plt.ylabel('PROMETHEE II Score (Higher is better)', fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
sns.scatterplot(data=results, x='Score_VIKOR', y='Score_PROMETHEE')
plt.title('VIKOR vs PROMETHEE II Scores', fontsize=12)
plt.xlabel('VIKOR Score (Lower is better!)', fontsize=10)
plt.ylabel('PROMETHEE II Score (Higher is better)', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scatter_comparison_scores.png', dpi=300, bbox_inches='tight')
plt.show()

# -- 7. Analisi di Sensibilità (Variazione Pesi) --
print("--- Analisi di Sensibilità (Variazione Pesi) ---")

# Creiamo 3 scenari di pesi per l'analisi di sensibilità
weights_orig = weights.copy()  # Scenario 1: Pesi Originali
rank_topsis_orig = results['Rank_TOPSIS'].copy()
rank_vikor_orig = results['Rank_VIKOR'].copy()
rank_promethee_orig = results['Rank_PROMETHEE'].copy()

# Prepara matrice per i risultati della sensibilità
sensitivity_results = []

# Lista di scenari di pesi per l'analisi di sensibilità
weight_scenarios = [
    {"name": "Baseline", "weights": weights_orig, "desc": "Pesi originali"},
    {"name": "MarketCap+", "weights": np.array([0.3, 0.2, 0.3, 0.2]), "desc": "Maggiore importanza a MarketCap"},
    {"name": "Momentum+", "weights": np.array([0.1, 0.5, 0.3, 0.1]), "desc": "Maggiore importanza a Momentum"},
    {"name": "Volatility+", "weights": np.array([0.1, 0.1, 0.7, 0.1]), "desc": "Maggiore importanza a Volatility"},
    {"name": "Return+", "weights": np.array([0.1, 0.2, 0.3, 0.4]), "desc": "Maggiore importanza a Return"}
]

# Esegui ogni scenario
for scenario in weight_scenarios:
    print(f"\nScenario: {scenario['name']} - {scenario['desc']}")
    print(f"Pesi: {scenario['weights']}")
    
    # Calcola i ranking per ogni metodo con i nuovi pesi
    pref_topsis_s = topsis(norm_matrix, scenario['weights'], types)
    rank_topsis_s = topsis.rank(pref_topsis_s)
    
    pref_vikor_s = vikor(norm_matrix, scenario['weights'], types)
    rank_vikor_s = vikor.rank(pref_vikor_s)
    
    pref_promethee_s = promethee(norm_matrix.astype(float), scenario['weights'].astype(float), types.astype(float))
    rank_promethee_s = promethee.rank(pref_promethee_s)
    
    # Calcola correlazioni con i ranking originali
    spearman_topsis, _ = spearmanr(rank_topsis_orig, rank_topsis_s)
    spearman_vikor, _ = spearmanr(rank_vikor_orig, rank_vikor_s)
    spearman_promethee, _ = spearmanr(rank_promethee_orig, rank_promethee_s)
    
    # Conserva i risultati
    sensitivity_results.append({
        'Scenario': scenario['name'],
        'Description': scenario['desc'],
        'Weights': str(scenario['weights']),
        'TOPSIS_Stability': spearman_topsis,
        'VIKOR_Stability': spearman_vikor,
        'PROMETHEE_Stability': spearman_promethee
    })
    
    # Mostra le correlazioni
    print(f"Stabilità TOPSIS: {spearman_topsis:.4f}")
    print(f"Stabilità VIKOR: {spearman_vikor:.4f}")
    print(f"Stabilità PROMETHEE II: {spearman_promethee:.4f}")
    
    # Analisi del cambio nelle top 5 alternative
    top5_orig_topsis = set(results.sort_values('Rank_TOPSIS').head(5)['Alternative'])
    top5_orig_vikor = set(results.sort_values('Rank_VIKOR').head(5)['Alternative'])
    top5_orig_promethee = set(results.sort_values('Rank_PROMETHEE').head(5)['Alternative'])
    
    # Crea nuovi risultati con i nuovi ranking
    new_results = pd.DataFrame({
        'Alternative': df.index,
        'Rank_TOPSIS_New': rank_topsis_s,
        'Rank_VIKOR_New': rank_vikor_s,
        'Rank_PROMETHEE_New': rank_promethee_s
    })
    
    top5_new_topsis = set(new_results.sort_values('Rank_TOPSIS_New').head(5)['Alternative'])
    top5_new_vikor = set(new_results.sort_values('Rank_VIKOR_New').head(5)['Alternative'])
    top5_new_promethee = set(new_results.sort_values('Rank_PROMETHEE_New').head(5)['Alternative'])
    
    # Misura cambiamenti nelle top 5
    changed_topsis = len(top5_orig_topsis - top5_new_topsis)
    changed_vikor = len(top5_orig_vikor - top5_new_vikor)
    changed_promethee = len(top5_orig_promethee - top5_new_promethee)
    
    print(f"Cambiamenti Top 5 - TOPSIS: {changed_topsis}/5, VIKOR: {changed_vikor}/5, PROMETHEE II: {changed_promethee}/5")

# Crea DataFrame per il riassunto della sensibilità
sensitivity_df = pd.DataFrame(sensitivity_results)
print("\n--- Riassunto Analisi di Sensibilità ---")
print(sensitivity_df[['Scenario', 'TOPSIS_Stability', 'VIKOR_Stability', 'PROMETHEE_Stability']])

# Crea grafico di sensibilità
plt.figure(figsize=(10, 6))
sensitivity_plot_data = sensitivity_df.melt(
    id_vars=['Scenario'],
    value_vars=['TOPSIS_Stability', 'VIKOR_Stability', 'PROMETHEE_Stability'],
    var_name='Method', value_name='Stability'
)

sns.barplot(x='Scenario', y='Stability', hue='Method', data=sensitivity_plot_data)
plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label='Soglia di stabilità (0.7)')
plt.title('Stabilità dei metodi alla variazione dei pesi', fontsize=14)
plt.xlabel('Scenario', fontsize=12)
plt.ylabel('Correlazione di Spearman con ranking originale', fontsize=12)
plt.ylim(0, 1)
plt.legend(title='Metodo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# -- 8. Benchmark dei tempi di esecuzione --
print("\n--- Calcolo Tempi di Esecuzione Medi ---")

# Funzione di benchmark
def benchmark_method(method, matrix, weights, types, repeats=10):
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
    'Spearman Sensibilità Media': [
        sensitivity_df['TOPSIS_Stability'].mean(), 
        sensitivity_df['VIKOR_Stability'].mean(), 
        sensitivity_df['PROMETHEE_Stability'].mean()
    ],
    'Stabilità Pesi': [
        "Alta" if sensitivity_df['TOPSIS_Stability'].min() > 0.7 else "Media" if sensitivity_df['TOPSIS_Stability'].min() > 0.5 else "Bassa",
        "Alta" if sensitivity_df['VIKOR_Stability'].min() > 0.7 else "Media" if sensitivity_df['VIKOR_Stability'].min() > 0.5 else "Bassa",
        "Alta" if sensitivity_df['PROMETHEE_Stability'].min() > 0.7 else "Media" if sensitivity_df['PROMETHEE_Stability'].min() > 0.5 else "Bassa"
    ],
    'Interpretabilità': [
        "Alta (distanza dall'ideale)", 
        "Media (max utilità di gruppo)", 
        "Alta (flussi di preferenza)"
    ]
})

# Ordina per tempo di esecuzione
summary_table = summary_table.sort_values(by='Tempo Medio (ms)')

print("\n--- Tabella Riassuntiva Prestazioni ---")
print(summary_table)

# Salva su CSV
summary_table.to_csv("summary_table_mcdm.csv", index=False)
print("✅ Tabella riassuntiva salvata in 'summary_table_mcdm.csv'")

# -- 9. Visualizzazione dei Tempi di Esecuzione (Bar Chart) --
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
plt.savefig('execution_time_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# -- 10. Analisi e visualizzazione caratteristiche top 10 --
print("\n--- Analisi Caratteristiche Top 10 ---")

# Ottieni le top 10 alternative per ogni metodo
top10_alternatives = {
    'TOPSIS': results.sort_values('Rank_TOPSIS').head(10)['Alternative'].tolist(),
    'VIKOR': results.sort_values('Rank_VIKOR').head(10)['Alternative'].tolist(),
    'PROMETHEE': results.sort_values('Rank_PROMETHEE').head(10)['Alternative'].tolist()
}

# Crea sottoinsiemi di dati per le top 10 alternative di ogni metodo
top10_data = {}
for method, alts in top10_alternatives.items():
    top10_data[method] = df.loc[alts]

# Media e deviazioni standard dei criteri per ogni top 10
stats_summary = {}
for method, data in top10_data.items():
    stats_summary[method] = {
        'mean': data.mean(),
        'std': data.std()
    }

# Visualizza le medie
print("\nMedia dei criteri nelle top 10 alternative:")
means_df = pd.DataFrame({
    'TOPSIS': stats_summary['TOPSIS']['mean'],
    'VIKOR': stats_summary['VIKOR']['mean'],
    'PROMETHEE': stats_summary['PROMETHEE']['mean']
})
print(means_df)

# Crea un radar chart per confrontare le caratteristiche medie delle top 10
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    class RadarAxes(PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
        def fill(self, *args, **kwargs):
            return super().fill(*args, **kwargs)
        def plot(self, *args, **kwargs):
            return super().plot(*args, **kwargs)
    register_projection(RadarAxes)
    return theta

# Normalizza i valori per il radar chart
means_norm = means_df.copy()
for col in means_norm.columns:
    if col == 'Volatility':  # Per volatility, valori bassi sono preferibili
        means_norm.loc['Volatility', col] = 1 / means_norm.loc['Volatility', col]
# Normalizzazione min-max per ogni riga
for idx in means_norm.index:
    min_val = means_norm.loc[idx].min()
    max_val = means_norm.loc[idx].max()
    if max_val > min_val:
        means_norm.loc[idx] = (means_norm.loc[idx] - min_val) / (max_val - min_val)

# Crea il radar chart
theta = radar_factory(len(means_norm.index))
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
ax.set_xticks(theta)
ax.set_xticklabels(means_norm.index)

colors = ['b', 'g', 'r']
for i, (method, values) in enumerate(means_norm.items()):
    values_list = values.values.flatten().tolist()
    values_list += values_list[:1]  # chiude il poligono
    ax.plot(np.append(theta, theta[0]), values_list, color=colors[i], label=method)
    ax.fill(np.append(theta, theta[0]), values_list, color=colors[i], alpha=0.1)

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.title('Confronto delle caratteristiche medie nelle Top 10 alternative', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('top10_characteristics_radar.png', dpi=300, bbox_inches='tight')
plt.show()

print("-" * 30)
print("--- Analisi Completata ---")
print("\n")