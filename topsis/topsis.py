import yfinance as yf
import pandas as pd
import numpy as np  
import requests
import os 
from io import StringIO
from pymcdm.methods import TOPSIS
from pymcdm.visuals import boxplot
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

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

# STEP 2: Definisco i pesi e normalizzo la matrice di valutazione
weights = np.array([0.1, 0.3, 0.4, 0.2]) 
criteria_types = [1, 1, -1, 1]  
matrix = df.values # Solo i valori perché lo richiede pyMCDM
norm_matrix = df / np.linalg.norm(matrix, axis=0)  # Norma L2, quindi la radice quafrata della somma dei quadrati
# Dataframe leggibile per la normalizzazione
norm_df = pd.DataFrame(norm_matrix, index=df.index, columns=df.columns)
print("Matrice normalizzata (prime 10 aziende):")
print(norm_df.head(10))
print("\n")

# STEP 3: Chiamo la funzione TOPSIS
topsis = TOPSIS()
scores = topsis(norm_matrix, weights, criteria_types)

# STEP 4: Ordino le aziende in base al punteggio
results = pd.DataFrame({
    "TOPSIS Score": scores
}, index=df.index)
results = results.sort_values(by="TOPSIS Score", ascending=False)  # ordina in base al punteggio
print("Classifica TOPSIS: ")
print(results.head(20))
print("\n")

def parte_grafica(norm_df, results):

    # --- 1. GRAFICO A BARRE SEMPLICE ---
    plt.figure(figsize=(14, 6))
    results['TOPSIS Score'].head(20).plot(kind='bar', color='teal')
    plt.title('Top 20 aziende secondo TOPSIS')
    plt.ylabel('Punteggio TOPSIS')
    plt.xlabel('Aziende')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('grafico_topsis_barre.png')
    plt.show()

    # --- 2. GRAFICO RADAR ---
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

    theta = radar_factory(len(norm_df.columns))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(theta)
    ax.set_xticklabels(norm_df.columns)

    colors = plt.cm.get_cmap('tab10', 5)

    # Prendiamo le prime 4 aziende in classifica
    top_companies = results.head(4).index
    for i, company in enumerate(top_companies):
        values = norm_df.loc[company].values.flatten().tolist()
        values += values[:1]  # chiude il poligono
        ax.plot(np.append(theta, theta[0]), values, color=colors(i), label=company)
        ax.fill(np.append(theta, theta[0]), values, color=colors(i), alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.title('Performance delle aziende sui criteri (normalizzati)')
    plt.tight_layout()
    plt.savefig('grafico_radar.png')
    plt.show()

parte_grafica(norm_df, results)