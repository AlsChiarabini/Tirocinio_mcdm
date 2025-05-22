import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import QuantileTransformer
from pymcdm.methods import TOPSIS, VIKOR, PROMETHEE_II
from pymcdm.weights import equal_weights
from ordering_MJ import voting_ordering
import matplotlib.pyplot as plt

# Lista completa dei criteri
criteri = [
    "MarketCap", "PriceToBook", "Beta", "DividendYield",
    "Return_6m", "Momentum_6m", "Volatility", "BookValue",
    "PE", "PB", "ROE", "ROA", "DebtToEquity", "SharpeRatio"
]

# Lista di un sottoinsieme di criteri per studiare portafogli mcdm con 4 criteri
subset_criteri = {
    "1": ["MarketCap", "Beta", "Momentum_6m", "SharpeRatio"],
    "2": ["DividendYield", "Return_6m", "Volatility", "ROE"]
}

# 0,5. Calcola Sharpe-ratio nei file non normalizzati
def calcola_sharpe_ratio(path="dataset_mcdm", anni=[2021, 2022, 2023, 2024]):
    for anno in anni:
        filename = os.path.join(path, f"mcdm_{anno}.csv")
        df = pd.read_csv(filename, index_col=0)

        if "SharpeRatio" in df.columns:
            print(f"✔️ SharpeRatio già presente in {filename}, salto.")
            continue

        if "Return_6m" in df.columns and "Volatility" in df.columns:
            # Calcolo Sharpe solo dove entrambi sono disponibili
            df["SharpeRatio"] = df["Return_6m"] / df["Volatility"]
            df.to_csv(filename)
            print(f" Aggiunto SharpeRatio a {filename} ({df['SharpeRatio'].notna().sum()} aziende calcolate)")
        else:
            print(f"⚠️ Colonne 'Return_6m' o 'Volatility' mancanti in {filename}, salto...")

# 1. Calcola solo Momentum nei file non normalizzati
def calcola_momentum(path="dataset_mcdm", anni=[2021, 2022, 2023, 2024]):
    for anno in anni:
        filename = os.path.join(path, f"mcdm_{anno}.csv")
        df = pd.read_csv(filename, index_col=0)

        if "Momentum_6m" in df.columns:
            print(f"✔️ Momentum_6m già presente in {filename}, salto.")
            continue

        if "Return_6m" in df.columns:
            # Calcolo il rank percentuale solo per chi ha Return_6m valido
            df["Momentum_6m"] = df["Return_6m"].rank(pct=True)
            df.to_csv(filename)
            print(f" Aggiunto Momentum_6m a {filename} ({df['Momentum_6m'].notna().sum()} aziende calcolate)")
        else:
            print(f"⚠️ Colonna 'Return_6m' mancante in {filename}, salto...")

# 2. Normalizza i file per MCDM classici
def normalizza_matrici(path="dataset_mcdm", anni=[2021, 2022, 2023, 2024]):
    for anno in anni:
        file = os.path.join(path, f"mcdm_{anno}.csv")
        df = pd.read_csv(file, index_col=0)
        df = df.dropna(subset=criteri)

        scaler = QuantileTransformer(output_distribution='uniform', random_state=0)
        df_norm = df.copy()
        df_norm[criteri] = scaler.fit_transform(df[criteri])

        out_file = file.replace(".csv", "_normalized.csv")
        df_norm.to_csv(out_file)
        print(f" Salvato file normalizzato: {out_file}")

# 3. Carica le matrici normalizzate per i metodi MCDM classici
def compute_matrix(anni=[2021, 2022, 2023, 2024]):
    matrici = {}
    for anno in anni:
        path = f"dataset_mcdm/mcdm_{anno}_normalized.csv"
        df = pd.read_csv(path, index_col=0)
        matrici[anno] = df[criteri]
    return matrici

# 4. Costruzione portafogli MCDM classici
def costruisci_portafogli_mcdm(matrix_years, decile=0.1, save_path="portafogli_mcdm"):
    os.makedirs(save_path, exist_ok=True)

    metodi = {
        "topsis": TOPSIS(),
        "vikor": VIKOR(v=0.5),
        "promethee": PROMETHEE_II(preference_function='usual')
    }

    portafogli = {}

    for anno, df in matrix_years.items():
        matrix = df.values
        weights = equal_weights(matrix)
        types = [1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1]
        tickers = df.index.tolist()

        for nome, metodo in metodi.items():
            print(f"\n Applico {nome.upper()} per anno {anno}...")

            scores = metodo(matrix, weights, types) # Come vuole pymcdm
            df_risultati = pd.DataFrame({
                "Ticker": tickers,
                "Score": scores
            }).sort_values("Score", ascending=False)

            top_n = int(len(df_risultati) * decile)
            top_aziende = df_risultati.head(top_n)["Ticker"].tolist()

            portafogli[(anno, nome)] = top_aziende

            df_risultati.head(top_n).to_csv(
                os.path.join(save_path, f"portafoglio_{nome}_{anno}.csv"),
                index=False
            )

            print(f"Salvato portafoglio {nome} {anno} con {top_n} aziende")

    return portafogli

# 5. Costruzione portafogli con metodo MJ (MJ della prof)
def costruisci_portafogli_MJ(anni=[2021,2022,2023,2024], path="dataset_mcdm", decile=0.1, save_path="portafogli_mcdm"):
    os.makedirs(save_path, exist_ok=True)
    portafogli_mj = {}

    for anno in anni:
        print(f"\n MJ --> Elaboro anno {anno}")
        file = os.path.join(path, f"mcdm_{anno}.csv")
        df = pd.read_csv(file, index_col=0)
        df = df.dropna(subset=criteri)
        sub_df = df[criteri]

        M = sub_df.rank(pct=True)
        M = pd.qcut(M.stack(), q=6, labels=[1, 2, 3, 4, 5, 6]).unstack().astype(int)

        _, _, new_indices, _ = voting_ordering(M.values, disentagle_df=sub_df)

        tickers = sub_df.index[new_indices].tolist()
        top_n = int(len(tickers) * decile)
        top_aziende = tickers[:top_n]

        portafogli_mj[anno] = top_aziende

        pd.DataFrame(top_aziende, columns=["Ticker"]).to_csv(
            os.path.join(save_path, f"portafoglio_MJ_{anno}.csv"), index=False
        )

        print(f"Salvato portafoglio MJ {anno} con {top_n} aziende")

        # {
            #(2021, "MJ"): ["AAPL", "MSFT", "NVDA"],
            #(2021, "TOPSIS"): [...],
            #(2022, "MJ"): [...],
            #...
        #}

    return portafogli_mj

# 6. Calcola rendimenti futuri equal-weighted
def calcola_rendimenti_portafogli(
    path="dataset_mcdm",
    anni=[2021, 2022, 2023, 2024],
    portafogli_dict={}
):
    risultati = []

    for anno in anni:
        print(f"\n📈 Calcolo rendimenti portafogli per anno {anno}...")
        df = pd.read_csv(os.path.join(path, f"mcdm_{anno}.csv"), index_col=0)

        # Benchmark
        if "Return_6m" in df.columns:
            rendimento_benchmark = df["Return_6m"].dropna().mean()
            risultati.append((anno, "benchmark", rendimento_benchmark))

        # Prendi tutti i metodi presenti nel dizionario
        metodi_anno = [metodo for (a, metodo) in portafogli_dict.keys() if a == anno]
        metodi_anno = list(set(metodi_anno))  # rimuove duplicati

        for metodo in metodi_anno:
            tickers = portafogli_dict.get((anno, metodo))
            if not tickers:
                print(f"Nessun portafoglio trovato per {metodo} {anno}, salto...")
                continue

            subset = df.loc[df.index.intersection(tickers)]
            subset = subset.dropna(subset=["Return_6m"])
            rendimento = subset["Return_6m"].mean()

            risultati.append((anno, metodo, rendimento))

    # Costruzione DataFrame
    df_risultati = pd.DataFrame(risultati, columns=["Anno", "Metodo", "Rendimento"])
    df_risultati = df_risultati.sort_values(["Anno", "Metodo"]).reset_index(drop=True)
    df_risultati.to_csv("rendimenti_portafogli.csv", index=False)

    # Stampa ordinata
    print("\nRENDIMENTI PER METODO (ORDINATI PER ANNO):")
    for anno in df_risultati["Anno"].unique():
        print("\n" + "*" * 40)
        print(f"ANNO: {anno}")
        print("*" * 40)
        print(df_risultati[df_risultati["Anno"] == anno][["Metodo", "Rendimento"]].to_string(index=False))

    return df_risultati

# 7. Grafico dei vari portafogli
def plot_crescita_cumulata(df_rendimenti, capitale_iniziale=1.0):
    df = df_rendimenti.copy()
    df = df.sort_values(["Metodo", "Anno"])

    df["Capitale"] = df.groupby("Metodo")["Rendimento"].transform(
        lambda x: (1 + x).cumprod() * capitale_iniziale
    )

    plt.figure(figsize=(10, 6))
    for metodo, gruppo in df.groupby("Metodo"):
        linestyle = "--" if metodo == "benchmark" else "-"
        plt.plot(gruppo["Anno"], gruppo["Capitale"], label=metodo, linestyle=linestyle)

    plt.title("Crescita Cumulata del Capitale (ribilanciamento annuale)")
    plt.xlabel("Anno")
    plt.ylabel("Capitale")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("crescita_portafogli.png", dpi=300)
    plt.show()

# 8. Calcolo delle metriche per i vari portafogli
def calcola_metriche_performance(df_rendimenti, output_path="."):

    metrics = []

    for metodo, gruppo in df_rendimenti.groupby("Metodo"):
        rendimenti = gruppo["Rendimento"].values

        avg_return = np.mean(rendimenti)
        volatility = np.std(rendimenti)
        sharpe = avg_return / volatility if volatility > 0 else np.nan
        total_return = np.prod(1 + rendimenti) - 1

        # Calcolo drawdown
        cum_returns = np.cumprod(1 + rendimenti)
        rolling_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns / rolling_max) - 1
        max_drawdown = np.min(drawdowns)

        metrics.append({
            "Metodo": metodo,
            "Rendimento_Medio": avg_return,
            "Rendimento_Totale": total_return,
            "Volatilità": volatility,
            "Sharpe_Ratio": sharpe,
            "Max_Drawdown": max_drawdown
        })

    df_metrics = pd.DataFrame(metrics)
    file_path = os.path.join(output_path, "metriche_performance.csv")
    df_metrics.to_csv(file_path, index=False)
    print(f"Salvato: {file_path}")
    return df_metrics

# 9. Costruzione di portafogli mcdm basati su 4 criteri (e non 12)
def costruisci_portafogli_mcdm_subset(matrix_years, subsets, decile=0.1, save_path="portafogli_mcdm"):

    os.makedirs(save_path, exist_ok=True)

    metodi = {
        "topsis": TOPSIS(),
        "vikor": VIKOR(v=0.5),
        "promethee": PROMETHEE_II(preference_function='usual')
    }

    portafogli = {}

    for subset_name, criteri_subset in subsets.items():
        print(f"\n🔧 Subset: {subset_name} con criteri: {criteri_subset}")
        for anno, df in matrix_years.items():
            df_filtered = df[criteri_subset].dropna()
            matrix = df_filtered.values
            weights = equal_weights(matrix)
            types = []

            for col in criteri_subset:
                # +1 se da premiare, -1 se da penalizzare
                if col in ["MarketCap", "SharpeRatio", "Momentum_6m", "ROE", "Return_6m", "DividendYield"]:
                    types.append(1)
                else:
                    types.append(-1)

            tickers = df_filtered.index.tolist()

            for nome, metodo in metodi.items():
                metodo_nome = f"{nome}_sub{subset_name}"
                print(f"{metodo_nome} - anno {anno}")

                scores = metodo(matrix, weights, types)
                df_risultati = pd.DataFrame({"Ticker": tickers, "Score": scores}).sort_values("Score", ascending=False)

                top_n = int(len(df_risultati) * decile)
                top_aziende = df_risultati.head(top_n)["Ticker"].tolist()

                portafogli[(anno, metodo_nome)] = top_aziende

                df_risultati.head(top_n).to_csv(
                    os.path.join(save_path, f"portafoglio_{metodo_nome}_{anno}.csv"), index=False
                )

                print(f"Salvato portafoglio {metodo_nome} {anno} con {top_n} aziende")

    return portafogli

# === ESECUZIONE ===
calcola_sharpe_ratio()
calcola_momentum()
normalizza_matrici()
matrix_years = compute_matrix()
portafogli_mcdm = costruisci_portafogli_mcdm(matrix_years)
portafogli_MJ = costruisci_portafogli_MJ()
portafogli_mcdm_subset = costruisci_portafogli_mcdm_subset(matrix_years, subset_criteri)

# Unione portafogli
portafogli_all = {}
for (anno, metodo), tickers in portafogli_mcdm.items():
    portafogli_all[(anno, metodo)] = tickers
for anno, tickers in portafogli_MJ.items():
    portafogli_all[(anno, "MJ")] = tickers
for (anno, metodo), tickers in portafogli_mcdm_subset.items():
    portafogli_all[(anno, metodo)] = tickers

# Calcolo rendimenti
df_rendimenti = calcola_rendimenti_portafogli(portafogli_dict=portafogli_all)
df_rendimenti.to_csv("rendimenti_portafogli.csv", index=False)
print("\nRENDIMENTI PORTAFOGLI:\n")
print(df_rendimenti)

df_metriche = calcola_metriche_performance(df_rendimenti, output_path=".")
print("\nMetriche di performance:\n")
print(df_metriche)

# Visualizzazione dei rendimenti
plot_crescita_cumulata(df_rendimenti)

# Lagga market cap quando fa rendimento pesato (su suo .csv già fatto --> me_lag (1 mese)). 
# Plot con linea (excess retun)
# annualizza il tutto (altrimenti uno giugno uno dicembre sbagliato)
# prova add.dea