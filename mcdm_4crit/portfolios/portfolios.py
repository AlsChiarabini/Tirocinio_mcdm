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
    "PE", "PB", "ROE", "ROA", "DebtToEquity"
]

# 1. Calcola solo Momentum nei file non normalizzati
def calcola_momentum(path="dataset_mcdm", anni=[2021, 2022, 2023, 2024]):
    for anno in anni:
        filename = os.path.join(path, f"mcdm_{anno}.csv")
        print(f"\n📂 Calcolo Momentum in: {filename}")
        try:
            df = pd.read_csv(filename, index_col=0)
            if "Return_6m" in df.columns:
                df = df.dropna(subset=["Return_6m"])
                df["Momentum_6m"] = df["Return_6m"].rank(pct=True)
                df.to_csv(filename)
                print(f"✅ Aggiornato con Momentum_6m ({len(df)} aziende)")
            else:
                print("⚠️ Return_6m non trovato, salto...")
        except Exception as e:
            print(f"❌ Errore con il file {anno}: {e}")

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
        print(f"✅ Salvato file normalizzato: {out_file}")

# 3. Carica le matrici normalizzate per i metodi MCDM classici
def update_matrix(anni=[2021, 2022, 2023, 2024]):
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
        types = [1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1]
        tickers = df.index.tolist()

        for nome, metodo in metodi.items():
            print(f"\n📊 Applico {nome.upper()} per anno {anno}...")

            scores = metodo(matrix, weights, types)
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

            print(f"✅ Salvato portafoglio {nome} {anno} con {top_n} aziende")

    return portafogli

# 5. Costruzione portafogli con metodo MJ (MJ della prof)
def costruisci_portafogli_MJ(anni=[2021,2022,2023,2024], path="dataset_mcdm", decile=0.1, save_path="portafogli_mcdm"):
    os.makedirs(save_path, exist_ok=True)
    portafogli_mj = {}

    for anno in anni:
        print(f"\n📂 MJ --> Elaboro anno {anno}")
        file = os.path.join(path, f"mcdm_{anno}.csv")
        df = pd.read_csv(file, index_col=0)
        df = df.dropna(subset=criteri)
        sub_df = df[criteri]

        M = sub_df.rank(pct=True)
        M = pd.qcut(M.stack(), q=3, labels=[1, 2, 3]).unstack().astype(int)

        _, _, new_indices, _ = voting_ordering(M.values, disentagle_df=sub_df)

        tickers = sub_df.index[new_indices].tolist()
        top_n = int(len(tickers) * decile)
        top_aziende = tickers[:top_n]

        portafogli_mj[anno] = top_aziende

        pd.DataFrame(top_aziende, columns=["Ticker"]).to_csv(
            os.path.join(save_path, f"portafoglio_MJ_{anno}.csv"), index=False
        )

        print(f"✅ Salvato portafoglio MJ {anno} con {top_n} aziende")

    return portafogli_mj

# 6. Calcola rendimenti futuri equal-weighted
def calcola_rendimenti_portafogli(
    path="dataset_mcdm",
    anni=[2021, 2022, 2023],
    portafogli_dict={},
    metodi=["topsis", "vikor", "promethee", "MJ"]
):
    risultati = []

    for anno in anni:
        print(f"\n📈 Calcolo rendimenti portafogli per anno {anno}...")
        df = pd.read_csv(os.path.join(path, f"mcdm_{anno}.csv"), index_col=0)

        if "Return_6m" in df.columns:
            rendimento_benchmark = df["Return_6m"].dropna().mean()
            risultati.append((anno, "benchmark", rendimento_benchmark))

        for metodo in metodi:
            tickers = portafogli_dict.get((anno, metodo))
            if not tickers:
                print(f"⚠️ Nessun portafoglio trovato per {metodo} {anno}, salto...")
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
    print("\n📊 RENDIMENTI PER METODO (ORDINATI PER ANNO):")
    anni_unici = df_risultati["Anno"].unique()
    for anno in anni_unici:
        print("\n" + "*" * 40)
        print(f"📅 ANNO: {anno}")
        print("*" * 40)
        anno_df = df_risultati[df_risultati["Anno"] == anno][["Metodo", "Rendimento"]]
        print(anno_df.to_string(index=False))

    return df_risultati

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
    print(f"📈 Salvato: {file_path}")
    return df_metrics


# === ESECUZIONE ===
calcola_momentum()
normalizza_matrici()
matrix_years = update_matrix()
portafogli_mcdm = costruisci_portafogli_mcdm(matrix_years)
portafogli_MJ = costruisci_portafogli_MJ()

# Unione portafogli
portafogli_all = {}
for (anno, metodo), tickers in portafogli_mcdm.items():
    portafogli_all[(anno, metodo)] = tickers
for anno, tickers in portafogli_MJ.items():
    portafogli_all[(anno, "MJ")] = tickers

# Calcolo rendimenti
df_rendimenti = calcola_rendimenti_portafogli(portafogli_dict=portafogli_all)
df_rendimenti.to_csv("rendimenti_portafogli.csv", index=False)
print("\n📊 RENDIMENTI PORTAFOGLI:\n")
print(df_rendimenti)

df_metriche = calcola_metriche_performance(df_rendimenti, output_path=".")
print("\n📊 Metriche di performance:\n")
print(df_metriche)

# Visualizzazione dei rendimenti
plot_crescita_cumulata(df_rendimenti)