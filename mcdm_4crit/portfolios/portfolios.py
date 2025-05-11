import pandas as pd
import os
from sklearn.preprocessing import QuantileTransformer
from pymcdm.methods import TOPSIS, VIKOR, PROMETHEE_II
from pymcdm.weights import equal_weights

criteri = [
    "MarketCap", "PriceToBook", "Volatility", "BookValue",
    "PE", "PB", "ROE", "ROA", "Beta", "Return_6m", "Momentum_6m"
]

def calcola_momentum_e_normalizza(path="dataset_mcdm", anni=[2021, 2022, 2023, 2024]):
    criteri = [
        "MarketCap", "PriceToBook", "Beta", "DividendYield",
        "Return_6m", "Volatility", "BookValue", "PE", "PB",
        "ROE", "ROA", "DebtToEquity"
    ]

    for anno in anni:
        filename = os.path.join(path, f"mcdm_{anno}.csv")
        print(f"\n📂 Elaboro: {filename}")

        try:
            df = pd.read_csv(filename, index_col=0)

            # Calcolo Momentum_6m se manca
            if "Return_6m" in df.columns:
                df = df.dropna(subset=["Return_6m"])
                df["Momentum_6m"] = df["Return_6m"].rank(pct=True)
            else:
                print(f"⚠️ 'Return_6m' non presente nel file {anno}, salto...")
                continue

            # Colonne da normalizzare
            colonne_da_normalizzare = [col for col in criteri + ["Momentum_6m"] if col in df.columns]

            # Normalizzazione con QuantileTransformer
            scaler = QuantileTransformer(output_distribution='uniform', random_state=0)
            df_norm = df.copy()
            df_norm[colonne_da_normalizzare] = scaler.fit_transform(df[colonne_da_normalizzare])

            # Salva il file normalizzato
            out_file = os.path.join(path, f"mcdm_{anno}_normalized.csv")
            df_norm.to_csv(out_file)
            print(f"✅ Salvato '{out_file}' ({len(df)} aziende)")

        except Exception as e:
            print(f"❌ Errore con il file {anno}: {e}")

def update_matrix(anni=[2021, 2022, 2023, 2024]):
    matrici = {}
    for anno in anni:
        path = f"dataset_mcdm/mcdm_{anno}_normalized.csv"
        df = pd.read_csv(path, index_col=0)
        matrici[anno] = df[criteri]
    return matrici

def costruisci_portafogli_mcdm(matrix_years, decile=0.1, save_path="portafogli_mcdm"):
    os.makedirs(save_path, exist_ok=True)

    metodi = {
        "topsis": TOPSIS(),
        "vikor": VIKOR(v=0.5),
        "promethee": PROMETHEE_II(preference_function='usual')
    }

    portafogli = {}

    for anno, df in matrix_years.items():
        criteri = df.columns.tolist()
        matrix = df.values
        weights = equal_weights(matrix)
        types = [1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1,] # premio redd. e dim., penalizzo risl e multipli troppo alti
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

            # Salva CSV con le aziende del portafoglio
            df_risultati.head(top_n).to_csv(
                os.path.join(save_path, f"portafoglio_{nome}_{anno}.csv"),
                index=False
            )

            print(f"✅ Salvato portafoglio {nome} {anno} con {top_n} aziende")

    return portafogli

def compute_MJ():
    ...

# Esegui lo script
calcola_momentum_e_normalizza()
matrix_years = update_matrix()
portafogli = costruisci_portafogli_mcdm(matrix_years)
