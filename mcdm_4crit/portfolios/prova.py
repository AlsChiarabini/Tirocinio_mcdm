import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import QuantileTransformer
from pymcdm.methods import TOPSIS, VIKOR, PROMETHEE_II
from pymcdm.weights import equal_weights
from ordering_MJ import voting_ordering
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

class AdvancedPortfolioManager:
    """
    Portfolio Manager con funzionalità avanzate di selezione e analisi.
    Supporta metodi MCDM multipli, subset di criteri e analisi di performance.
    """
    
    def __init__(self, 
                 data_path: str = "dataset_mcdm", 
                 output_path: str = "portafogli_mcdm", 
                 years: List[int] = None):
        """
        Inizializza il gestore di portafogli avanzato.
        
        Args:
            data_path: Percorso ai dati di input
            output_path: Percorso per salvare i risultati
            years: Anni da analizzare
        """
        self.data_path = data_path
        self.output_path = output_path
        self.years = years or [2021, 2022, 2023, 2024]
        
        # Lista completa dei criteri
        self.criteri = [
            "MarketCap", "PriceToBook", "Beta", "DividendYield",
            "Return_6m", "Momentum_6m", "Volatility", "BookValue",
            "PE", "PB", "ROE", "ROA", "DebtToEquity", "SharpeRatio"
        ]
        
        # Sottoinsieme di criteri per portfolio MCDM
        self.subset_criteri = {
            "1": ["MarketCap", "Beta", "Momentum_6m", "SharpeRatio"],
            "2": ["DividendYield", "Return_6m", "Volatility", "ROE"]
        }
        
        # Crea directory di output
        os.makedirs(self.output_path, exist_ok=True)
        
        # Portafogli generati
        self.portfolios = {}
        self.normalized_data = {}
        self.returns_df = None
        self.performance_metrics = None
    
    def calcola_sharpe_ratio(self) -> None:
        """Calcola Sharpe Ratio per ogni anno."""
        print("\n🧮 Calcolo Sharpe Ratio")
        for anno in self.years:
            filename = os.path.join(self.data_path, f"mcdm_{anno}.csv")
            try:
                df = pd.read_csv(filename, index_col=0)
                if "Return_6m" in df.columns and "Volatility" in df.columns:
                    df = df.dropna(subset=["Return_6m", "Volatility"])
                    df["SharpeRatio"] = df["Return_6m"] / df["Volatility"]
                    df.to_csv(filename)
                    print(f"✅ Aggiornato mcdm_{anno}.csv con Sharpe-ratio per ({len(df)} aziende)")
                else:
                    print(f"⚠️ Colonne 'Return_6m' o 'Volatility' mancanti in {filename}, salto...")
            except Exception as e:
                print(f"❌ Errore con il file {anno}: {e}")
    
    def calcola_momentum(self) -> None:
        """Calcola Momentum per ogni anno."""
        print("\n🚀 Calcolo Momentum")
        for anno in self.years:
            filename = os.path.join(self.data_path, f"mcdm_{anno}.csv")
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
    
    def normalizza_matrici(self) -> None:
        """Normalizza le matrici dei dati per ogni anno."""
        print("\n📊 Normalizzazione Matrici")
        for anno in self.years:
            file = os.path.join(self.data_path, f"mcdm_{anno}.csv")
            df = pd.read_csv(file, index_col=0)
            df = df.dropna(subset=self.criteri)

            scaler = QuantileTransformer(output_distribution='uniform', random_state=0)
            df_norm = df.copy()
            df_norm[self.criteri] = scaler.fit_transform(df[self.criteri])

            out_file = file.replace(".csv", "_normalized.csv")
            df_norm.to_csv(out_file)
            
            # Salva in memoria
            self.normalized_data[anno] = df_norm[self.criteri]
            print(f"✅ Salvato file normalizzato: {out_file}")
    
    def update_matrix(self) -> Dict[int, pd.DataFrame]:
        """Carica le matrici normalizzate."""
        matrici = {}
        for anno in self.years:
            path = os.path.join(self.data_path, f"mcdm_{anno}_normalized.csv")
            df = pd.read_csv(path, index_col=0)
            matrici[anno] = df[self.criteri]
        return matrici
    
    def costruisci_portafogli_mcdm(self, 
                                   matrix_years: Dict[int, pd.DataFrame], 
                                   decile: float = 0.1) -> Dict[Tuple[int, str], List[str]]:
        """
        Costruisce portafogli utilizzando metodi MCDM classici.
        
        Args:
            matrix_years: Dizionario con matrici normalizzate per anno
            decile: Percentuale di aziende da selezionare
        """
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
                print(f"\n📊 Applico {nome.upper()} per anno {anno}...")

                scores = metodo(matrix, weights, types)
                df_risultati = pd.DataFrame({
                    "Ticker": tickers,
                    "Score": scores
                }).sort_values("Score", ascending=False)

                top_n = int(len(df_risultati) * decile)
                top_aziende = df_risultati.head(top_n)["Ticker"].tolist()

                portafogli[(anno, nome)] = top_aziende

                # Salva i risultati
                df_risultati.head(top_n).to_csv(
                    os.path.join(self.output_path, f"portafoglio_{nome}_{anno}.csv"),
                    index=False
                )

                print(f"✅ Salvato portafoglio {nome} {anno} con {top_n} aziende")

        return portafogli
    
    def costruisci_portafogli_MJ(self, decile: float = 0.1) -> Dict[int, List[str]]:
        """
        Costruisce portafogli utilizzando il metodo MJ.
        
        Args:
            decile: Percentuale di aziende da selezionare
        """
        portafogli_mj = {}

        for anno in self.years:
            print(f"\n📂 MJ --> Elaboro anno {anno}")
            file = os.path.join(self.data_path, f"mcdm_{anno}.csv")
            df = pd.read_csv(file, index_col=0)
            df = df.dropna(subset=self.criteri)
            sub_df = df[self.criteri]

            M = sub_df.rank(pct=True)
            M = pd.qcut(M.stack(), q=3, labels=[1, 2, 3]).unstack().astype(int)

            _, _, new_indices, _ = voting_ordering(M.values, disentagle_df=sub_df)

            tickers = sub_df.index[new_indices].tolist()
            top_n = int(len(tickers) * decile)
            top_aziende = tickers[:top_n]

            portafogli_mj[anno] = top_aziende

            pd.DataFrame(top_aziende, columns=["Ticker"]).to_csv(
                os.path.join(self.output_path, f"portafoglio_MJ_{anno}.csv"), 
                index=False
            )

            print(f"✅ Salvato portafoglio MJ {anno} con {top_n} aziende")

        return portafogli_mj
    
    def costruisci_portafogli_mcdm_subset(self, 
                                          matrix_years: Dict[int, pd.DataFrame], 
                                          decile: float = 0.1) -> Dict[Tuple[int, str], List[str]]:
        """
        Costruisce portafogli utilizzando subset di criteri.
        
        Args:
            matrix_years: Dizionario con matrici normalizzate per anno
            decile: Percentuale di aziende da selezionare
        """
        metodi = {
            "topsis": TOPSIS(),
            "vikor": VIKOR(v=0.5),
            "promethee": PROMETHEE_II(preference_function='usual')
        }

        portafogli = {}

        for subset_name, criteri_subset in self.subset_criteri.items():
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
                    print(f"📊 {metodo_nome} - anno {anno}")

                    scores = metodo(matrix, weights, types)
                    df_risultati = pd.DataFrame({"Ticker": tickers, "Score": scores}).sort_values("Score", ascending=False)

                    top_n = int(len(df_risultati) * decile)
                    top_aziende = df_risultati.head(top_n)["Ticker"].tolist()

                    portafogli[(anno, metodo_nome)] = top_aziende

                    df_risultati.head(top_n).to_csv(
                        os.path.join(self.output_path, f"portafoglio_{metodo_nome}_{anno}.csv"), 
                        index=False
                    )

                    print(f"✅ Salvato portafoglio {metodo_nome} {anno} con {top_n} aziende")

        return portafogli
    
    def calcola_rendimenti_portafogli(self, portafogli_dict: Dict[Tuple[int, str], List[str]]) -> pd.DataFrame:
        """
        Calcola i rendimenti dei portafogli.
        
        Args:
            portafogli_dict: Dizionario con i portafogli per anno e metodo
        """
        risultati = []

        for anno in self.years:
            print(f"\n📈 Calcolo rendimenti portafogli per anno {anno}...")
            df = pd.read_csv(os.path.join(self.data_path, f"mcdm_{anno}.csv"), index_col=0)

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
                    print(f"⚠️ Nessun portafoglio trovato per {metodo} {anno}, salto...")
                    continue

                subset = df.loc[df.index.intersection(tickers)]
                subset = subset.dropna(subset=["Return_6m"])
                rendimento = subset["Return_6m"].mean()

                risultati.append((anno, metodo, rendimento))

        # Costruzione DataFrame
        df_risultati = pd.DataFrame(risultati, columns=["Anno", "Metodo", "Rendimento"])
        df_risultati = df_risultati.sort_values(["Anno", "Metodo"]).reset_index(drop=True)
        df_risultati.to_csv(os.path.join(self.output_path, "rendimenti_portafogli.csv"), index=False)

        # Stampa ordinata
        print("\n📊 RENDIMENTI PER METODO (ORDINATI PER ANNO):")
        for anno in df_risultati["Anno"].unique():
            print("\n" + "*" * 40)
            print(f"📅 ANNO: {anno}")
            print("*" * 40)
            print(df_risultati[df_risultati["Anno"] == anno][["Metodo", "Rendimento"]].to_string(index=False))

        self.returns_df = df_risultati
        return df_risultati
    
    def calcola_metriche_performance(self) -> pd.DataFrame:
        """Calcola le metriche di performance dai rendimenti."""
        if self.returns_df is None:
            raise ValueError("Calcolare prima i rendimenti dei portafogli")

        metrics = []

        for metodo, gruppo in self.returns_df.groupby("Metodo"):
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
        file_path = os.path.join(self.output_path, "metriche_performance.csv")
        df_metrics.to_csv(file_path, index=False)
        print(f"📈 Salvato: {file_path}")
        
        self.performance_metrics = df_metrics
        return df_metrics
    
    def plot_crescita_cumulata(self, capitale_iniziale: float = 1.0) -> None:
        """
        Visualizza la crescita cumulata del capitale.
        
        Args:
            capitale_iniziale: Capitale iniziale per la simulazione
        """
        if self.returns_df is None:
            raise ValueError("Calcolare prima i rendimenti dei portafogli")

        df = self.returns_df.copy()
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
        plt.savefig(os.path.join(self.output_path, "crescita_portafogli.png"), dpi=300)
        plt.show()
    
    def run_full_analysis(self, decile: float = 0.1) -> None:
        """
        Esegue l'analisi completa dei portafogli.
        
        Args:
            decile: Percentuale di aziende da selezionare per ogni portafoglio
        """
        # Calcoli preliminari
        self.calcola_sharpe_ratio()
        self.calcola_momentum()
        self.normalizza_matrici()
        
        # Carica le matrici normalizzate
        matrix_years = self.update_matrix()
        
        # Costruzione portafogli
        portafogli_mcdm = self.costruisci_portafogli_mcdm(matrix_years, decile)
        portafogli_MJ = self.costruisci_portafogli_MJ(decile)
        portafogli_mcdm_subset = self.costruisci_portafogli_mcdm_subset(matrix_years, decile)
        
        # Unione portafogli
        portafogli_all = {}
        for (anno, metodo), tickers in portafogli_mcdm.items():
            portafogli_all[(anno, metodo)] = tickers
        for anno, tickers in portafogli_MJ.items():
            portafogli_all[(anno, "MJ")] = tickers
        for (anno, metodo), tickers in portafogli_mcdm_subset.items():
            portafogli_all[(anno, metodo)] = tickers
        
        # Calcolo rendimenti
        self.returns_df = self.calcola_rendimenti_portafogli(portafogli_all)
        
        # Calcolo metriche di performance
        self.performance_metrics = self.calcola_metriche_performance()
        print("\n📊 Metriche di performance:\n")
        print(self.performance_metrics)
        
        # Visualizzazione crescita cumulata
        self.plot_crescita_cumulata()

# Esempio di utilizzo
if __name__ == "__main__":
    # Crea un'istanza del gestore di portafogli avanzato
    pm = AdvancedPortfolioManager(
        data_path="dataset_mcdm",
        output_path="portafogli_mcdm",
        years=[2021, 2022, 2023, 2024]
    )
    
    # Esegui l'analisi completa
    pm.run_full_analysis(decile=0.1)