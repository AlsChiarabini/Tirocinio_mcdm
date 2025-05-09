import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from io import StringIO
import requests
from time import sleep

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(StringIO(requests.get(url).text))
    sp500 = tables[0]
    tickers = [t.replace(".", "-") for t in sp500["Symbol"].tolist()]
    return tickers

def scarica_dati_storici(tickers, start="2020-01-01", end="2025-01-01", cartella_output="dati_prezzi"):
    os.makedirs(cartella_output, exist_ok=True)
    sleep(3)

    for ticker in tickers:
        filepath = os.path.join(cartella_output, f"{ticker}.csv")

        if os.path.exists(filepath):
            print(f"✅ File già presente per {ticker}, salto.")
            continue

        try:
            print(f"📥 Scarico dati per {ticker}...")
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start, end=end)

            if hist.empty or len(hist) < 252:
                print(f"⚠️ Dati insufficienti per {ticker}, salto.")
                continue

            hist.reset_index(inplace=True)
            hist.to_csv(filepath, index=False)
            print(f"✅ Salvato {ticker} ({len(hist)} righe)")
            sleep(1)

        except Exception as e:
            print(f"❌ Errore con {ticker}: {e}")
            continue

    print("🏁 Download completato.")

def get_static_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "PE": info.get("trailingPE", np.nan),
            "PB": info.get("priceToBook", np.nan),
            "ROE": info.get("returnOnEquity", np.nan),
            "ROA": info.get("returnOnAssets", np.nan),
            "DebtToEquity": info.get("debtToEquity", np.nan),
            "Beta": info.get("beta", np.nan),
            "MarketCap": info.get("marketCap", np.nan),
            "DividendYield": info.get("dividendYield", np.nan)
        }
    except Exception as e:
        print(f"Errore fondamentali {ticker}: {e}")
        return {key: np.nan for key in ["PE", "PB", "ROE", "ROA", "DebtToEquity", "Beta", "MarketCap", "DividendYield"]}

def crea_snapshot_annuali(tickers, cartella_prezzi="dati_prezzi", anni=range(2020, 2025), mesi=6):
    dataset_completo = []

    for anno in anni:
        print(f"\n📅 Elaborazione snapshot per anno {anno}...")

        data_riferimento = datetime(anno, mesi, 1)
        data_inizio_vol = data_riferimento - relativedelta(months=6)
        data_fine_return = data_riferimento + relativedelta(months=6)

        for ticker in tickers:
            try:
                path = os.path.join(cartella_prezzi, f"{ticker}.csv")
                if not os.path.exists(path):
                    print(f"❌ File dati mancante per {ticker}, salto.")
                    continue

                df = pd.read_csv(path, parse_dates=["Date"])
                df = df.set_index("Date")

                if data_riferimento not in df.index:
                    idx = df.index.get_indexer([data_riferimento], method='bfill')
                    if idx[0] == -1:
                        continue
                    data_rif = df.index[idx[0]]
                else:
                    data_rif = data_riferimento

                prezzo_corrente = df.loc[data_rif]["Close"]
                df_vol = df.loc[data_inizio_vol:data_rif]
                df_ret = df.loc[data_rif:data_fine_return]

                if len(df_vol) < 60 or len(df_ret) < 60:
                    continue

                volatility = df_vol["Close"].pct_change().std() * np.sqrt(252)
                momentum = (df.loc[data_rif]["Close"] / df_vol.iloc[0]["Close"]) - 1
                return_6m = (df_ret.iloc[-1]["Close"] / prezzo_corrente) - 1
                sharpe = return_6m / volatility if volatility != 0 else np.nan

                fondamentali = get_static_fundamentals(ticker)

                row = {
                    "Ticker": ticker,
                    "Anno": anno,
                    "Data": data_rif.date(),
                    "Close": prezzo_corrente,
                    "Volatility": volatility,
                    "Momentum_6m": momentum,
                    "Return_6m": return_6m,
                    "Sharpe_6m": sharpe,
                    **fondamentali
                }

                dataset_completo.append(row)

            except Exception as e:
                print(f"⚠️ Errore con {ticker}: {e}")
                continue

    final_df = pd.DataFrame(dataset_completo)
    final_df.to_csv("dataset_snapshot_completo.csv", index=False)
    print("✅ Dataset finale salvato come 'dataset_snapshot_completo.csv'")
    return final_df

# --- ESECUZIONE COMPLETA ---
tickers = get_sp500_tickers()
scarica_dati_storici(tickers)
df_snapshot = crea_snapshot_annuali(tickers)
print(df_snapshot.head())

