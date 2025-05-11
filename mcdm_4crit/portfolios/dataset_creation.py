import numpy as np
import pandas as pd
import yfinance as yf
import requests
import os
import time
from io import StringIO

def genera_matrice_anno(anno, output_dir="dataset_mcdm"):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"mcdm_{anno}.csv")
    print(f"\n📅 Creo dataset per giugno {anno}...")

    # Ottieni lista S&P500
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(StringIO(requests.get(url).text))
    sp500 = tables[0]
    tickers = [symbol.replace(".", "-") for symbol in sp500["Symbol"].tolist()]

    data = {}
    for i, ticker in enumerate(tickers):
        print(f"\n📥 [{i+1}/{len(tickers)}] Ticker: {ticker}")
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5y")

            if hist.empty or "Close" not in hist.columns:
                print("❌ Nessun dato storico, salto...")
                continue

            # Trova la prima data disponibile a giugno
            giugno = hist[(hist.index.month == 6) & (hist.index.year == anno)]
            if giugno.empty:
                print(f"📭 Nessun dato a giugno {anno}, salto...")
                continue

            ref_date = giugno.index[0]
            close = hist["Close"]

            if ref_date not in close.index:
                print(f"📭 Nessun prezzo esatto per {ref_date.date()}, salto...")
                continue

            idx_ref = close.index.get_loc(ref_date)
            if idx_ref >= 126:
                price_now = close.iloc[idx_ref]
                price_past = close.iloc[idx_ref - 126]
                return_6m = (price_now - price_past) / price_past
            else:
                return_6m = np.nan

            volatility = close.iloc[:idx_ref+1].pct_change().std() * np.sqrt(252)

            try:
                info = stock.info
                data[ticker] = {
                    "MarketCap": info.get("marketCap", np.nan),
                    "PriceToBook": info.get("priceToBook", np.nan),
                    "Beta": info.get("beta", np.nan),
                    "DividendYield": info.get("dividendYield", np.nan),
                    "Return_6m": return_6m,
                    "Momentum_6m": np.nan,  # lo calcoliamo dopo
                    "Volatility": volatility,
                    "BookValue": info.get("bookValue", np.nan),
                    "PE": info.get("trailingPE", np.nan),
                    "PB": info.get("priceToBook", np.nan),
                    "ROE": info.get("returnOnEquity", np.nan),
                    "ROA": info.get("returnOnAssets", np.nan),
                    "DebtToEquity": info.get("debtToEquity", np.nan),
                }
                print("✅ OK")
            except Exception as e:
                print(f"⚠️ Errore info(): {e}")
                continue

            time.sleep(1.2)
        except Exception as e:
            print(f"❌ Errore con {ticker}: {e}")
            continue

    df = pd.DataFrame.from_dict(data, orient="index")

    print(f"\n🔍 Aziende prima del dropna: {len(df)}")
    print(f"🔍 Colonne con NaN:\n{df.isna().sum()}")

    # Colonne essenziali per i tuoi ranking
    criteri_obbligatori = [
        "MarketCap", "PriceToBook", "Volatility", "BookValue",
        "PE", "PB", "ROE", "ROA", "Beta", "Return_6m"
    ]
    df.dropna(subset=criteri_obbligatori, inplace=True)

    if "Return_6m" in df.columns:
        df["Momentum_6m"] = df["Return_6m"].rank(pct=True)

    df.to_csv(filename)
    print(f"\n✅ Salvato '{filename}' con {len(df)} aziende valide.")
    return df


def genera_matrici_anni(start=2020, end=2024, output_dir="dataset_mcdm"):
    for anno in range(start, end + 1):
        try:
            genera_matrice_anno(anno, output_dir=output_dir)
        except Exception as e:
            print(f"❌ Errore durante generazione {anno}: {e}")

genera_matrici_anni()