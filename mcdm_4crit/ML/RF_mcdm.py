import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from io import StringIO
import requests

PRED_FILENAME = "sp500_pred.csv"

# --- 1. Costruisci dataset storico time-series (per training) ---
def crea_dataset_storico(tickers, period='2y'):
    dataset = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if len(hist) < 252:
                print(f"Non abbiamo abbastanza dati per l'azienda {ticker}")
                continue
            info = stock.info
            for i in range(len(hist) - 126):
                current = hist.iloc[i]
                future_close = hist.iloc[i + 126]["Close"]
                row = {
                    "PE": info.get("trailingPE", np.nan),
                    "PB": info.get("priceToBook", np.nan),
                    "ROE": info.get("returnOnEquity", np.nan),
                    "ROA": info.get("returnOnAssets", np.nan),
                    "DebtToEquity": info.get("debtToEquity", np.nan),
                    "Beta": info.get("beta", np.nan),
                    "MarketCap": info.get("marketCap", np.nan),
                    "DividendYield": info.get("dividendYield", np.nan),
                    "Volatility": abs(hist["Close"].iloc[max(0, i-126):i].pct_change(fill_method=None).std() * np.sqrt(252)),
                    "Momentum_6m": (current["Close"] / hist.iloc[max(0, i-126)]["Close"]) - 1 if i >= 126 else np.nan,
                    "Return_6m": (future_close / current["Close"]) - 1
                }
                dataset.append(row)
        except Exception:
            print("Errore di qualche tipo, controllare il codice")
            continue

    df = pd.DataFrame(dataset)
    df.dropna(inplace=True)
    return df

# --- 2. Preprocessing ---
def prepara_dati(df):
    X = df.drop(columns=["Return_6m"])
    y = df["Return_6m"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler

# --- 3. Ottimizzazione RF ---
def ottimizza_rf(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    random_search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_

    param_grid = {
        'n_estimators': [best_params['n_estimators']],
        'max_depth': [best_params['max_depth']],
        'min_samples_split': [best_params['min_samples_split']],
        'min_samples_leaf': [best_params['min_samples_leaf']],
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# --- 4. Predici ritorni futuri per tutte le aziende oggi ---
def predici_oggi(model, scaler, tickers):
    pred_data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='6mo')
            if hist.empty:
                continue
            info = stock.info
            row = {
                "ticker": ticker,
                "PE": info.get("trailingPE", np.nan),
                "PB": info.get("priceToBook", np.nan),
                "ROE": info.get("returnOnEquity", np.nan),
                "ROA": info.get("returnOnAssets", np.nan),
                "DebtToEquity": info.get("debtToEquity", np.nan),
                "Beta": info.get("beta", np.nan),
                "MarketCap": info.get("marketCap", np.nan),
                "DividendYield": info.get("dividendYield", np.nan),
                "Volatility": abs(hist["Close"].pct_change(fill_method=None).std() * np.sqrt(252)),
                "Momentum_6m": (hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1 if len(hist) > 1 else np.nan,
            }
            pred_data.append(row)
        except:
            continue

    df_pred = pd.DataFrame(pred_data)
    df_pred.dropna(inplace=True)
    X_today = scaler.transform(df_pred.drop(columns=["ticker"]))
    df_pred["Return_6m_Pred"] = model.predict(X_today) # funzione in scikit-learn, ereditata da RFRegressor
    df_pred.set_index("ticker", inplace=True)
    return df_pred

# --- 5. Grafico top aziende ---
def grafico_top20(df_pred):
    top20 = df_pred.sort_values("Return_6m_Pred", ascending=False).head(20)
    plt.figure(figsize=(14, 6))
    top20["Return_6m_Pred"].plot(kind="bar", color="mediumseagreen")
    plt.title("Top 20 aziende previste con miglior Return_6m_Pred")
    plt.ylabel("Return_6m previsto")
    plt.xlabel("Ticker")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("top20_return_pred.png")
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    if os.path.exists(PRED_FILENAME):
        print(f"*** File '{PRED_FILENAME}' trovato, salto training. ***")
        df_pred = pd.read_csv(PRED_FILENAME, index_col=0)
    else:
        print("*** Scarico tickers S&P500 ***")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500 = pd.read_html(StringIO(requests.get(url).text))[0]
        tickers = [t.replace(".", "-") for t in sp500["Symbol"].tolist()[:503]]

        print("*** Creo dataset storico ***")
        df_train = crea_dataset_storico(tickers)
        (X_train, X_test, y_train, y_test), scaler = prepara_dati(df_train)

        print("*** Ottimizzo Random Forest ***")
        model = ottimizza_rf(X_train, y_train)

        print("*** Valutazione su test set: ***")
        y_pred = model.predict(X_test)
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"R²: {r2_score(y_test, y_pred):.4f}")

        print("*** Predizione Return_6m da oggi... ***")
        df_pred = predici_oggi(model, scaler, tickers)

        print(f"*** Salvo file '{PRED_FILENAME} ***")
        df_pred.to_csv(PRED_FILENAME)

    grafico_top20(df_pred)
