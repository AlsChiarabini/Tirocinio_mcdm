import yfinance as yf
import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from io import StringIO
import requests
import os

filename = "sp500_ML.csv"

def importaDataset(filename):
    if os.path.exists(filename):
        print(f"File {filename} trovato, non lo scarico di nuovo.")
        return pd.read_csv(filename, index_col=0)
    else:
        print("\U0001F504 File non trovato, scarico i dati...")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(StringIO(requests.get(url).text))
        sp500 = tables[0]
        tickers = [symbol.replace(".", "-") for symbol in sp500["Symbol"].tolist()]
        valid_tickers = []
        data = {}

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='1y') # Dati storici ad un anno
                if hist.empty:
                    print(f"⚠️ No data for {ticker}, skipping...")
                    continue
                info = stock.info
                data[ticker] = {
                "PE": info.get("trailingPE", np.nan),
                "PB": info.get("priceToBook", np.nan),
                "ROE": info.get("returnOnEquity", np.nan),
                "ROA": info.get("returnOnAssets", np.nan),
                "DebtToEquity": info.get("debtToEquity", np.nan),
                "Beta": info.get("beta", np.nan),
                "MarketCap": info.get("marketCap", np.nan),
                "DividendYield": info.get("dividendYield", np.nan),
                "52WeekChange": info.get("52WeekChange", np.nan),
                "Momentum_6m": hist["Close"].pct_change(min(len(hist), 126)).iloc[-1] if len(hist) > 126 else np.nan,
                "Volatility": hist["Close"].pct_change().std() * (252 ** 0.5),
                "Return_6m": hist["Close"].pct_change(min(len(hist), 126)).iloc[-1] if len(hist) > 126 else np.nan,
            }
                valid_tickers.append(ticker)
            except Exception:
                continue

        df = pd.DataFrame.from_dict(data, orient="index")
        df.dropna(inplace=True) # Rimuovo valori mancanti 
        df.to_csv(filename)
        print(f"\u2705 Dati salvati in '{filename}' per {len(valid_tickers)} aziende.")
        return pd.read_csv(filename, index_col=0)

def preProcessing_data(df):
    X = df.drop(columns=["Return_6m"])  # Tolgo il target ed il resto lo uso come features
    y = df["Return_6m"]  # Nuovo target --> variabile dipendente, obiettivo da prevedere nel modello. I dati del ritorno a 6 mesi non entrano nel training, ma vengono usati per testare il modello
                        # modelli migliori prevedono il futuro senza dati da confrontare

    scaler = StandardScaler() # Creo l'oggetto (da sklearn) che standardizza i dati
    X_scaled = scaler.fit_transform(X) # Fit calcola la media e la deviazione standard per ogni feature e poi li usa per standardizzare i dati
                                    # X_scaled ora contiene la versione standardizzata di X
                                    # X' = (X - media) / stdev
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # 80% train, 20% test

    return X_train, X_test, y_train, y_test

# 3️⃣ Ottimizzazione RandomForest (Random + Grid Search)
def ottimizza_modello(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # Random Search
    random_search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    print("🔍 Migliori parametri (Random Search):", best_params)

    # Grid Search attorno ai migliori parametri
    def safe(val): 
        return max(val, 1)

    param_grid = {
        'n_estimators': list(set([safe(best_params['n_estimators'] - 50), best_params['n_estimators'], best_params['n_estimators'] + 50])),
        'max_depth': [safe(best_params['max_depth'] - 10), best_params['max_depth'], best_params['max_depth'] + 10] if best_params['max_depth'] else [None],
        'min_samples_split': list(set([safe(best_params['min_samples_split']), best_params['min_samples_split'] + 1, best_params['min_samples_split'] + 2])),
        'min_samples_leaf': list(set([safe(best_params['min_samples_leaf'] - 1), best_params['min_samples_leaf'], best_params['min_samples_leaf'] + 1])),
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("✅ Migliori parametri (Grid Search):", grid_search.best_params_)
    return grid_search.best_estimator_

# 4️⃣ Valutazione
def valuta_modello(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n📈 Risultati finali sul test set:\nMAE: {mae:.4f}\nR²: {r2:.4f}")
    return mae, r2


# --- MAIN ---
df = importaDataset(filename)
print("Matrice di valutazione, solo le prime 5 come esempio:")
print(df.head(5))
print('\n')

X_train, X_test, y_train, y_test = preProcessing_data(df)
best_model = ottimizza_modello(X_train, y_train)
valuta_modello(best_model, X_test, y_test)
