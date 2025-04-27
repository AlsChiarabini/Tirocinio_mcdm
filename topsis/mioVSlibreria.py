import numpy as np
import pandas as pd
from pymcdm.methods import TOPSIS
from pymcdm.visuals import boxplot
import matplotlib.pyplot as plt

def normalize(df):
    norm_col = df.copy()
    for col in df.columns:
        norm_col[col] = df[col] / np.sqrt(df[col] ** 2).sum()
    return norm_col

# STEP 1: Creo una matrice di valutazione --> dataframe
data = {
    "AZIENDA": ["Az.A", "Az.B", "Az.C", "Az.D"],
    "Prezzo (C1)": [100, 120, 110, 95],
    "Qualità (C2)": [80, 70, 90, 83],
    "Affidabilità (C3)": [70, 60, 80, 75],
    "Velocità (C4)": [4, 3, 5, 7]
}

df = pd.DataFrame(data)
df.set_index("AZIENDA", inplace=True)

print("Matrice di valutazione:")
print(df)
print("\n")

# STEP 2: Normalizzo la matrice di valutazione
df_norm = normalize(df)
print("Matrice normalizzata manualmente:")
print(df_norm)
print("\n")

# STEP 3: Creo la matrice di pesi
weights = np.array([0.3, 0.2, 0.3, 0.2])  # Pesi per ogni criterio
df_weighted = df_norm * weights
print("Matrice pesata manualmente:")
print(df_weighted)
print("\n")

# STEP 4: Creo la matrice di ideal e anti-ideal
criteria_types = [-1, 1, 1, -1]  # 1 per criteri da massimizzare, -1 per criteri da minimizzare
ideal = []
anti_ideal = []
for i, col in enumerate(df_weighted.columns):       # i per criteri, col per colonne
    if criteria_types[i] == 1:
        ideal.append(df_weighted[col].max())        # Per questo criterio, questo è l'ideale
        anti_ideal.append(df_weighted[col].min())   # Per questo criterio, questo è l'anti-ideale
    else:
        ideal.append(df_weighted[col].min())
        anti_ideal.append(df_weighted[col].max())

# STEP 5: Calcolo le distanze
d_plus = np.sqrt(((df_weighted - ideal) ** 2).sum(axis=1))  # distanza ideale, axis=1 vuol dire che calcola la distanza per ogni riga, no ciclo ma lavora in modo vettoriale
d_minus = np.sqrt(((df_weighted - anti_ideal) ** 2).sum(axis=1))  # distanza anti-ideale

scores = d_minus / (d_plus + d_minus)  # calcolo il punteggio di ogni azienda

# STEP 6: Ordino le aziende in base al punteggio
results = pd.DataFrame({                                    # Qua sto proprio creando un nuovo dataframe, una nuova tabella
    "D+": d_plus,
    "D-": d_minus,
    "TOPSIS Score": scores
}, index=df.index)
results = results.sort_values(by="TOPSIS Score", ascending=False)  # ordina in base al punteggio
print("Classifica TOPSIS: ")
print(results)

# STEP 8: Implemento pymcdm e confronto
topsis = TOPSIS()

# Matrice senza l'indice "AZIENDA" (solo i dati numerici)
matrix_pymcdm = df.values.astype(float)

# Calcolo i punteggi usando pymcdm
scores_pymcdm = topsis(matrix_pymcdm, weights, criteria_types)

# Aggiungi i punteggi calcolati da pymcdm nel dataframe
df["TOPSIS Score (pymcdm)"] = scores_pymcdm             # Aggiungo il nome della colonna qui, con il vettore

# Aggiungi i punteggi manuali nel dataframe
df["TOPSIS Score"] = scores                             # Qua aggiungo alla tabella iniziale, le colonne che mi interessano

# Mostra la classifica in base ai punteggi di pymcdm
print("\nClassifica TOPSIS (pymcdm): ")
print(df[["TOPSIS Score (pymcdm)"]].sort_values(by="TOPSIS Score (pymcdm)", ascending=False))

# Confronto finale tra il punteggio manuale e pymcdm
df["Differenza tra punteggi"] = np.abs(df["TOPSIS Score"] - df["TOPSIS Score (pymcdm)"])  # Creo una nuova colonna qui
print("\nDifferenza tra punteggi manuali e pymcdm: ")
print(df[["TOPSIS Score", "TOPSIS Score (pymcdm)", "Differenza tra punteggi"]])             # Stampo solo le colonne che mi interessano dalla tabella iniziale

'''

def parte_grafica ():

    # 1. Grafico a barre per confrontare i punteggi TOPSIS
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(df.index))

    plt.bar(index, df["TOPSIS Score"], bar_width, label='Punteggio Manuale')
    plt.bar(index + bar_width, df["TOPSIS Score (pymcdm)"], bar_width, label='Punteggio PyMCDM')

    plt.xlabel('Aziende')
    plt.ylabel('Punteggio TOPSIS')
    plt.title('Confronto Punteggi TOPSIS')
    plt.xticks(index + bar_width/2, df.index)
    plt.legend()
    plt.tight_layout()
    plt.savefig('topsis_comparison.png')
    plt.show()

    # 2. Grafico radar per visualizzare le performance di ciascuna azienda sui diversi criteri
    from matplotlib.path import Path
    from matplotlib.spines import Spine
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
                return PolarAxes.fill(self, *args, **kwargs)
                
            def plot(self, *args, **kwargs):
                return PolarAxes.plot(self, *args, **kwargs)
                
        register_projection(RadarAxes)
        
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='radar')
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        
        # Qui NON impostiamo xticks vuoti perché vogliamo i nomi dei criteri
        # plt.xticks(theta, [])
        
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
        
        return fig, ax

    # Normalizzare i dati originali per il grafico radar (0-1)
    radar_df = df[["Prezzo (C1)", "Qualità (C2)", "Affidabilità (C3)", "Velocità (C4)"]].copy()
    for col in radar_df.columns:
        if col in ["Prezzo (C1)", "Velocità (C4)"]:  # Criteri da minimizzare
            radar_df[col] = 1 - (radar_df[col] - radar_df[col].min()) / (radar_df[col].max() - radar_df[col].min())
        else:  # Criteri da massimizzare
            radar_df[col] = (radar_df[col] - radar_df[col].min()) / (radar_df[col].max() - radar_df[col].min())

    # Crea il grafico radar
    fig, ax = radar_factory(len(radar_df.columns), frame='polygon')
    colors = ['b', 'g', 'r', 'c']
    theta = np.linspace(0, 2*np.pi, len(radar_df.columns), endpoint=False)

    # Aggiungi etichette per gli assi (nomi dei criteri)
    ax.set_xticks(theta)
    ax.set_xticklabels(radar_df.columns)

    for i, (idx, row) in enumerate(radar_df.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1]  # Chiude il poligono
        ax.plot(np.append(theta, theta[0]), values, color=colors[i], label=idx)
        ax.fill(np.append(theta, theta[0]), values, color=colors[i], alpha=0.1)

    plt.legend(loc='upper right')
    plt.title('Performance delle aziende sui diversi criteri')
    plt.tight_layout()
    plt.savefig('radar_performance.png')
    plt.show()

parte_grafica()
'''