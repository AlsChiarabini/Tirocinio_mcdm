import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import os
from pymcdm.methods import PROMETHEE_II

# STEP 1: Carica dati
filename = "sp500_data.csv"
if not os.path.exists(filename):
    raise FileNotFoundError(f"❌ File {filename} non trovato. Assicurati di averlo generato prima.")

df = pd.read_csv(filename, index_col=0)
print("Matrice di valutazione (prime 10 aziende):")
print(df.head(10))
print("\n")

# STEP 2: Definisco pesi e tipi di criterio, normalizzo la matrice
weights = np.array([0.1, 0.3, 0.4, 0.2])
criteria_types = [1, 1, -1, 1]

matrix = df.values
norm_matrix = df / np.linalg.norm(df, axis=0)  # Normalizzazione L2
norm_df = pd.DataFrame(norm_matrix, index=df.index, columns=df.columns)
print("Matrice normalizzata (prime 10 aziende):")
print(norm_df.head(10))
print("\n")

# --- STEP 1: Calcolo PROMETHEE II ---
def calcola_promethee(norm_df, weights, criteria_types):
    promethee = PROMETHEE_II(preference_function='usual')  # NON normalizziamo dentro
    matrix = norm_df.values
    phi_scores = promethee(matrix, weights, criteria_types)
    
    results = pd.DataFrame({
        'PROMETHEE Score': phi_scores
    }, index=norm_df.index)
    
    results = results.sort_values(by='PROMETHEE Score', ascending=False)
    return results

# --- STEP 2: Parte grafica generica ---
def parte_grafica(norm_df, results, score_column, metodo_nome):
    plt.figure(figsize=(14, 6))
    results[score_column].head(20).plot(kind='bar', color='deepskyblue')
    plt.title(f'Top 20 aziende secondo {metodo_nome}')
    plt.ylabel(f'Punteggio {metodo_nome}')
    plt.xlabel('Aziende')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'grafico_{metodo_nome.lower()}_barre.png')
    plt.show()

    # --- Radar ---
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

    colors = plt.colormaps.get_cmap('tab10')

    top_companies = results.head(4).index
    for i, company in enumerate(top_companies):
        values = norm_df.loc[company].values.flatten().tolist()
        values += values[:1]
        ax.plot(np.append(theta, theta[0]), values, color=colors(i % 10), label=company)
        ax.fill(np.append(theta, theta[0]), values, color=colors(i % 10), alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.title(f'Performance delle aziende sui criteri ({metodo_nome})')
    plt.tight_layout()
    plt.savefig(f'grafico_{metodo_nome.lower()}_radar.png')
    plt.show()

# --- Calcolo PROMETHEE II ---
results_promethee = calcola_promethee(norm_df, weights, criteria_types)

# --- Visualizzazione grafica ---
parte_grafica(norm_df, results_promethee, "PROMETHEE Score", "PROMETHEE II")

