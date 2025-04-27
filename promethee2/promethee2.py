import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import os

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

# STEP 3: Funzione PROMETHEE II
def promethee_2(matrix, weights, criteria_types):
    n, m = matrix.shape
    preference_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                preference = 0
                for k in range(m):
                    diff = matrix[i, k] - matrix[j, k]
                    if criteria_types[k] == 1:
                        pref = 1 if diff > 0 else 0
                    else:
                        pref = 1 if diff < 0 else 0
                    preference += weights[k] * pref
                preference_matrix[i, j] = preference

    leaving_flows = preference_matrix.sum(axis=1) / (n - 1)
    entering_flows = preference_matrix.sum(axis=0) / (n - 1)
    net_flows = leaving_flows - entering_flows
    return net_flows

# STEP 4: Calcolo punteggi PROMETHEE II
phi = promethee_2(norm_matrix.values, weights, criteria_types)


# Ordino i risultati
results = pd.DataFrame({
    "PROMETHEE II Score": phi
}, index=df.index)
results = results.sort_values(by="PROMETHEE II Score", ascending=False)

print("Classifica PROMETHEE II:")
print(results.head(20))
print("\n")

# STEP 5: Visualizzazione dei risultati
def parte_grafica_promethee(norm_df, results):
    # --- Grafico barre ---
    plt.figure(figsize=(14, 6))
    results['PROMETHEE II Score'].head(20).plot(kind='bar', color='skyblue')
    plt.title('Top 20 aziende secondo PROMETHEE II')
    plt.ylabel('Punteggio Φ (PROMETHEE II)')
    plt.xlabel('Aziende')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('grafico_promethee_barre.png')
    plt.show()

    # --- Grafico radar ---
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

    colors = plt.get_cmap('tab10')

    top_companies = results.head(4).index
    for i, company in enumerate(top_companies):
        values = norm_df.loc[company].values.flatten().tolist()
        values += values[:1]
        ax.plot(np.append(theta, theta[0]), values, color=colors(i), label=company)
        ax.fill(np.append(theta, theta[0]), values, color=colors(i), alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.title('Performance delle aziende sui criteri (normalizzati) - PROMETHEE II')
    plt.tight_layout()
    plt.savefig('grafico_promethee_radar.png')
    plt.show()

# Chiamo la funzione di grafica
parte_grafica_promethee(norm_df, results)
