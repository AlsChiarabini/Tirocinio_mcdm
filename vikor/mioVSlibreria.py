import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymcdm.methods import VIKOR
from pymcdm.visuals import boxplot

# STEP 1: Creo una matrice di valutazione --> dataframe
data = {
    "AZIENDA": ["Az.A", "Az.B", "Az.C", "Az.D"],
    "Prezzo (C1)": [100, 120, 110, 95],
    "Qualità (C2)": [80, 70, 90, 83],
    "Affidabilità (C3)": [70, 60, 80, 75],
    "Velocità (C4)": [4, 3, 5, 7]
}
df = pd.DataFrame(data)
df.set_index("AZIENDA", inplace=True) # Uso azienda come indice della tabella
print("Matrice di valutazione:")
print(df)
print("\n")

# STEP 2: Definire pesi e tipi di criteri
weights = np.array([0.3, 0.2, 0.3, 0.2])  # Pesi per ogni criterio
criteria_types = [-1, 1, 1, -1]  # 1 per criteri da massimizzare, -1 per criteri da minimizzare

# STEP 3: Identificare i valori ideali e anti-ideali per ogni criterio
ideal = []
anti_ideal = []

for i, col in enumerate(df.columns):
    if criteria_types[i] == 1:  # Criterio da massimizzare
        ideal.append(df[col].max())  # Massimo è ideale
        anti_ideal.append(df[col].min())  # Minimo è anti-ideale
    else:  # Criterio da minimizzare
        ideal.append(df[col].min())  # Minimo è ideale
        anti_ideal.append(df[col].max())  # Massimo è anti-ideale

print("Valori ideali:", ideal)
print("Valori anti-ideali:", anti_ideal)
print("\n")

# STEP 4: Calcolare l'utilità e il rimpianto per ogni alternativa
S = []  # S(i) = somma pesata delle distanze tra il valore dell'alternativa e l'ideale --> è il comportamento globale
R = []  # Rimpianto individuale (massima distanza pesata dall'ideale)

# Si = sommatoria(w_j * [(f_i - f*_j) / (f*_j - f**_j)]), dove f* = ideale, f** = anti-ideale, w_j = peso del criterio j
# Ri = max(w_j * [(f_i - f*_j) / (f*_j - f**_j)])

for i, alt in enumerate(df.index): # enumerate restituisce due valori: i (indice) e alt (nome dell'alternativa)
    s_i = 0  # Utilità per l'alternativa i
    r_i = 0  # Rimpianto per l'alternativa i
    
    for j, col in enumerate(df.columns): # j per accedere ai vettori (pesi, id, anti-id), col nome del criterio 
        # Calcola la distanza normalizzata dall'ideale
        if ideal[j] != anti_ideal[j]:  # Evita divisione per zero
            normalized_dist = weights[j] * (abs(ideal[j] - df.loc[alt, col]) / abs(ideal[j] - anti_ideal[j]))
        else:
            normalized_dist = 0
            
        # Aggiorna utilità
        s_i += normalized_dist
        
        # Aggiorna rimpianto (massima distanza)
        r_i = max(r_i, normalized_dist)
    
    S.append(s_i)
    R.append(r_i)

# STEP 5: Calcolare i valori di VIKOR (Q)
# Prima normalizzare S e R
S_star = min(S)  # Minimo S (migliore utilità)
S_minus = max(S)  # Massimo S (peggiore utilità)
R_star = min(R)  # Minimo R (miglior rimpianto)
R_minus = max(R)  # Massimo R (peggior rimpianto)

# Parametro v di compromesso (tipicamente 0.5)
v = 0.5

# Calcolo dei valori Q per ogni alternativa
Q = [] # Q è il punteggio di VIKOR per ogni alternativa, cioè il compromesso tra S e R
# Q = v*(S - S*)/(S- - S*) + (1-v)*(R - R*)/(R- - R*), dove v = 0.5 per bilanciare
for i in range(len(df.index)):
    # Formula VIKOR: Q = v*(S - S*)/(S- - S*) + (1-v)*(R - R*)/(R- - R*)
    if S_minus != S_star: # Sempre per evitare divisione per zero
        term1 = v * (S[i] - S_star) / (S_minus - S_star)
    else:
        term1 = 0
        
    if R_minus != R_star:
        term2 = (1 - v) * (R[i] - R_star) / (R_minus - R_star)
    else:
        term2 = 0
        
    Q.append(term1 + term2)

# STEP 6: Ordinare le alternative in base a S, R e Q
results = pd.DataFrame({
    "S (Utilità)": S,
    "R (Rimpianto)": R,
    "Q (VIKOR)": Q
}, index=df.index)

# Ordina in base a Q (valori più bassi sono migliori)
results = results.sort_values(by="Q (VIKOR)")
print("Classifica VIKOR manuale:")
print(results)
print("\n")

# STEP 7: Verifica delle condizioni di accettabilità
# Condizione 1: Vantaggio accettabile
sorted_alternatives = results.sort_values(by="Q (VIKOR)").index
best_alt = sorted_alternatives[0]
second_best = sorted_alternatives[1]
threshold = 1 / (len(df.index) - 1) # Soglia di accettabilità, quindi 1/(n-1)

advantage = results.loc[second_best, "Q (VIKOR)"] - results.loc[best_alt, "Q (VIKOR)"]
print(f"Vantaggio accettabile: {advantage}, soglia: {threshold}")
if advantage < threshold:
    print("Vantaggio non accettabile, considerare alternative diverse.") # Nel caso, ottimizza pesi, modificare v
else:
    print("Vantaggio accettabile, procedere con l'alternativa migliore.")
print("\n")

# Condizione 2: Stabilità decisionale
best_in_S = results["S (Utilità)"].idxmin()
best_in_R = results["R (Rimpianto)"].idxmin()
print(f"Migliore alternativa in S: {best_in_S}")
print(f"Migliore alternativa in R: {best_in_R}")
print(f"Migliore alternativa in Q: {best_alt}")
print(f"Stabilità decisionale: {best_alt == best_in_S or best_alt == best_in_R}")
print("\n")

# STEP 8: Implementazione con PyMCDM e confronto
vikor = VIKOR(normalization_function=None)

# Matrice senza l'indice "AZIENDA" (solo i dati numerici)
matrix_pymcdm = df.values.astype(float)

# Calcolo dei punteggi usando pymcdm
q_pymcdm = vikor(matrix_pymcdm, weights, criteria_types)

# Il metodo VIKOR restituisce direttamente i punteggi Q, se volessimo S e R dovremmo calcolarli a mano
# Non lo faccio perché non è il punto del codice

# Aggiungi solo i punteggi Q calcolati da pymcdm nel dataframe originale
df["Q (VIKOR pymcdm)"] = q_pymcdm

# Confronto finale
print("Confronto tra punteggi manuali e pymcdm:")
comparison = pd.DataFrame({
    'Q Manual': Q,
    'Q PyMCDM': q_pymcdm,
    'Diff Q': np.abs(np.array(Q) - q_pymcdm)
}, index=df.index)
print(comparison)
print("\n")

# STEP 9: Visualizzazione grafica dei risultati
# Grafico a barre per confrontare i punteggi Q di VIKOR
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(df.index))

plt.bar(index, Q, bar_width, label='Q Manuale')
plt.bar(index + bar_width, q_pymcdm, bar_width, label='Q PyMCDM')

plt.xlabel('Aziende')
plt.ylabel('Punteggio Q VIKOR')
plt.title('Confronto Punteggi Q VIKOR')
plt.xticks(index + bar_width/2, df.index)
plt.legend()
plt.tight_layout()
plt.savefig('vikor_comparison.png')
plt.show()

# Grafico radar per visualizzare S e R (solo implementazione manuale)
categories = ['S (Utilità)', 'R (Rimpianto)']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Chiudi il cerchio

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Normalizza S e R per il grafico radar
S_norm = [(s - min(S)) / (max(S) - min(S)) for s in S]
R_norm = [(r - min(R)) / (max(R) - min(R)) for r in R]

for i, alt in enumerate(df.index):
    values = [S_norm[i], R_norm[i]]
    values += values[:1]  # Chiudi il cerchio
    ax.plot(angles, values, linewidth=2, label=alt)
    ax.fill(angles, values, alpha=0.1)

plt.xticks(angles[:-1], categories)
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
plt.ylim(0, 1)

plt.legend(loc='upper right')
plt.title('Visualizzazione di S e R per ogni alternativa')
plt.tight_layout()
plt.savefig('vikor_radar.png')
plt.show()