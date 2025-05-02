# 📊 Multi-Criteria Decision Making (MCDM) per Portfolio Management

Questo progetto nasce come parte di un tirocinio universitario finalizzato alla realizzazione di una tesi in Scienze dell'informazione. L’obiettivo è studiare, implementare e testare i principali metodi di **MCDM (Multi-Criteria Decision Making)** applicati alla valutazione di portafogli finanziari, con particolare attenzione alla scalabilità in spazi ad alta dimensionalità.

## 🎯 Obiettivi del progetto

- ✅ Studiare e implementare metodi classici MCDM:
- ✅ Valutare le performance dei metodi su dataset reali con **fino a 4 criteri**
- ✅ Dimostrare i limiti di scalabilità dei metodi classici su **10 o più criteri**
- ✅ Applicare un algoritmo MCDM avanzato per gestire l’alta dimensionalità
- ✅ Confrontare prestazioni, classifiche e stabilità tra metodi

## 🧠 Metodi MCDM Implementati

| Metodo     | Descrizione Breve                                    | Stato |
|------------|------------------------------------------------------|-------|
| **TOPSIS** | Scelta dell'alternativa più vicina alla soluzione ideale | ✅ |
| **PROMETHEE II**| Qualsiasi differenza è preferenza totale | ✅ |
| **VIKOR**  | Approccio di compromesso tra criteri conflittuali      | ✅ |
| **MIT-MCDM** | Algoritmo avanzato per gestire oltre 10 criteri     | 🔜 |

---

## 🧪 Dataset utilizzati

- 🔸 Dataset reali da Yahoo Finance e fonti pubbliche
- 🔸 Feature usate: MarketCap, Momentum_6m, Return_6m, Volatilità, ecc.
- 🔸 Dataset sintetici per test controllati (500 aziende × 4 o 10 criteri)

---

## 🧮 Modello ML (facoltativo)

Nel progetto è incluso anche un modulo per stimare il rendimento a 6 mesi delle aziende, tramite:
- Regressione Random Forest ottimizzato
- Valutazione tramite MAE e R²
- Uso del valore stimato come input per il MCDM

---

## 📊 Output principali

- Tabelle di ranking per ciascun metodo
- Grafici comparativi tra TOPSIS, PROMETHEE_II, VIKOR e MIT-MCDM
- Benchmark su tempo di esecuzione e coerenza dei risultati
- Report tecnici e documentazione tesi

---

## 🧰 Librerie principali usate

- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
- `yfinance`, `openpyxl`

---

## 📚 Risorse e ispirazioni

- Patari et al. (2020) – _Sustainable investing via MCDM_
- Ricerca su MCDM in ambienti ad alta dimensionalità
- Appunti e materiale sviluppato durante il tirocinio universitario

---

## 👨‍🎓 Autore

**Alessandro Chiarabini**  
Laureando in Informatica – Università degli studi di Modena e Reggio Emilia.  
> Questo progetto è parte della mia tesi in ambito Scienze dell'informazione.

---

## 📝 Licenza

Questo progetto è a scopo accademico. 



