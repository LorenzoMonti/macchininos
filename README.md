# MACCHININOS

Ecco il README aggiornato, con la rimozione della sezione SCP e l'aggiunta di una guida dettagliata per interpretare i risultati nei file CSV generati su Leonardo.

---

# Funnel Architecture for Variable Star Classification 🌟

Questo progetto implementa una pipeline di classificazione gerarchica "a imbuto" (Funnel) per identificare stelle variabili rare, filtrando al contempo il rumore (contaminanti) e le classi comuni su larga scala (milioni di sorgenti).

## 🏗 Architettura del Sistema

Il sistema è diviso in tre stadi logici:

1.  **Anomaly Filter:** Autoencoder (Keras) o Isolation Forest (Sklearn) per scartare oggetti che non somigliano ai dati di training (rumore/contaminanti).
2.  **Gatekeeper:** Balanced Random Forest che decide se un oggetto è un sospetto "raro" o se appartiene alle classi comuni.
3.  **Specialist & Generalist:** 
    *   **Specialist (TabPFN):** Un Transformer su GPU che analizza i sospetti rari.
    *   **Generalist (XGBoost):** Classificatore robusto per le classi comuni (es. RRab vs RRc).

---

## 🚀 Guida all'Esecuzione

### 1. Training (Locale o Leonardo)
Il file `main.py` addestra l'architettura, confronta i filtri e salva il sistema in `saved_models`.
Lo script `launch_train.sh` lancia in background il file `main.py`.

```bash
python3 main.py # locale
sbatch launch_train.sh # Leonardo
```

### 2. Valutazione Massiva su Leonardo (4 GPU)
Per processare file di grandi dimensioni (es. `filtrato_fmem1.csv`), usa lo script `evaluation_cluster.py` tramite SLURM.

```bash
sbatch launch_multi_gpu.sh
```

---

## 📊 Interpretazione dei Risultati (CSV Output)

I file CSV generati in `evaluation/` hanno la seguente struttura:
`id, status, pred_label, confidence, stage, filter_score`

### Esempi e Spiegazione Colonne:

| Esempio Riga | ID Sorgente | Status | Label | Confidenza | Stage (Percorso) | Filter Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `625266...` | **OK** | RRab | 1.0 | **Generalist_Direct** | 3.9e-05 |
| **2** | `221966...` | **ANOMALY** | None | 0.0 | **Filter** | 0.014739 |
| **3** | `680539...` | **OK** | RRc | 1.0 | **Rejected_by_Spec** | 1.4e-05 |
| **4** | `465449...` | **OK** | RRd | 0.948 | **Confirmed_Rare** | 3.8e-05 |

### Legenda delle Colonne:

1.  **ID**: L'identificativo univoco della sorgente (`sourceid`).
2.  **Status**: 
    *   `OK`: Predizione completata con successo.
    *   `ANOMALY`: Bloccata dal filtro iniziale (non è una variabile nota).
    *   `UNCERTAIN`: Predizione completata ma con confidenza bassa (< 0.5).
3.  **Pred_Label**: La classe assegnata (RRab, RRc, RRd, None). È `None` se lo status è ANOMALY.
4.  **Confidence**: Probabilità della predizione (da 0.0 a 1.0).
5.  **Stage**: Indica quale parte della pipeline ha emesso il verdetto:
    *   `Filter`: Bloccata dal filtro anomalie.
    *   `Generalist_Direct`: Il Gatekeeper ha capito subito che era una classe comune.
    *   `Confirmed_Rare`: Lo Specialist (TabPFN) ha confermato che si tratta di una classe rara.
    *   `Rejected_by_Spec`: Lo Specialist ha ritenuto che il sospetto raro fosse un falso allarme e lo ha rimandato al Generalista.
6.  **Filter_Score**: L'errore di ricostruzione (MSE) dell'Autoencoder. Più è alto, più l'oggetto è "strano" rispetto al training set.

---

## 🔬 Esempi d'Uso del Codice (Python)

### Caso A: Valutazione Batch (Test Set)
Usa questo per calcolare le metriche su un dataset di validazione.

```python
from utils.util import load_hierarchical_system, predict_hierarchical_batch

system = load_hierarchical_system("saved_models", "Funnel_Autoencoder_TabPFN")
y_pred, y_conf, y_stage = predict_hierarchical_batch(X_test_scaled, system['models_dict'])
```

### Caso B: Predizione Singola (Inference)
Usa questo per analizzare un singolo oggetto e vedere il suo percorso nel funnel.

```python
from utils.util import load_hierarchical_system, predict_hierarchical_single

res = predict_hierarchical_single(sample_scaled, system['models_dict'])
print(f"Predizione: {res['class']} via {res['stage']} (Conf: {res['confidence']})")
```

---

## 🛠 Note per il Cluster
*   **Sharding**: Ogni GPU scrive un file separato (`results_shard_X.csv`).
*   **Unione Risultati**: Per unire i file prodotti dalle 4 GPU in uno solo, usa:
    ```bash
    cat evaluation/results_shard_*.csv > evaluation/final_all_results.csv
    ```
*   **Requisiti**: Assicurati che i moduli `cineca-ai` siano caricati correttamente per sfruttare le librerie PyTorch/TensorFlow ottimizzate.