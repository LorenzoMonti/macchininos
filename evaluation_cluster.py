import os
import argparse
import pandas as pd
import numpy as np
import gc
import time
from tqdm import tqdm
import joblib
import tensorflow as tf

# --- CONFIGURAZIONE AMBIENTE ---
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Import delle utility dal tuo file util.py
# Assicurati che predict_hierarchical_batch in util.py sia la versione vettorizzata
from utils.util import load_hierarchical_system, predict_hierarchical_batch

# --- PARAMETRI FISSI ---
MODEL_DIR = "saved_models"
EXP_NAME = "Funnel_Autoencoder_TabPFN" # O il nome della tua cartella modello
NEW_DATA_PATH = "data/filtrato_fmem1.csv"
ID_COLUMN = "sourceid"

TRAINING_PATH_CHECK = "data/campioneLARGE05_allinfo.csv"
CONTAMINANTS_PATH_CHECK = "data/contaminants/contaminants.csv"
PROCESSED_FULL_PATH = "evaluation/processed_full_catalog.csv"

# Chunk molto grande per saturare la GPU (500k-800k righe alla volta)
CHUNK_SIZE = 600000 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, default=0, help="ID del processo SLURM")
    parser.add_argument("--num_shards", type=int, default=4, help="Numero totale di GPU")
    return parser.parse_args()

def load_comprehensive_blacklist():
    blacklist = set()
    for path in [TRAINING_PATH_CHECK, CONTAMINANTS_PATH_CHECK, PROCESSED_FULL_PATH]:
        if path and os.path.exists(path):
            try:
                # Carichiamo solo la colonna sourceid
                df = pd.read_csv(path, usecols=[ID_COLUMN], dtype={ID_COLUMN: str})
                blacklist.update(df[ID_COLUMN].values)
                print(f"   Aggiunti {len(df)} ID da {path}")
            except Exception as e:
                print(f"   Errore nel caricamento di {path}: {e}")
    return blacklist

def run_pipeline():
    args = get_args()
    worker_id = args.shard_id
    total_workers = args.num_shards
    
    prefix = f"[GPU-{worker_id}]"
    print(f"{prefix} Inizializzazione...")

    # 1. Caricamento Sistema
    try:
        system = load_hierarchical_system(MODEL_DIR, EXP_NAME)
        scaler = system['scaler']
        filter_model = system['filter_model']
        filter_type = system['filter_type']
        ae_threshold = system['ae_threshold']
        models_dict = system['models_dict']
        le = system.get('le', None)
        features = system['features']
    except Exception as e:
        print(f"{prefix} ❌ Errore caricamento: {e}")
        return

    # 2. Setup Output
    output_dir = "evaluation"
    os.makedirs(output_dir, exist_ok=True)
    MY_OUTPUT_PATH = os.path.join(output_dir, f"results_{EXP_NAME}_shard_{worker_id}.csv")
    
    # Header del CSV
    csv_header = ["id", "status", "pred_label", "confidence", "stage", "filter_score"]
    with open(MY_OUTPUT_PATH, 'w') as f:
        f.write(",".join(csv_header) + "\n")

    # 3. Caricamento Blacklist
    blacklist_ids = load_comprehensive_blacklist()
    print(f"{prefix} Blacklist caricata: {len(blacklist_ids)} sorgenti ignorate.")

    # 4. Processamento a Chunk
    if not os.path.exists(NEW_DATA_PATH):
        print(f"{prefix} ❌ File non trovato: {NEW_DATA_PATH}")
        return

    cols_to_load = list(set([ID_COLUMN] + features))
    # Il reader caricherà i dati a blocchi
    reader = pd.read_csv(NEW_DATA_PATH, usecols=cols_to_load, chunksize=CHUNK_SIZE)
    
    start_time = time.time()
    processed_count = 0

    print(f"{prefix} Inizio elaborazione batch...")

    for chunk_idx, chunk in enumerate(reader):
        # --- LOGICA DI SHARDING ---
        # Ogni GPU processa solo i chunk che le competono (es: 0, 4, 8... la GPU 0)
        if chunk_idx % total_workers != worker_id:
            continue
        
        # A. Rimozione Blacklist
        chunk[ID_COLUMN] = chunk[ID_COLUMN].astype(str)
        chunk = chunk[~chunk[ID_COLUMN].isin(blacklist_ids)]
        if chunk.empty: continue

        # B. Pulizia Dati (NaN/Inf)
        for col in features:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        
        # Teniamo traccia degli ID originali
        valid_mask = chunk[features].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        valid_chunk = chunk[valid_mask].copy()
        
        if valid_chunk.empty: continue
        
        # C. Scaling (Batch)
        X_scaled = scaler.transform(valid_chunk[features].values)
        ids = valid_chunk[ID_COLUMN].values

        # --- FASE 1: FILTRO ANOMALIE (GPU Batch) ---
        if filter_type == 'Autoencoder':
            # Predizione massiva su GPU
            reconstructions = filter_model.predict(X_scaled, batch_size=4096, verbose=0)
            filter_scores = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
            is_anomaly = filter_scores > ae_threshold
        else:
            # Isolation Forest (CPU, ma molto veloce in batch)
            is_anomaly = filter_model.predict(X_scaled) == -1
            filter_scores = filter_model.decision_function(X_scaled)

        # Dividiamo il chunk in Inliers (vanno al funnel) e Outliers (bloccati)
        mask_inlier = ~is_anomaly
        
        # --- PREPARAZIONE RISULTATI OUTLIERS (Anomalie) ---
        results = []
        if np.any(is_anomaly):
            anom_ids = ids[is_anomaly]
            anom_scores = filter_scores[is_anomaly]
            for i in range(len(anom_ids)):
                results.append([anom_ids[i], "ANOMALY", "None", 0.0, "Filter", round(anom_scores[i], 6)])

        # --- FASE 2: FUNNEL (GPU Batch) ---
        if np.any(mask_inlier):
            X_funnel = X_scaled[mask_inlier]
            ids_funnel = ids[mask_inlier]
            scores_funnel = filter_scores[mask_inlier]
            
            # Chiamata alla funzione BATCH ottimizzata
            y_pred, y_conf, y_stage = predict_hierarchical_batch(
                X_funnel, models_dict, gate_threshold=0.05, spec_threshold=0.90
            )
            
            # Formattazione risultati
            for i in range(len(ids_funnel)):
                label = str(y_pred[i])
                if le:
                    try: label = le.inverse_transform([y_pred[i]])[0]
                    except: pass
                
                results.append([
                    ids_funnel[i],
                    "OK" if y_conf[i] >= 0.5 else "UNCERTAIN",
                    label,
                    round(y_conf[i], 4),
                    y_stage[i],
                    round(scores_funnel[i], 6)
                ])

        # --- D. SCRITTURA SU DISCO ---
        if results:
            df_out = pd.DataFrame(results, columns=csv_header)
            df_out.to_csv(MY_OUTPUT_PATH, mode='a', header=False, index=False)
            processed_count += len(df_out)

        # Log di progresso ogni chunk
        elapsed = (time.time() - start_time) / 60
        print(f"{prefix} Chunk {chunk_idx} completato. Processate: {processed_count} stelle. ({elapsed:.1f} min)")
        
        # Pulizia memoria aggressiva
        del chunk, valid_chunk, X_scaled, results, df_out
        gc.collect()

    print(f"\n✅ {prefix} LAVORO COMPLETATO. Totale: {processed_count} stelle salvate in {MY_OUTPUT_PATH}")

if __name__ == "__main__":
    run_pipeline()