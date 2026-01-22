import os
import argparse

# --- CONFIGURAZIONE AMBIENTE (DA FARE SUBITO) ---
# 1. Permetti a TF di crescere nella memoria (evita allocazione 100% subito)
# 2. Disabilita XLA per evitare conflitti con i driver del cluster
# 3. Riduci i log inutili
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
from utils.util import load_hierarchical_system, predict_hierarchical_batch, check_anomaly_ae
import joblib
from joblib import Parallel, delayed
import tensorflow as tf

# --- PARAMETRI DI CONFIGURAZIONE ---
MODEL_DIR = "saved_models"
EXP_NAME = "Funnel_Autoencoder_TabPFN"
NEW_DATA_PATH = "data/filtrato_fmem1.csv"
ID_COLUMN = "sourceid"

TRAINING_PATH_CHECK = "data/campioneLARGE05_allinfo.csv"
CONTAMINANTS_PATH_CHECK = "data/contaminants/contaminants.csv"

# Su Leonardo (120GB RAM) possiamo osare chunk grandi per velocità
CHUNK_SIZE = 500000 

# CPUs per ogni task GPU (32 core totali / 4 task = 8 core a testa)
N_JOBS = 8 
# -----------------------------------

# Variabili Globali per i Worker
GLOBAL_MODELS_DICT = None
GLOBAL_LE = None

def get_args():
    """Legge gli argomenti passati da SLURM/srun"""
    parser = argparse.ArgumentParser()
    # Identificativo del processo corrente (0, 1, 2, 3...)
    parser.add_argument("--shard_id", type=int, default=0, help="ID del worker corrente")
    # Numero totale di processi lanciati
    parser.add_argument("--num_shards", type=int, default=1, help="Numero totale di GPU usate")
    return parser.parse_args()

def load_blacklist_ids():
    """Carica gli ID da ignorare (Training + Contaminanti)"""
    blacklist = set()
    # Carichiamo silenziosamente per non spammare i log 4 volte
    for path in [TRAINING_PATH_CHECK, CONTAMINANTS_PATH_CHECK]:
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path, usecols=[ID_COLUMN])
                ids = set(df[ID_COLUMN].astype(str).values)
                blacklist.update(ids)
                del df; gc.collect()
            except: pass
    return blacklist

def process_single_prediction(sample_flat, source_id):
    """Funzione eseguita in parallelo sulle CPU del task"""
    sample = sample_flat.reshape(1, -1)
    
    # Predizione usando i modelli globali
    pred = predict_hierarchical_batch(
        sample, GLOBAL_MODELS_DICT, gate_threshold=0.05, spec_threshold=0.90
    )
    
    res = {
        "id": source_id, "status": "OK", 
        "pred_label": str(pred['class']), "confidence": round(pred['confidence'], 4),
        "stage": pred['stage'], "filter_score": 0.0
    }
    
    # Decoding etichetta
    if GLOBAL_LE:
        try: res["pred_label"] = GLOBAL_LE.inverse_transform([int(pred['class'])])[0]
        except: pass
        
    if pred['confidence'] < 0.5: res["status"] = "UNCERTAIN"
    return res

def run_pipeline():
    global GLOBAL_MODELS_DICT, GLOBAL_LE
    
    # 1. Recupero Argomenti Sharding
    args = get_args()
    worker_id = args.shard_id
    total_workers = args.num_shards
    
    # Nome file output UNIVOCO per questo worker
    MY_OUTPUT_PATH = f"evaluation/results_{EXP_NAME}_gpu_{worker_id}.csv"
    
    prefix = f"[GPU-{worker_id}]"
    print(f"{prefix} Avviato. File output: {MY_OUTPUT_PATH}")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"{prefix} Vedo {len(gpus)} GPU. (Dovrebbe essere 1 grazie a srun)")
    else:
        print(f"{prefix} ⚠️ ATTENZIONE: Nessuna GPU rilevata! Userò CPU.")

    # 2. Caricamento Modelli
    try:
        pipeline = load_hierarchical_system(MODEL_DIR, EXP_NAME)
        scaler = pipeline['scaler']
        filter_model = pipeline['filter_model']
        filter_type = pipeline['filter_type']
        ae_threshold = pipeline['ae_threshold']
        GLOBAL_MODELS_DICT = pipeline['models_dict']
        GLOBAL_LE = pipeline.get('le', None)
        features = pipeline['features']
    except Exception as e:
        print(f"{prefix} ❌ Errore caricamento modelli: {e}")
        return

    if not os.path.exists(NEW_DATA_PATH): return
    blacklist_ids = load_blacklist_ids()

    # 3. Setup Output
    os.makedirs("evaluation", exist_ok=True)
    # Se il file esiste già (magari da un run fallito), lo cancelliamo per ripartire
    if os.path.exists(MY_OUTPUT_PATH): os.remove(MY_OUTPUT_PATH)
    
    csv_header = ["id", "status", "pred_label", "confidence", "stage", "filter_score"]
    with open(MY_OUTPUT_PATH, 'w') as f: f.write(",".join(csv_header) + "\n")

    # 4. Elaborazione Streaming con Sharding
    # Lettura veloce righe totali
    with open(NEW_DATA_PATH, 'rb') as f: total_file_lines = sum(1 for _ in f) - 1
    
    # Calcolo righe stimate per QUESTO worker (per la barra di caricamento)
    est_lines_per_worker = total_file_lines // total_workers
    
    cols_to_load = list(set([ID_COLUMN] + features))
    reader = pd.read_csv(NEW_DATA_PATH, usecols=cols_to_load, chunksize=CHUNK_SIZE)
    
    processed_count = 0
    
    # Usiamo tqdm con descrizione personalizzata
    with tqdm(total=est_lines_per_worker, unit="row", desc=f"Worker-{worker_id}", position=worker_id) as pbar:
        
        for chunk_idx, chunk in enumerate(reader):
            
            # --- LOGICA SHARDING ---
            # Se questo chunk non tocca a me, lo salto
            if chunk_idx % total_workers != worker_id:
                continue
            
            chunk_len = len(chunk)

            # A. Filtri Blacklist
            chunk[ID_COLUMN] = chunk[ID_COLUMN].astype(str)
            chunk = chunk[~chunk[ID_COLUMN].isin(blacklist_ids)]
            if chunk.empty: pbar.update(chunk_len); continue

            # B. Preprocessing
            for col in features: chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            valid_mask = chunk[features].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            valid_chunk = chunk[valid_mask].copy()
            
            if valid_chunk.empty: pbar.update(chunk_len); continue

            # C. Scaling
            try: X_scaled = scaler.transform(valid_chunk[features].values)
            except: pbar.update(chunk_len); continue
            
            ids = valid_chunk[ID_COLUMN].values
            
            # --- FASE 1: FILTRO ANOMALIE (GPU Batch) ---
            rows_to_predict = []
            anomalies = []
            
            if filter_type == 'Autoencoder':
                # Batch prediction su GPU (Velocissimo)
                recons = filter_model.predict(X_scaled, verbose=0)
                mses = np.mean(np.power(X_scaled - recons, 2), axis=1)
                
                for i, mse in enumerate(mses):
                    if mse > ae_threshold:
                        anomalies.append({
                            "id": ids[i], "status": "ANOMALY", "pred_label": "None",
                            "confidence": 0.0, "stage": "None", "filter_score": round(mse, 6)
                        })
                    else:
                        rows_to_predict.append((X_scaled[i], ids[i], mse))
            else:
                 # Isolation Forest (CPU fallback)
                 preds = filter_model.predict(X_scaled)
                 scores = filter_model.decision_function(X_scaled)
                 for i, lbl in enumerate(preds):
                    if lbl == -1:
                        anomalies.append({
                            "id": ids[i], "status": "ANOMALY", "pred_label": "None",
                            "confidence": 0.0, "stage": "None", "filter_score": round(scores[i], 4)
                        })
                    else:
                        rows_to_predict.append((X_scaled[i], ids[i], scores[i]))

            # --- FASE 2: FUNNEL (CPU Parallel) ---
            if rows_to_predict:
                # Generatore per joblib
                tasks_generator = (
                    delayed(process_single_prediction)(r[0], r[1]) 
                    for r in rows_to_predict
                )
                # Esecuzione parallela sugli 8 core assegnati a questo task
                results_funnel = Parallel(n_jobs=N_JOBS)(tasks_generator)
                
                # Reinseriamo lo score del filtro
                for res, orig in zip(results_funnel, rows_to_predict):
                    res["filter_score"] = round(orig[2], 6)
            else:
                results_funnel = []

            # D. Scrittura Output
            all_results = anomalies + results_funnel
            if all_results:
                df_res = pd.DataFrame(all_results)
                # Scriviamo in append senza header
                df_res[csv_header].to_csv(MY_OUTPUT_PATH, mode='a', header=False, index=False)
                processed_count += len(df_res)

            pbar.update(chunk_len)
            
            # Pulizia Memoria
            del chunk, valid_chunk, X_scaled, rows_to_predict, results_funnel, recons, mses
            gc.collect()

    print(f"\n✅ {prefix} COMPLETATO. Predizioni: {processed_count}. File: {MY_OUTPUT_PATH}")

if __name__ == "__main__":
    run_pipeline()