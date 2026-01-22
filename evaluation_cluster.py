import os
import argparse  # <--- NUOVO: Per leggere argomenti da riga di comando

# Configurazione base
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

# --- PARAMETRI ---
MODEL_DIR = "saved_models"
EXP_NAME = "Funnel_Autoencoder_TabPFN"
NEW_DATA_PATH = "data/filtrato_fmem1.csv"
# Il nome output sarà dinamico (es. results_part_0.csv)
ID_COLUMN = "sourceid"
TRAINING_PATH_CHECK = "data/campioneLARGE05_allinfo.csv"
CONTAMINANTS_PATH_CHECK = "data/contaminants/contaminants.csv"

CHUNK_SIZE = 500000 
N_JOBS = 8 # 8 CPU per ogni GPU (32 cpu totali / 4 gpu = 8)
# -----------------

GLOBAL_MODELS_DICT = None
GLOBAL_LE = None

def get_args():
    parser = argparse.ArgumentParser()
    # Argomenti per il parallelismo su più GPU
    parser.add_argument("--shard_id", type=int, default=0, help="Indice del worker corrente (0, 1, 2, 3)")
    parser.add_argument("--num_shards", type=int, default=1, help="Numero totale di GPU/Worker usati")
    return parser.parse_args()

def load_blacklist_ids():
    blacklist = set()
    for path in [TRAINING_PATH_CHECK, CONTAMINANTS_PATH_CHECK]:
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path, usecols=[ID_COLUMN])
                blacklist.update(set(df[ID_COLUMN].astype(str).values))
                del df; gc.collect()
            except: pass
    return blacklist

def process_single_prediction(sample_flat, source_id):
    # (Identica a prima)
    sample = sample_flat.reshape(1, -1)
    pred = predict_hierarchical_batch(sample, GLOBAL_MODELS_DICT)
    res = {
        "id": source_id, "status": "OK", 
        "pred_label": str(pred['class']), "confidence": round(pred['confidence'], 4),
        "stage": pred['stage'], "filter_score": 0.0
    }
    if GLOBAL_LE:
        try: res["pred_label"] = GLOBAL_LE.inverse_transform([int(pred['class'])])[0]
        except: pass
    if pred['confidence'] < 0.5: res["status"] = "UNCERTAIN"
    return res

def run_pipeline():
    global GLOBAL_MODELS_DICT, GLOBAL_LE
    args = get_args()
    
    # Nome file output specifico per questo worker (es. results_gpu_0.csv)
    MY_OUTPUT_PATH = f"evaluation/results_{EXP_NAME}_gpu_{args.shard_id}.csv"
    
    print(f"{'='*60}")
    print(f"🚀 WORKER {args.shard_id + 1}/{args.num_shards} AVVIATO")
    print(f"   GPU Assegnata: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Auto')}")
    print(f"   Output file: {MY_OUTPUT_PATH}")
    print(f"{'='*60}")

    # Caricamento Modelli
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
        print(f"❌ Errore caricamento: {e}")
        return

    if not os.path.exists(NEW_DATA_PATH): return
    blacklist_ids = load_blacklist_ids()

    os.makedirs("evaluation", exist_ok=True)
    if os.path.exists(MY_OUTPUT_PATH): os.remove(MY_OUTPUT_PATH)
    
    csv_header = ["id", "status", "pred_label", "confidence", "stage", "filter_score"]
    with open(MY_OUTPUT_PATH, 'w') as f: f.write(",".join(csv_header) + "\n")

    cols_to_load = list(set([ID_COLUMN] + features))
    
    # Lettura file
    with open(NEW_DATA_PATH, 'rb') as f: est_lines = sum(1 for _ in f) - 1
    reader = pd.read_csv(NEW_DATA_PATH, usecols=cols_to_load, chunksize=CHUNK_SIZE)
    
    processed_count = 0
    
    # --- LOGICA DI SHARDING ---
    # Ogni worker processa solo i chunk dove (index % num_shards) == shard_id
    # Esempio: 
    # Worker 0 fa i chunk 0, 4, 8...
    # Worker 1 fa i chunk 1, 5, 9...
    
    with tqdm(total=est_lines // args.num_shards, unit="row", desc=f"GPU-{args.shard_id}") as pbar:
        for chunk_idx, chunk in enumerate(reader):
            
            # SE IL CHUNK NON TOCCA A ME, LO SALTO
            if chunk_idx % args.num_shards != args.shard_id:
                continue

            chunk_len = len(chunk)
            
            # --- (Da qui in poi il codice è identico a prima) ---
            chunk[ID_COLUMN] = chunk[ID_COLUMN].astype(str)
            chunk = chunk[~chunk[ID_COLUMN].isin(blacklist_ids)]
            if chunk.empty: pbar.update(chunk_len); continue

            for col in features: chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            valid_mask = chunk[features].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            valid_chunk = chunk[valid_mask].copy()
            if valid_chunk.empty: pbar.update(chunk_len); continue

            try: X_scaled = scaler.transform(valid_chunk[features].values)
            except: pbar.update(chunk_len); continue
            
            ids = valid_chunk[ID_COLUMN].values
            
            # FASE 1: FILTRO
            rows_to_predict = []
            anomalies = []
            
            if filter_type == 'Autoencoder':
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
                 # Isolation Forest code...
                 pass

            # FASE 2: FUNNEL
            if rows_to_predict:
                tasks_generator = (delayed(process_single_prediction)(r[0], r[1]) for r in rows_to_predict)
                results_funnel = Parallel(n_jobs=N_JOBS)(tasks_generator)
                for res, orig in zip(results_funnel, rows_to_predict):
                    res["filter_score"] = round(orig[2], 6)
            else:
                results_funnel = []

            all_results = anomalies + results_funnel
            if all_results:
                df_res = pd.DataFrame(all_results)
                df_res[csv_header].to_csv(MY_OUTPUT_PATH, mode='a', header=False, index=False)
                processed_count += len(df_res)

            pbar.update(chunk_len)
            del chunk, valid_chunk, X_scaled, rows_to_predict, results_funnel; gc.collect()

    print(f"\n✅ WORKER {args.shard_id} COMPLETATO: {processed_count} predizioni.")

if __name__ == "__main__":
    run_pipeline()