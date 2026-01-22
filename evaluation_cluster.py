import os

# --- CONFIGURAZIONE GPU (Cruciale per Leonardo) ---
# 1. NON nascondiamo la GPU (altrimenti TabPFN crasha)
# 2. Impostiamo flag per evitare crash dei driver TensorFlow (INVALID_HANDLE)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# --------------------------------------------------

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
OUTPUT_PATH = f"evaluation/results_{EXP_NAME}_gpu.csv"
ID_COLUMN = "sourceid"
TRAINING_PATH_CHECK = "data/campioneLARGE05_allinfo.csv"
CONTAMINANTS_PATH_CHECK = "data/contaminants/contaminants.csv"

CHUNK_SIZE = 500000 
N_JOBS = 16  # Riduciamo leggermente rispetto a -1 per non saturare la GPU
# -----------------

GLOBAL_MODELS_DICT = None
GLOBAL_LE = None

# ... (Le funzioni load_blacklist_ids e process_single_prediction rimangono IDENTICHE a prima) ...
def load_blacklist_ids():
    blacklist = set()
    print(f"\n1️⃣  CARICAMENTO BLACKLIST...")
    for path, label in [(TRAINING_PATH_CHECK, "Training"), (CONTAMINANTS_PATH_CHECK, "Contaminants")]:
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path, usecols=[ID_COLUMN])
                ids = set(df[ID_COLUMN].astype(str).values)
                blacklist.update(ids)
                del df; gc.collect()
            except: pass
    print(f"   🚫 ID in Blacklist: {len(blacklist):,}")
    return blacklist

def process_single_prediction(sample_flat, source_id):
    sample = sample_flat.reshape(1, -1)
    # Predizione usando i modelli globali (che ora stanno su GPU o CPU a seconda di come sono stati salvati)
    pred = predict_hierarchical_batch(
        sample, GLOBAL_MODELS_DICT, gate_threshold=0.05, spec_threshold=0.90
    )
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

def run_parallel_pipeline():
    global GLOBAL_MODELS_DICT, GLOBAL_LE
    
    # Setup TensorFlow Memory Growth (Anti-Crash)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"   ✅ GPU rilevata e configurata: {len(gpus)}")
        except RuntimeError as e:
            print(e)
    else:
        print("   ⚠️  Nessuna GPU rilevata (TensorFlow userà CPU)")

    print(f"{'='*60}")
    print(f"🚀 VALUTAZIONE CLUSTER (GPU ENABLED) - N_JOBS={N_JOBS}")
    print(f"{'='*60}")

    try:
        pipeline = load_hierarchical_system(MODEL_DIR, EXP_NAME)
        scaler = pipeline['scaler']
        filter_model = pipeline['filter_model']
        filter_type = pipeline['filter_type']
        ae_threshold = pipeline['ae_threshold']
        
        GLOBAL_MODELS_DICT = pipeline['models_dict']
        GLOBAL_LE = pipeline.get('le', None)
        features = pipeline['features']
        print(f"   ✅ Modelli caricati.")
    except Exception as e:
        print(f"❌ Errore caricamento: {e}")
        return

    if not os.path.exists(NEW_DATA_PATH): return
    blacklist_ids = load_blacklist_ids()

    os.makedirs("evaluation", exist_ok=True)
    if os.path.exists(OUTPUT_PATH): os.remove(OUTPUT_PATH)
    
    csv_header = ["id", "status", "pred_label", "confidence", "stage", "filter_score"]
    with open(OUTPUT_PATH, 'w') as f: f.write(",".join(csv_header) + "\n")

    print(f"\n2️⃣  AVVIO ELABORAZIONE...")
    with open(NEW_DATA_PATH, 'rb') as f: est_lines = sum(1 for _ in f) - 1
    
    cols_to_load = list(set([ID_COLUMN] + features))
    reader = pd.read_csv(NEW_DATA_PATH, usecols=cols_to_load, chunksize=CHUNK_SIZE)
    
    processed_count = 0
    
    with tqdm(total=est_lines, unit="row", desc="Processing") as pbar:
        for chunk in reader:
            chunk_len = len(chunk)
            
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
            
            # --- FASE 1: FILTRO ANOMALIE ---
            rows_to_predict = []
            anomalies = []
            
            # L'Autoencoder su GPU vettorizzato è fulmineo
            if filter_type == 'Autoencoder':
                # Batch prediction diretta Keras
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
                # Isolation Forest (CPU Sklearn)
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

            # --- FASE 2: FUNNEL ---
            if rows_to_predict:
                # Creazione generatore per Parallel
                tasks_generator = (
                    delayed(process_single_prediction)(row[0], row[1]) 
                    for row in rows_to_predict
                )
                results_funnel = Parallel(n_jobs=N_JOBS)(tasks_generator)
                
                for res, orig in zip(results_funnel, rows_to_predict):
                    res["filter_score"] = round(orig[2], 6)
            else:
                results_funnel = []

            all_results = anomalies + results_funnel
            if all_results:
                df_res = pd.DataFrame(all_results)
                df_res[csv_header].to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
                processed_count += len(df_res)

            pbar.update(chunk_len)
            pbar.set_postfix(Preds=processed_count)
            del chunk, valid_chunk, X_scaled, rows_to_predict, results_funnel; gc.collect()

    print(f"\n✅ COMPLETATO: {processed_count} predizioni salvate.")

if __name__ == "__main__":
    run_parallel_pipeline()