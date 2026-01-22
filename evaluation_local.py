import os
import tensorflow as tf

# --- CONFIGURAZIONE GPU LOCALE ---
# 1. NON disabilitiamo la GPU (togliamo CUDA_VISIBLE_DEVICES = -1)
# 2. Configurazione "Memory Growth": evita che TF rubi tutta la VRAM subito
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Meno log

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"   ✅ GPU Rilevata e configurata (Memory Growth attivo).")
    except RuntimeError as e:
        print(f"   ⚠️ Errore config GPU: {e}")
else:
    print("   ⚠️ Nessuna GPU trovata (Userò CPU).")
# ---------------------------------

import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
from utils.util import load_hierarchical_system, predict_hierarchical_batch, check_anomaly_ae

# --- PARAMETRI ---
MODEL_DIR = "saved_models"
EXP_NAME = "Funnel_Autoencoder_TabPFN"
NEW_DATA_PATH = "data/filtrato_fmem1.csv"
OUTPUT_PATH = f"evaluation/results_{EXP_NAME}_local_gpu.csv"
ID_COLUMN = "sourceid"

TRAINING_PATH_CHECK = "data/campioneLARGE05_allinfo.csv"
CONTAMINANTS_PATH_CHECK = "data/contaminants/contaminants.csv"

# Con la GPU, 50.000 va bene. Se va in "Out of Memory", scendi a 10.000 o 20.000
CHUNK_SIZE = 50000  
# -----------------

def load_blacklist_ids():
    """Carica ID da ignorare (Training + Contaminanti)"""
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

def run_local_pipeline():
    print(f"{'='*60}")
    print(f"💻 VALUTAZIONE LOCALE (STREAMING - GPU ATTIVA)")
    print(f"{'='*60}")

    # 1. Caricamento Modelli
    try:
        pipeline = load_hierarchical_system(MODEL_DIR, EXP_NAME)
        scaler = pipeline['scaler']
        filter_model = pipeline['filter_model']
        filter_type = pipeline['filter_type']
        ae_threshold = pipeline['ae_threshold']
        models_dict = pipeline['models_dict']
        features = pipeline['features']
        le = pipeline.get('le', None)
        print("   ✅ Modelli caricati correttamente.")
    except Exception as e:
        print(f"❌ Errore caricamento modelli: {e}")
        return

    if not os.path.exists(NEW_DATA_PATH):
        print(f"❌ File dati non trovato: {NEW_DATA_PATH}")
        return

    # 2. Blacklist
    blacklist_ids = load_blacklist_ids()

    # 3. Setup Output
    os.makedirs("evaluation", exist_ok=True)
    if os.path.exists(OUTPUT_PATH): os.remove(OUTPUT_PATH)
    
    csv_header = ["id", "status", "pred_label", "confidence", "stage", "filter_score"]
    with open(OUTPUT_PATH, 'w') as f: f.write(",".join(csv_header) + "\n")

    # 4. Elaborazione Streaming
    print(f"\n2️⃣  AVVIO ELABORAZIONE...")
    # Stima righe per la barra
    with open(NEW_DATA_PATH, 'rb') as f: est_lines = sum(1 for _ in f) - 1
    
    cols_to_load = list(set([ID_COLUMN] + features))
    reader = pd.read_csv(NEW_DATA_PATH, usecols=cols_to_load, chunksize=CHUNK_SIZE)
    
    processed_count = 0
    
    with tqdm(total=est_lines, unit="row", desc="Processing") as pbar:
        for chunk in reader:
            chunk_len = len(chunk)
            
            # A. Filtro Blacklist
            chunk[ID_COLUMN] = chunk[ID_COLUMN].astype(str)
            chunk = chunk[~chunk[ID_COLUMN].isin(blacklist_ids)]
            if chunk.empty:
                pbar.update(chunk_len); continue

            # B. Preprocessing Numerico
            for col in features: chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            
            # C. Rimozione NaN/Inf
            temp = chunk[features].replace([np.inf, -np.inf], np.nan)
            valid_mask = temp.notna().all(axis=1)
            valid_chunk = chunk[valid_mask].copy()
            
            if valid_chunk.empty:
                pbar.update(chunk_len); continue

            # D. Scaling
            try:
                X_scaled = scaler.transform(valid_chunk[features].values)
            except:
                pbar.update(chunk_len); continue
            
            ids = valid_chunk[ID_COLUMN].values
            results_buffer = []

            # E. Predizione
            # Nota: Poiché siamo in locale e potremmo avere poca VRAM,
            # facciamo comunque un ciclo sequenziale o un piccolo batch predict se il modello lo supporta.
            # Per sicurezza manteniamo il ciclo, ma l'inferenza interna userà la GPU.
            
            for i, sample_flat in enumerate(X_scaled):
                sample = sample_flat.reshape(1, -1)
                
                res = {
                    "id": ids[i], "status": "OK", "pred_label": "None", 
                    "confidence": 0.0, "stage": "None", "filter_score": 0.0
                }

                # 1. Filtro Anomalia
                is_anomaly = False
                if filter_type == 'Autoencoder':
                    # Check anomaly usa Keras -> GPU
                    is_anomaly, mse = check_anomaly_ae(filter_model, sample, ae_threshold, return_mse=True)
                    res["filter_score"] = round(mse, 6)
                else:
                    # Isolation Forest (CPU)
                    if filter_model.predict(sample)[0] == -1: is_anomaly = True
                    res["filter_score"] = round(filter_model.decision_function(sample)[0], 4)

                if is_anomaly:
                    res["status"] = "ANOMALY"
                else:
                    # 2. Funnel Classifier
                    # TabPFN userà la GPU se disponibile (Joblib/Torch)
                    pred = predict_hierarchical_batch(
                        sample, models_dict, gate_threshold=0.05, spec_threshold=0.90
                    )
                    res["stage"] = pred['stage']
                    res["confidence"] = round(pred['confidence'], 4)
                    res["pred_label"] = str(pred['class'])
                    
                    if le:
                        try: res["pred_label"] = le.inverse_transform([int(pred['class'])])[0]
                        except: pass
                    
                    if pred['confidence'] < 0.5:
                        res["status"] = "UNCERTAIN"
                
                results_buffer.append(res)

            # F. Scrittura su Disco
            if results_buffer:
                df_res = pd.DataFrame(results_buffer)
                df_res[csv_header].to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
                processed_count += len(df_res)

            pbar.update(chunk_len)
            pbar.set_postfix(Preds=processed_count)
            
            del chunk, valid_chunk, X_scaled, results_buffer
            gc.collect()

    print(f"\n✅ COMPLETATO: {processed_count} predizioni salvate in {OUTPUT_PATH}")

if __name__ == "__main__":
    run_local_pipeline()