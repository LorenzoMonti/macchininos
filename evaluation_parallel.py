import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
from utils.util import load_hierarchical_system, predict_hierarchical_batch, check_anomaly_ae
import joblib
from joblib import Parallel, delayed

# --- CONFIGURAZIONE LEONARDO ---
MODEL_DIR = "saved_models"
EXP_NAME = "Funnel_Autoencoder_TabPFN"
NEW_DATA_PATH = "data/filtrato_fmem1.csv"
OUTPUT_PATH = f"evaluation/results_{EXP_NAME}_parallel.csv"
ID_COLUMN = "sourceid"

TRAINING_PATH_CHECK = "data/campioneLARGE05_allinfo.csv"
CONTAMINANTS_PATH_CHECK = "data/contaminants/contaminants.csv"

# Chunk size grande per il cluster
CHUNK_SIZE = 500000 
# Numero di CPU da usare (su Leonardo usa -1 per usarle tutte)
N_JOBS = -1 
# -------------------------------

# Variabile globale per i modelli (per condividerla coi processi figli senza ricopiarla)
GLOBAL_MODELS_DICT = None
GLOBAL_LE = None

def load_blacklist_ids():
    blacklist = set()
    print(f"\n1️⃣  CARICAMENTO BLACKLIST...")
    for path, label in [(TRAINING_PATH_CHECK, "Training"), (CONTAMINANTS_PATH_CHECK, "Contaminants")]:
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path, usecols=[ID_COLUMN])
                ids = set(df[ID_COLUMN].astype(str).values)
                blacklist.update(ids)
                del df
                gc.collect()
            except: pass
    print(f"   🚫 ID in Blacklist: {len(blacklist):,}")
    return blacklist

def process_single_prediction(sample_flat, source_id):
    """
    Questa funzione verrà eseguita in parallelo da ogni CPU.
    Usa le variabili globali per accedere ai modelli TabPFN.
    """
    # Funnel Prediction
    sample = sample_flat.reshape(1, -1)
    
    # Nota: predict_hierarchical_batch è thread-safe se i modelli sono joblib
    pred = predict_hierarchical_batch(
        sample, GLOBAL_MODELS_DICT, gate_threshold=0.05, spec_threshold=0.90
    )
    
    res = {
        "id": source_id,
        "status": "OK",
        "pred_label": str(pred['class']),
        "confidence": round(pred['confidence'], 4),
        "stage": pred['stage'],
        "filter_score": 0.0 # Non disponibile qui, è stato calcolato prima
    }
    
    # Decoding label
    if GLOBAL_LE:
        try:
            res["pred_label"] = GLOBAL_LE.inverse_transform([int(pred['class'])])[0]
        except: pass
        
    if pred['confidence'] < 0.5:
        res["status"] = "UNCERTAIN"
        
    return res

def run_parallel_pipeline():
    global GLOBAL_MODELS_DICT, GLOBAL_LE
    
    print(f"{'='*60}")
    print(f"🚀 VALUTAZIONE PARALLELA SU CLUSTER (N_JOBS={N_JOBS})")
    print(f"{'='*60}")

    # 1. Caricamento Sistema
    # Usiamo os.environ per assicurarci che usi CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    try:
        pipeline = load_hierarchical_system(MODEL_DIR, EXP_NAME)
        scaler = pipeline['scaler']
        filter_model = pipeline['filter_model']
        filter_type = pipeline['filter_type']
        ae_threshold = pipeline['ae_threshold']
        
        # Salviamo in variabili globali per il multiprocessing
        GLOBAL_MODELS_DICT = pipeline['models_dict']
        GLOBAL_LE = pipeline.get('le', None)
        features = pipeline['features']
        
        print(f"   ✅ Modelli caricati. Pronti per parallelismo.")
    except Exception as e:
        print(f"❌ Errore caricamento: {e}")
        return

    if not os.path.exists(NEW_DATA_PATH): return
    blacklist_ids = load_blacklist_ids()

    # 2. Setup Output
    os.makedirs("evaluation", exist_ok=True)
    if os.path.exists(OUTPUT_PATH): os.remove(OUTPUT_PATH)
    csv_header = ["id", "status", "pred_label", "confidence", "stage", "filter_score"]
    with open(OUTPUT_PATH, 'w') as f:
        f.write(",".join(csv_header) + "\n")

    # 3. Processing
    print(f"\n2️⃣  AVVIO ELABORAZIONE...")
    print(f"   Chunk Size: {CHUNK_SIZE:,} | CPUs: {joblib.cpu_count()}")
    
    with open(NEW_DATA_PATH, 'rb') as f: est_lines = sum(1 for _ in f) - 1
    
    cols_to_load = list(set([ID_COLUMN] + features))
    reader = pd.read_csv(NEW_DATA_PATH, usecols=cols_to_load, chunksize=CHUNK_SIZE)
    
    processed_count = 0
    
    with tqdm(total=est_lines, unit="row", desc="Parallel Proc") as pbar:
        for chunk in reader:
            chunk_len = len(chunk)
            
            # A. FILTRO BLACKLIST
            chunk[ID_COLUMN] = chunk[ID_COLUMN].astype(str)
            chunk = chunk[~chunk[ID_COLUMN].isin(blacklist_ids)]
            if chunk.empty:
                pbar.update(chunk_len)
                continue

            # B. PREPROCESSING
            for col in features:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            
            temp = chunk[features].replace([np.inf, -np.inf], np.nan)
            valid_mask = temp.notna().all(axis=1)
            valid_chunk = chunk[valid_mask].copy()
            
            if valid_chunk.empty:
                pbar.update(chunk_len)
                continue

            # C. SCALING
            try:
                X_scaled = scaler.transform(valid_chunk[features].values)
            except:
                pbar.update(chunk_len)
                continue
            
            ids = valid_chunk[ID_COLUMN].values
            
            # --- FASE I: FILTRO ANOMALIE (SEQUENZIALE MA VELOCE) ---
            # L'Autoencoder è vettorizzato, quindi è velocissimo anche su 1 CPU
            # Non lo passiamo a Parallel perché i modelli Keras non si possono picklare bene
            
            rows_to_predict = []    # Lista di (array, id) da passare al Funnel
            anomalies = []          # Lista risultati anomalie
            
            # Calcolo anomalie in batch se possibile, o row-by-row se usiamo check_anomaly_ae custom
            # Per sicurezza usiamo un loop veloce qui, l'AE è leggero
            for i, sample_flat in enumerate(X_scaled):
                sample = sample_flat.reshape(1, -1)
                
                is_anomaly = False
                score = 0.0
                
                if filter_type == 'IsolationForest':
                     if filter_model.predict(sample)[0] == -1: is_anomaly = True
                     score = filter_model.decision_function(sample)[0]
                elif filter_type == 'Autoencoder':
                     # Qui usiamo la funzione importata
                     is_anomaly, mse = check_anomaly_ae(filter_model, sample, ae_threshold, return_mse=True)
                     score = mse
                
                if is_anomaly:
                    anomalies.append({
                        "id": ids[i], "status": "ANOMALY", "pred_label": "None",
                        "confidence": 0.0, "stage": "None", "filter_score": round(score, 6)
                    })
                else:
                    # Se è buono, lo aggiungiamo alla lista per il multiprocessing
                    # Passiamo solo i dati grezzi (numpy array), non oggetti complessi
                    rows_to_predict.append((sample_flat, ids[i], score))

            # --- FASE II: FUNNEL PREDICTION (PARALLELA) ---
            # Questa è la parte lenta (TabPFN) che ora distribuiamo su 32 CPU
            if rows_to_predict:
                # joblib.Parallel esegue la funzione su N core
                # backend='loky' è default e robusto
                results_funnel = Parallel(n_jobs=N_JOBS)(
                    delayed(process_single_prediction)(row[0], row[1])