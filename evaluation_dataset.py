import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
from utils.util import load_hierarchical_system, predict_hierarchical_batch, check_anomaly_ae

# --- CONFIGURAZIONE ---
MODEL_DIR = "saved_models"
EXP_NAME = "Funnel_Autoencoder_TabPFN"
NEW_DATA_PATH = "data/filtrato_fmem1.csv"
OUTPUT_PATH = f"evaluation/results_{EXP_NAME}_final.csv"
ID_COLUMN = "sourceid"

TRAINING_PATH_CHECK = "data/campioneLARGE05_allinfo.csv"
CONTAMINANTS_PATH_CHECK = "data/contaminants/contaminants.csv"

# RAM safe: 500000. Se crasha, abbassa a 100000.
CHUNK_SIZE = 500000
# ----------------------

def load_blacklist_ids():
    """Carica gli ID da escludere (Training + Contaminanti) in un set."""
    blacklist = set()
    print(f"\n1️⃣  CARICAMENTO BLACKLIST...")
    
    for path, label in [(TRAINING_PATH_CHECK, "Training"), (CONTAMINANTS_PATH_CHECK, "Contaminants")]:
        if path and os.path.exists(path):
            try:
                print(f"   📖 Leggo {label}...")
                df = pd.read_csv(path, usecols=[ID_COLUMN])
                ids = set(df[ID_COLUMN].astype(str).values)
                blacklist.update(ids)
                del df
                gc.collect()
            except Exception as e:
                print(f"   ⚠️ Errore lettura {label}: {e}")
                
    print(f"   🚫 ID in Blacklist: {len(blacklist):,}")
    return blacklist

def analyze_dataset_detailed(filepath, features, blacklist_ids):
    """
    PASSAGGIO 1:
    Scorre il file per generare statistiche dettagliate su NaN/Inf
    SOLO sulle righe che non sono nella blacklist.
    """
    print(f"\n2️⃣  ANALISI STATISTICA DETTAGLIATA...")
    print("   (Scansione del file per controllare la qualità dei dati filtrati)")

    # Inizializza contatori per ogni feature
    nan_counts = pd.Series(0, index=features)
    inf_counts = pd.Series(0, index=features)
    
    stats = {
        "total_rows_file": 0,
        "blacklisted": 0,
        "rows_after_blacklist": 0,
        "bad_data_rows": 0,
        "valid_for_prediction": 0
    }
    
    cols_needed = list(set([ID_COLUMN] + features))
    
    # Stima righe totali per progress bar
    with open(filepath, 'rb') as f:
        estimated_lines = sum(1 for _ in f) - 1

    reader = pd.read_csv(filepath, usecols=cols_needed, chunksize=CHUNK_SIZE)
    
    with tqdm(total=estimated_lines, unit="row", desc="Analyzing Stats") as pbar:
        for chunk in reader:
            chunk_len = len(chunk)
            stats["total_rows_file"] += chunk_len
            
            # 1. Filtro Blacklist
            chunk[ID_COLUMN] = chunk[ID_COLUMN].astype(str)
            mask_blacklist = chunk[ID_COLUMN].isin(blacklist_ids)
            n_black = mask_blacklist.sum()
            stats["blacklisted"] += n_black
            
            # Lavoriamo solo su quelle NON blacklistate
            clean_chunk = chunk[~mask_blacklist].copy()
            stats["rows_after_blacklist"] += len(clean_chunk)
            
            if clean_chunk.empty:
                pbar.update(chunk_len)
                continue
                
            # 2. Check Qualità Dati (NaN/Inf)
            # Convertiamo tutto a numerico
            for col in features:
                clean_chunk[col] = pd.to_numeric(clean_chunk[col], errors='coerce')
            
            # Aggiorniamo statistiche per colonna
            nans = clean_chunk[features].isna().sum()
            infs = ((clean_chunk[features] == np.inf) | (clean_chunk[features] == -np.inf)).sum()
            
            nan_counts += nans
            inf_counts += infs
            
            # Contiamo le righe intere da scartare
            temp_check = clean_chunk[features].replace([np.inf, -np.inf], np.nan)
            mask_valid = temp_check.notna().all(axis=1)
            
            n_valid = mask_valid.sum()
            stats["valid_for_prediction"] += n_valid
            stats["bad_data_rows"] += (len(clean_chunk) - n_valid)
            
            pbar.update(chunk_len)
            
            del chunk, clean_chunk, temp_check
            gc.collect()
            
    # --- STAMPA DEL REPORT ---
    print(f"\n{'='*40}")
    print(f"📊 REPORT STATISTICO DATASET")
    print(f"{'='*40}")
    print(f"Totale Righe nel file:      {stats['total_rows_file']:,}")
    print(f"Sorgenti Note (Blacklist): -{stats['blacklisted']:,}")
    print(f"Righe Residue (Candidate):  {stats['rows_after_blacklist']:,}")
    print(f"Dati Corrotti (NaN/Inf):   -{stats['bad_data_rows']:,}")
    print(f"{'-'*40}")
    print(f"✅ RIGHE VALIDE PER PREDIZIONE: {stats['valid_for_prediction']:,}")
    print(f"{'='*40}")

    # Dettaglio Features (se ci sono problemi)
    total_bad_features = nan_counts + inf_counts
    if total_bad_features.sum() > 0:
        print("\n🔻 Dettaglio Problemi per Feature (Top 10):")
        report_df = pd.DataFrame({'NaNs': nan_counts, 'Infs': inf_counts, 'Total': total_bad_features})
        report_df = report_df[report_df['Total'] > 0].sort_values('Total', ascending=False)
        print(report_df.head(10))
    else:
        print("\n✨ Nessun problema (NaN/Inf) trovato nelle features!")

    return stats

def run_pipeline():
    print(f"{'='*60}")
    print(f"🚀 PIPELINE COMPLETA: STATS -> FILTRO -> PREDIZIONE")
    print(f"{'='*60}")

    # --- STEP 0: Caricamento Modello ---
    try:
        pipeline = load_hierarchical_system(MODEL_DIR, EXP_NAME)
        scaler = pipeline['scaler']
        filter_model = pipeline['filter_model']
        filter_type = pipeline['filter_type']
        ae_threshold = pipeline['ae_threshold']
        models_dict = pipeline['models_dict']
        features = pipeline['features']
        le = pipeline.get('le', None)
        print("   ✅ Modello caricato.")
    except Exception as e:
        print(f"❌ Errore modello: {e}")
        return

    if not os.path.exists(NEW_DATA_PATH):
        print(f"❌ File dati non trovato: {NEW_DATA_PATH}")
        return

    # --- STEP 1: Carica Blacklist ---
    blacklist_ids = load_blacklist_ids()

    # --- STEP 2: Analisi Dettagliata ---
    stats = analyze_dataset_detailed(NEW_DATA_PATH, features, blacklist_ids)
    
    if stats['valid_for_prediction'] == 0:
        print("⚠️ Nessuna riga valida da predire. Stop.")
        return

    # --- STEP 3: Preparazione Output ---
    os.makedirs("evaluation", exist_ok=True)
    if os.path.exists(OUTPUT_PATH): os.remove(OUTPUT_PATH)
    
    csv_header = ["id", "status", "pred_label", "confidence", "stage", "filter_score"]
    with open(OUTPUT_PATH, 'w') as f:
        f.write(",".join(csv_header) + "\n")

    # --- STEP 4: Predizione Streaming ---
    print(f"\n3️⃣  AVVIO VALUTAZIONE (Streaming)...")
    print(f"   (Predizione su {stats['valid_for_prediction']:,} righe pulite)")
    
    cols_to_load = list(set([ID_COLUMN] + features))
    reader = pd.read_csv(NEW_DATA_PATH, usecols=cols_to_load, chunksize=CHUNK_SIZE)
    
    predicted_counter = 0
    
    # La barra totale è basata sulla lunghezza del file, ma monitoriamo le predizioni
    with tqdm(total=stats['total_rows_file'], unit="row", desc="Processing") as pbar:
        for chunk in reader:
            results_buffer = []
            initial_len = len(chunk)
            
            # A. Filtro Blacklist
            chunk[ID_COLUMN] = chunk[ID_COLUMN].astype(str)
            chunk = chunk[~chunk[ID_COLUMN].isin(blacklist_ids)]
            
            if chunk.empty:
                pbar.update(initial_len)
                continue
                
            # B. Preprocessing & Check NaN (Replica logica Stats)
            for col in features:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                
            temp_chunk = chunk[features].replace([np.inf, -np.inf], np.nan)
            valid_mask = temp_chunk.notna().all(axis=1)
            valid_chunk = chunk[valid_mask].copy()
            
            if valid_chunk.empty:
                pbar.update(initial_len)
                continue

            # C. Scaling
            try:
                X_scaled = scaler.transform(valid_chunk[features].values)
            except Exception as e:
                print(f"❌ Error scaling: {e}")
                continue
            
            ids = valid_chunk[ID_COLUMN].values
            
            # D. Predizione
            for i, sample_flat in enumerate(X_scaled):
                sample = sample_flat.reshape(1, -1)
                
                res = {
                    "id": ids[i],
                    "status": "OK",
                    "pred_label": "None",
                    "confidence": 0.0,
                    "stage": "None",
                    "filter_score": 0.0
                }

                # Anomaly Filter
                is_anomaly = False
                if filter_type == 'IsolationForest':
                    if filter_model.predict(sample)[0] == -1: is_anomaly = True
                    res["filter_score"] = round(filter_model.decision_function(sample)[0], 4)
                elif filter_type == 'Autoencoder':
                    is_anomaly, mse = check_anomaly_ae(filter_model, sample, ae_threshold, return_mse=True)
                    res["filter_score"] = round(mse, 6)

                if is_anomaly:
                    res["status"] = "ANOMALY"
                else:
                    # Funnel Prediction
                    pred = predict_hierarchical_batch(
                        sample, models_dict, gate_threshold=0.05, spec_threshold=0.90
                    )
                    res["stage"] = pred['stage']
                    res["confidence"] = round(pred['confidence'], 4)
                    
                    if le:
                        try:
                            res["pred_label"] = le.inverse_transform([int(pred['class'])])[0]
                        except:
                            res["pred_label"] = str(pred['class'])
                    else:
                        res["pred_label"] = str(pred['class'])
                        
                    if pred['confidence'] < 0.5:
                        res["status"] = "UNCERTAIN"
                
                results_buffer.append(res)
            
            # E. Salvataggio
            if results_buffer:
                df_res = pd.DataFrame(results_buffer)
                df_res = df_res[csv_header]
                df_res.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
                predicted_counter += len(df_res)

            pbar.update(initial_len)
            pbar.set_postfix(Preds=predicted_counter)
            
            del chunk, valid_chunk, X_scaled, results_buffer, temp_chunk
            gc.collect()

    print(f"\n✅ COMPLETATO.")
    print(f"   Risultati salvati in: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_pipeline()