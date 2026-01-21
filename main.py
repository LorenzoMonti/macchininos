import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow & Keras
import tensorflow as tf

# Import delle utility (assumendo che utils/util.py contenga le nuove funzioni aggiornate)
from utils.util import *

# --- CONFIGURAZIONE FEATURES ---
# features base ['p1', 'peaktopeakg', 'peaktopeakrp', 'peaktopeakbp', 'phi21', 'r21', 'phi31', 'r31', 'phirise']
# features selector ['p1','nhg2', 'a2', 'r21', 'phi2', 'peaktopeakg', 'a1', 'er21', 'phi21', 'ephi21', 'peaktopeakgerror', 'intaveragegerror', 'phi1', 'lcqualityflagg', 'peaktopeakbp', 'phirise', 'peaktopeakrp', 'nhg1']
# features Ale ['phi21', 'ephi21', 'phi31', 'ephi31', 'numberofgbandcleanepochs', 'p1', 'p1error', 'intaverageg', 'intaveragegerror', 'peaktopeakg', 'peaktopeakgerror', 'r21', 'er21', 'r31', 'er31', 'phirise', 'lcqualityflagg', 'nhg1']

features = ['phi21', 'ephi21', 'phi31', 'ephi31', 'numberofgbandcleanepochs', 
            'p1', 'intaverageg', 'intaveragegerror', 'peaktopeakg', 
            'peaktopeakgerror', 'r21', 'er21', 'r31', 'er31', 'phirise', 
            'lcqualityflagg', 'nhg1']

features_name = "Ale_features"

# --- CONFIGURAZIONE PATH E PARAMETRI ---
TRAINING_PATH = "data/campioneLARGE05_allinfo.csv"
CONTAMINANTS_PATH = "data/contaminants/contaminants.csv"    # Se None, usa rumore sintetico
SAVE_DIR = "saved_models" # Cartella dove salvare i modelli

# Definizione Classi Rare (Quelle che avevano F1-Score 0)
RARE_CLASSES = [2, 3, 4] 

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"🚀 AVVIO PIPELINE: FUNNEL ARCHITECTURE (Gatekeeper -> TabPFN -> XGB)")
    print(f"{'='*60}")

    # 1. PREPROCESSING
    _, X, y, y_encoded, le = preprocess_data(TRAINING_PATH, features, isplot=False)

    # 2. SPLIT E SCALING
    print("\n1. Preparazione Dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. GESTIONE RUMORE
    if CONTAMINANTS_PATH:
        analyze_contaminants_statistics(CONTAMINANTS_PATH, features, label_col='bestclassification') 
        print(f"   Caricamento contaminanti da: {CONTAMINANTS_PATH}")
        X_noise_scaled = load_and_process_contaminants(CONTAMINANTS_PATH, features, scaler)
        if X_noise_scaled is None: X_noise_scaled = generate_synthetic_noise(X_train_scaled)
    else:
        print("ℹ️  Nessun file contaminanti. Uso rumore sintetico.")
        X_noise_scaled = generate_synthetic_noise(X_train_scaled)

    # 4. TRAINING FILTRI (Anomaly Detection)
    print("\n2. Addestramento e Confronto Filtri...")
    iso_forest = train_isolation_forest(X_train_scaled)
    autoencoder = train_autoencoder(X_train_scaled, epochs=100, batch_size=32)
    ae_threshold = get_ae_threshold(autoencoder, X_train_scaled, percentile=95)
    
    best_filter_name = compare_filters(
        iso_forest, autoencoder, ae_threshold, X_test_scaled, X_noise_scaled
    )
    print(f"   🏆 Filtro Vincitore: {best_filter_name}")
    
    if best_filter_name == "IsolationForest":
        final_filter_model = iso_forest
        final_ae_thresh = None
    else:
        final_filter_model = autoencoder
        final_ae_thresh = ae_threshold

    # 5. TRAINING CLASSIFICATORE A IMBUTO
    print(f"\n3. Addestramento Classificatore Funnel (Gatekeeper + Specialist + Generalist)...")
    
    # Ora catturiamo il DIZIONARIO, non più due variabili separate
    models_dict = train_hierarchical_classifier(
        X_train_scaled, y_train, rare_classes=RARE_CLASSES, use_augmentation=False
    )

    # 6. VALUTAZIONE SUL TEST SET (Vettorizzata)
    print("\n4. Valutazione Performance (Batch Prediction)...")
    y_test_merged = y_test.copy()
    mask_rare_test = np.isin(y_test, RARE_CLASSES)
    y_test_merged[mask_rare_test] = 2 
    target_names_merged = ['RRab', 'RRc', 'ALL RARES']

    # Passiamo il dizionario models_dict alla funzione di predizione batch
    thresholds_to_test = [0.90] #[0.5, 0.60, 0.70, 0.80, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.99]
    
    best_f1 = 0
    best_thresh = 0.5

    for t in thresholds_to_test:
        print(f"\n--- TEST SOGLIA TABPFN: {t} ---")
        y_pred, _, _ = predict_hierarchical_batch(
            X_test_scaled, models_dict, 
            gate_threshold=0.05,
            spec_threshold=t     
        )
        
        # Stampiamo solo la riga delle Rare per brevità
        report = classification_report(y_test_merged, y_pred, target_names=target_names_merged, output_dict=True)
        rare_stats = report['ALL RARES']
        print(f"Precision: {rare_stats['precision']:.2f} | Recall: {rare_stats['recall']:.2f} | F1: {rare_stats['f1-score']:.2f}")
        
        if rare_stats['f1-score'] > best_f1:
            best_f1 = rare_stats['f1-score']
            best_thresh = t

    print(f"\n🏆 Miglior Soglia Trovata: {best_thresh} (F1: {best_f1:.2f})")
    
    # Stampiamo il report completo della migliore configurazione
    print(f"\n--- REPORT FINALE (Soglia {best_thresh}) ---")
    y_pred_best, _, _ = predict_hierarchical_batch(X_test_scaled, models_dict, spec_threshold=best_thresh)
    print(classification_report(y_test_merged, y_pred_best, target_names=target_names_merged))
    
    gate_t = 0.05
    spec_t = 0.90

    analyze_pipeline_components(
        X_test_scaled, 
        y_test, 
        models_dict, 
        gate_thresh=gate_t, 
        spec_thresh=spec_t
    )

    if X_noise_scaled is not None:
        analyze_contaminants_flow(
            X_noise_scaled,
            final_filter_model,
            best_filter_name,
            final_ae_thresh,
            models_dict,
            gate_thresh=gate_t,
            spec_thresh=spec_t
        )

    # 7. SALVATAGGIO SISTEMA COMPLETO
    exp_name = f"Funnel_{best_filter_name}_TabPFN"
    print(f"\n5. Salvataggio Modelli in '{SAVE_DIR}/{exp_name}'...")
    
    save_hierarchical_system(
        save_dir=SAVE_DIR,
        exp_name=exp_name,
        scaler=scaler,
        filter_model=final_filter_model,
        filter_type=best_filter_name,
        models_dict=models_dict,
        features=features,
        rare_classes=RARE_CLASSES,
        ae_thresh=final_ae_thresh,
        le=le
    )

    # 8. TEST CARICAMENTO E INFERENZA
    print("\n6. Test Pipeline di Produzione (Load & Predict)...")
    loaded_pipeline = load_hierarchical_system(SAVE_DIR, exp_name)
    
    # Test su un dato valido (classe rara se esiste)
    rare_indices = np.where(np.isin(y_test, RARE_CLASSES))[0]
    if len(rare_indices) > 0:
        idx = rare_indices[0]
        sample_valid = X_test.iloc[idx].values # Dato grezzo originale
        true_lbl = y_test[idx]
        print(f"   Input (Classe Reale {true_lbl}):")
        print(f"   Output: {full_pipeline_predict(sample_valid, loaded_pipeline)}")
    else:
        print("   (Nessuna classe rara trovata nel test set per la demo)")

    # Test su Rumore
    # De-scaliamo per simulare dato grezzo
    sample_noise_raw = scaler.inverse_transform(X_noise_scaled[0].reshape(1, -1))[0]
    print(f"   Input (Rumore/Contaminante):")
    print(f"   Output: {full_pipeline_predict(sample_noise_raw, loaded_pipeline)}")

    print(f"\n{'='*60}")
    print("✅ PROCESSO COMPLETATO")
    print(f"{'='*60}")