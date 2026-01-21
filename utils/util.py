import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from tabpfn import TabPFNClassifier

def preprocess_data(training_path, features, isplot=True):
    # 1. Caricamento Dataset
    data = pd.read_csv(training_path)
    print(f"Dataset originale: {data.shape}")
    print(f"Feature selezionate: {features}")
    
    # Analisi preliminare sovrapposizione NaN
    analyze_nan_overlap(data, features, isplot=isplot)

    target_col = 'bestclassification_1'
    
    # Lista di colonne fondamentali da controllare (Feature + Target)
    cols_to_check = features + [target_col]
    
    # 2. Analisi dei NaN per Feature
    nan_counts = data[cols_to_check].isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    
    if not nan_counts.empty:
        print("\n--- Riepilogo NaN trovati ---")
        for col, count in nan_counts.items():
            perc = (count / len(data)) * 100
            print(f"Colonna '{col}': {count} mancanti ({perc:.2f}%)")
        
        if isplot:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=nan_counts.values, y=nan_counts.index, palette='viridis')
            plt.title('Conteggio Valori Mancanti (NaN) per Feature')
            plt.xlabel('Numero di NaN')
            plt.ylabel('Feature')
            plt.show()
    else:
        print("\nNessun NaN trovato nelle feature selezionate.")

    # 4. Eliminazione NaN (Drop)
    # Rimuoviamo le righe che hanno ALMENO un NaN nelle colonne selezionate
    initial_rows = data.shape[0]
    data_clean = data.dropna(subset=cols_to_check).copy()
    final_rows = data_clean.shape[0]
    
    print("-" * 30)
    print(f"Righe originali: {initial_rows}")
    print(f"Righe rimaste: {final_rows}")
    print(f"Righe eliminate: {initial_rows - final_rows}")
    print("-" * 30)

    # 5. Creazione X e y (sui dati puliti)
    X = data_clean[features]
    y = data_clean[target_col]

    # === NUOVA SEZIONE: ANALISI DISTRIBUZIONE CLASSI ===
    print("\n--- Distribuzione delle Classi (Dataset Pulito) ---")
    class_counts = y.value_counts().sort_index() # Ordiniamo per etichetta
    total_samples = len(y)
    
    # Stampa testuale
    for cls, count in class_counts.items():
        perc = (count / total_samples) * 100
        print(f"Classe '{cls}': {count} campioni ({perc:.2f}%)")

    if isplot:
        plt.figure(figsize=(10, 6))
        # Usiamo countplot per contare automaticamente le categorie
        # order=class_counts.index assicura che l'ordine sia coerente
        ax = sns.countplot(x=target_col, data=data_clean, palette='magma', order=class_counts.index)
        
        plt.title(f'Distribuzione Classi su {total_samples} campioni totali')
        plt.xlabel('Classe')
        plt.ylabel('Numero di Campioni')
        
        # Aggiunta etichette (Numero e Percentuale) sopra ogni barra
        for p in ax.patches:
            height = p.get_height()
            # Evitiamo errori se height è 0 o NaN
            if height > 0:
                ax.annotate(f'{int(height)}\n({height/total_samples:.1%})', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 10), 
                            textcoords = 'offset points',
                            fontsize=10, color='black')
        plt.show()
    # ===================================================

    # 6. Encoding del Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("Mappatura classi:", dict(zip(le.classes_, le.transform(le.classes_))))

    return data_clean, X, y, y_encoded, le

def analyze_nan_overlap(data, features, isplot=True):
    # 1. Filtriamo solo le colonne che hanno effettivamente dei NaN
    cols_with_nan = [col for col in features if data[col].isna().sum() > 0]
    
    if not cols_with_nan:
        # print("Nessun NaN presente nelle feature indicate.")
        return

    # 2. Ordiniamo le colonne dalla più "vuota" (più NaN) alla più piena
    sorted_cols = data[cols_with_nan].isna().sum().sort_values(ascending=False).index.tolist()
    
    print(f"Colonne con NaN (ordinate per quantità): {sorted_cols}")

    # 3. CONTROLLO LOGICO DI INCLUSIONE (Subset)
    print("\n--- Verifica Inclusione (Gerarchia) ---")
    
    major_col = sorted_cols[0]
    
    for i in range(1, len(sorted_cols)):
        minor_col = sorted_cols[i]
        
        rows_missing_minor = data[data[minor_col].isna()]
        
        is_subset = rows_missing_minor[major_col].isna().all()
        
        if is_subset:
            print(f"✅ I NaN di '{minor_col}' sono TUTTI inclusi in '{major_col}'")
        else:
            mismatch = (~rows_missing_minor[major_col].isna()).sum()
            print(f"❌ '{minor_col}' ha {mismatch} NaN che NON sono presenti in '{major_col}'")
            
    if isplot:
        # 4. VISUALIZZAZIONE (Heatmap)
        plt.figure(figsize=(12, 8))
        sns.heatmap(data[sorted_cols].isna(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title("Mappa dei Valori Mancanti (Giallo = NaN)")
        plt.xlabel("Features (ordinate per numero di NaN)")
        plt.ylabel("Righe del dataset")
        plt.show()

def evaluate_features_and_nans(training_path, target_col='bestclassification_1'):
    # 1. Caricamento Dati Completi (senza filtri iniziali)
    data = pd.read_csv(training_path)
    print(f"Dataset caricato: {data.shape}")
    
    if target_col not in data.columns:
        raise ValueError(f"La colonna target '{target_col}' non esiste nel dataset.")
        
    y = data[target_col]
    X = data.drop(columns=[target_col])
    
    # 3. Preprocessing Automatico per "Far Masticare" tutto al modello
    print("Preprocessing automatico in corso...")
    
    # Identifichiamo le colonne non numeriche (stringhe/oggetti)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    print(f"- Feature Numeriche trovate: {len(numeric_cols)}")
    print(f"- Feature Categoriche trovate: {len(categorical_cols)}")
    
    # A. Gestione Stringhe -> Numeri (Ordinal Encoding)
    # Usiamo OrdinalEncoder che gestisce anche i NaN (se configurato) o lo facciamo a mano
    # Per semplicità in questa fase di test, convertiamo le stringhe in numeri interi
    if len(categorical_cols) > 0:
        # Creiamo una copia per non toccare i dati originali troppo presto
        X_encoded = X.copy()
        for col in categorical_cols:
            # Verifica rapida: se è un ID (tutti valori diversi), lo scartiamo
            if X[col].nunique() > len(X) * 0.95:
                print(f"⚠️ Warning: La colonna '{col}' sembra un ID (troppi valori unici). Viene esclusa.")
                X_encoded = X_encoded.drop(columns=[col])
                continue
            
            # Convertiamo stringhe in numeri. pd.factorize gestisce bene i NaN dandogli -1
            X_encoded[col], _ = pd.factorize(X[col])
    else:
        X_encoded = X.copy()

    # Aggiorniamo la lista delle colonne rimaste
    final_features = X_encoded.columns.tolist()

    # 4. Calcolo Percentuale NaN (sui dati originali, per accuratezza)
    # Nota: X_encoded ha trasformato i NaN categorici in -1, ma noi vogliamo sapere quanti erano veri NaN
    nan_percent = (X[final_features].isna().sum() / len(X)) * 100
    
    # 5. Gestione NaN nei Numerici per il modello
    # Sostituiamo i NaN numerici con -999 per permettere al Random Forest di girare
    imputer = SimpleImputer(strategy='constant', fill_value=-999)
    X_final = imputer.fit_transform(X_encoded)
    
    # 6. Preparazione Target (gestione eventuali NaN nel target)
    # Se il target ha NaN, quelle righe sono inutili per il training dell'importanza
    valid_rows = ~y.isna()
    X_final = X_final[valid_rows]
    y = y[valid_rows]
    
    # Encoding del target (stringhe -> numeri) se necessario
    if y.dtype == 'object':
        y_encoded = pd.factorize(y)[0]
    else:
        y_encoded = y

    # 7. Feature Selection con Random Forest
    print("Addestramento modello per calcolo importanza...")
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_final, y_encoded)
    
    importances = model.feature_importances_
    
    # 8. Creazione DataFrame Risultati
    analysis_df = pd.DataFrame({
        'Feature': final_features,
        'Importance': importances,
        'NaN_Percent': nan_percent.values
    })
    
    analysis_df = analysis_df.sort_values(by='Importance', ascending=False)
    
    # 9. Output e Grafico
    print("\n--- TOP 35 FEATURES PIÙ IMPORTANTI (SU TUTTO IL DATASET) ---")
    #print(analysis_df.head(35))
    print(analysis_df[analysis_df['NaN_Percent'] < 10.].head(35))
    
    plt.figure(figsize=(14, 10))
    
    # Scatter plot
    sns.scatterplot(
        data=analysis_df, 
        x='Importance', 
        y='NaN_Percent', 
        size='Importance', 
        sizes=(20, 500), 
        hue='Importance', 
        palette='magma',
        legend=False
    )
    
    # Etichette intelligenti: mostriamo solo le feature interessanti
    # (Alta importanza O molti NaN ma importanza media)
    for i in range(analysis_df.shape[0]):
        row = analysis_df.iloc[i]
        # Mostra label se Importanza > soglia media o se è una top feature
        if row['Importance'] > 0.01: 
            plt.text(
                row['Importance'], 
                row['NaN_Percent'] + 1, # Spostiamo un po' la scritta in su
                row['Feature'], 
                fontsize=9, 
                rotation=15
            )

    plt.title(f"Discovery Mode: Tutte le Feature vs Target '{target_col}'")
    plt.xlabel("Importanza (Random Forest)")
    plt.ylabel("% Valori Mancanti (NaN)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return analysis_df

def load_and_process_contaminants(path, features, scaler, max_samples=10000):
    """
    Carica un dataset reale di 'contaminanti'.
    Miglioramenti:
    - Gestione errori di tipo (stringhe al posto di numeri)
    - Rimozione valori infiniti oltre ai NaN
    - Campionamento opzionale se il file è troppo grosso
    """
    print(f"\n   -> Caricamento dataset contaminanti da: {path}")
    
    # 1. Caricamento
    try:
        # Leggiamo tutto, ma potresti ottimizzare con usecols se il file è gigantesco
        df_noise = pd.read_csv(path)
    except FileNotFoundError:
        print(f"❌ ERRORE: Il file {path} non esiste.")
        return None
    except Exception as e:
        print(f"❌ ERRORE durante la lettura del CSV: {e}")
        return None

    # 2. Controllo esistenza feature
    # Set operation è più veloce e pulita
    missing_cols = set(features) - set(df_noise.columns)
    if missing_cols:
        print(f"❌ ERRORE: Nel dataset contaminanti mancano queste colonne: {list(missing_cols)}")
        return None

    # 3. Selezione e Conversione Forzata a Numerico
    # Questo step è CRUCIALE: trasforma eventuali stringhe sporche in NaN
    X_noise = df_noise[features].copy()
    
    for col in features:
        # errors='coerce' trasforma "errore", "?", " " in NaN
        X_noise[col] = pd.to_numeric(X_noise[col], errors='coerce')

    # 4. Pulizia Approfondita (NaN e Inf)
    initial_len = len(X_noise)
    
    # Sostituiamo infiniti con NaN e poi buttiamo tutto
    X_noise.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_noise = X_noise.dropna()
    
    final_len = len(X_noise)
    diff = initial_len - final_len
    
    if diff > 0:
        print(f"      (Rimossi {diff} campioni sporchi: NaN, stringhe o Inf)")
    
    if final_len == 0:
        print("❌ ERRORE: Il dataset contaminanti è vuoto dopo la pulizia.")
        return None

    # 5. Campionamento (Opzionale)
    # Se hai 1 milione di contaminanti, ne bastano meno per validare il filtro
    if max_samples and final_len > max_samples:
        print(f"      Dataset molto grande ({final_len}). Campionamento a {max_samples} righe...")
        X_noise = X_noise.sample(n=max_samples, random_state=42)

    # 6. Scaling
    # Fondamentale usare .transform() e non .fit_transform()
    try:
        X_noise_scaled = scaler.transform(X_noise)
    except Exception as e:
        print(f"❌ ERRORE durante lo scaling dei contaminanti: {e}")
        return None
    
    print(f"      Caricati e pronti {len(X_noise_scaled)} campioni di contaminazione.")
    return X_noise_scaled

def train_isolation_forest(X_train, contamination=0.01):
    print(f"   -> Training Isolation Forest (Contamination={contamination})...")
    iso_forest = IsolationForest(n_estimators=500, contamination=contamination, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    return iso_forest

def train_autoencoder(X_train, encoding_dim=32, epochs=20, batch_size=32):
    print("   -> Training Autoencoder...")
    input_dim = X_train.shape[1]
    
    # Architettura dinamica
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    encoded = Dense(int(encoding_dim/2), activation='relu')(encoded) # Bottleneck
    decoded = Dense(encoding_dim, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded) # Output 0-1
    
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        validation_split=0.1
    )
    return autoencoder

def get_ae_threshold(autoencoder, X_train, percentile=95):
    reconstructions = autoencoder.predict(X_train, verbose=0)
    mse = np.mean(np.power(X_train - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, percentile)
    return threshold

# ==========================================
# 2. VALUTAZIONE E CONFRONTO FILTRI
# ==========================================

def generate_synthetic_noise(X_sample, n_samples=200):
    input_dim = X_sample.shape[1]
    # Genera rumore random tra -0.2 e 1.2 (per dati scalati 0-1)
    return np.random.uniform(low=-0.2, high=1.2, size=(n_samples, input_dim))

def compare_filters(iso_forest, autoencoder, ae_threshold, X_test, X_noise):
    """
    Confronta i filtri basandosi su Score Combinato = (Acc_Buoni + Acc_Rumore)/2.
    In caso di pareggio, vince chi protegge meglio dal rumore.
    """
    print("\n--- CONFRONTO FILTRI ---")
    
    # Isolation Forest
    if_test = iso_forest.predict(X_test)
    if_noise = iso_forest.predict(X_noise)
    acc_if_valid = np.mean(if_test == 1)
    acc_if_noise = np.mean(if_noise == -1)
    score_if = (acc_if_valid + acc_if_noise) / 2

    # Autoencoder
    ae_test = check_anomaly_ae(autoencoder, X_test, ae_threshold)
    ae_noise = check_anomaly_ae(autoencoder, X_noise, ae_threshold)
    acc_ae_valid = np.mean(~ae_test)
    acc_ae_noise = np.mean(ae_noise)
    score_ae = (acc_ae_valid + acc_ae_noise) / 2

    print(f"{'Metodo':<20} | {'Acc. Buoni':<10} | {'Acc. Rumore':<10} | {'SCORE':<10}")
    print("-" * 60)
    print(f"{'Isolation Forest':<20} | {acc_if_valid:.2%}     | {acc_if_noise:.2%}    | {score_if:.4f}")
    print(f"{'Autoencoder':<20} | {acc_ae_valid:.2%}     | {acc_ae_noise:.2%}    | {score_ae:.4f}")

    # Logica Decisionale
    margin = 0.015
    if abs(score_if - score_ae) > margin:
        return "IsolationForest" if score_if > score_ae else "Autoencoder"
    
    # Tie-Breaker: Vince chi blocca meglio il rumore
    print(">>> Punteggio simile. Applico Tie-Breaker (Safety First).")
    if acc_if_noise >= acc_ae_noise:
        return "IsolationForest"
    return "Autoencoder"

# ==========================================
# 3. CLASSIFICATORE MULTI-CLASSE
# ==========================================

def train_multiclass_classifier(X_train, y_train, model_type='rf', class_weight='balanced'):
    """
    Addestra il classificatore finale gestendo lo sbilanciamento.
    
    Parametri:
    - model_type: 'rf' per Random Forest, 'xgb' per XGBoost.
    - class_weight: 'balanced' (default) gestisce automaticamente le classi rare.
    """
    
    if model_type == 'rf':
        print(f"\n   -> Training Random Forest (Class Weight: {class_weight})...")
        clf = RandomForestClassifier(
            n_estimators=100, 
            class_weight=class_weight, 
            random_state=42, 
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
    elif model_type == 'xgb':
        print(f"\n   -> Training XGBoost (Mode: {model_type})...")
        
        # 1. Calcolo Pesi per gestire lo sbilanciamento (Workaround per XGBoost Multi-class)
        if class_weight == 'balanced':
            print("      (Calcolo sample_weights per bilanciare le classi in XGBoost)")
            sample_weights = compute_sample_weight(
                class_weight='balanced',
                y=y_train
            )
        else:
            sample_weights = None

        # 2. Definizione Modello
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=6,             # Profondità standard, aumentala se underfitta
            learning_rate=0.1,       # Standard
            objective='multi:softprob', # Fondamentale per avere le probabilità
            num_class=len(np.unique(y_train)),
            n_jobs=-1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        # 3. Training con pesi
        clf.fit(X_train, y_train, sample_weight=sample_weights)
        
    else:
        raise ValueError(f"Modello '{model_type}' non supportato. Usa 'rf' o 'xgb'.")
        
    return clf

def evaluate_classifier(clf, X_test, y_test, label_encoder=None):
    """
    Stampa classification report e matrice di confusione.
    """
    y_pred = clf.predict(X_test)
    
    if label_encoder:
        target_names = [str(c) for c in label_encoder.classes_]
    else:
        target_names = None
        
    print("\n--- Report Classificazione ---")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Plot Matrice Confusione
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names if target_names else "auto",
                yticklabels=target_names if target_names else "auto")
    plt.title("Matrice di Confusione")
    plt.ylabel('Reale')
    plt.xlabel('Predetto')
    plt.show()

def augment_data_gaussian(X, y, target_count=500, noise_scale=0.02):
    n_present = len(X)
    if n_present >= target_count: return X, y
    n_copies = int(target_count / n_present)
    X_aug_list = [X]; y_aug_list = [y]
    stds = np.std(X, axis=0); stds[stds == 0] = 1.0
    for _ in range(n_copies):
        noise = np.random.normal(0, stds * noise_scale, X.shape)
        X_aug_list.append(X + noise); y_aug_list.append(y)
    return np.vstack(X_aug_list), np.hstack(y_aug_list)

def train_hierarchical_classifier(X_train, y_train, rare_classes=[2, 3, 4], use_augmentation=False):
    """
    Funnel Strategy.
    use_augmentation=False -> Usa i dati puri (~500 Rare vs ~500 Comuni).
    use_augmentation=True  -> Gonfia le rare a 1000 vs 1000 Comuni.
    """
    strategy_name = "Augmented" if use_augmentation else "Pure Data"
    print(f"\n   🏗️  Training Funnel ({strategy_name} Strategy)...")
    
    # --- 0. TARGET UNIFICATO ---
    y_merged = y_train.copy()
    mask_rare = np.isin(y_train, rare_classes)
    MERGED_LABEL = 2 
    y_merged[mask_rare] = MERGED_LABEL
    
    # --- 1. GATEKEEPER (BalancedRF) ---
    print(f"      [1/3] Training Gatekeeper...")
    clf_gatekeeper = BalancedRandomForestClassifier(
        n_estimators=300, sampling_strategy="all", 
        replacement=True, random_state=42, n_jobs=-1
    )
    clf_gatekeeper.fit(X_train, y_merged)
    
    # --- 2. SPECIALIST (TabPFN) ---
    X_rare = X_train[mask_rare]
    y_rare = np.full(len(X_rare), MERGED_LABEL)
    
    if use_augmentation:
        # STRATEGIA A: AUGMENTATION (1000 vs 1000)
        print(f"            Augmenting Rare (Target: 1000)...")
        X_rare_final, y_rare_final = augment_data_gaussian(X_rare, y_rare, target_count=1000, noise_scale=0.03)
        n_slots_common = len(y_rare_final) # 1:1 ratio
    else:
        # STRATEGIA B: DATI PURI (120 vs 500)
        # Non tocchiamo le rare. Sono poche ma buone.
        X_rare_final, y_rare_final = X_rare, y_rare
        # Prendiamo un po' più di comuni per coprire la varianza, ma non troppe
        # TabPFN regge bene fino a 1024 totali.
        # Abbiamo ~120 rare. Possiamo mettere ~800 comuni senza problemi.
        n_slots_common = min(800, 1024 - len(y_rare_final))

    # Selezione Comuni (Safety Net)
    mask_common = ~mask_rare
    X_common_all = X_train[mask_common]
    
    idx_common = np.random.choice(len(X_common_all), size=n_slots_common, replace=False)
    X_bg = X_common_all[idx_common]
    y_bg = np.full(n_slots_common, 99) # 99 = Falso Allarme
    
    # Unione
    X_spec = np.vstack([X_rare_final, X_bg])
    y_spec = np.hstack([y_rare_final, y_bg])
    
    print(f"      [2/3] Training Specialist (TabPFN): {len(X_spec)} samples.")
    print(f"            ({len(y_rare_final)} Rare Reali vs {len(y_bg)} Comuni Random)")
    
    clf_specialist = TabPFNClassifier(device='auto', n_estimators=32)
    clf_specialist.fit(X_spec, y_spec)
    
    # --- 3. GENERALIST (XGBoost) ---
    print(f"      [3/3] Training Generalist...")
    y_common_labels = y_train[mask_common]
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_common_labels)
    
    clf_generalist = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        n_jobs=-1, random_state=42, eval_metric='mlogloss'
    )
    clf_generalist.fit(X_common_all, y_common_labels, sample_weight=sample_weights)
    
    return {
        "gatekeeper": clf_gatekeeper,
        "specialist": clf_specialist,
        "generalist": clf_generalist
    }

def predict_hierarchical_batch(X_scaled, models_dict, rare_label_merged=2, gate_threshold=0.05, spec_threshold=0.60):
    """
    Predizione a Imbuto con Confidence Check sullo Specialista.
    
    Parametri:
    - gate_threshold (0.05): Il Gatekeeper lascia passare tutto ciò che ha >5% di probabilità di essere raro.
    - spec_threshold (0.60): TabPFN deve essere sicuro almeno al 60% per confermare una rara.
                             Altrimenti viene considerato un Falso Allarme.
    """
    clf_gate = models_dict["gatekeeper"]
    clf_spec = models_dict["specialist"]
    clf_gen  = models_dict["generalist"]
    
    n_samples = len(X_scaled)
    final_preds = np.zeros(n_samples, dtype=int)
    final_probs = np.zeros(n_samples, dtype=float)
    
    status_flags = np.full(n_samples, "Direct_Generalist", dtype=object)

    # --- 1. GATEKEEPER (Balanced RF) ---
    gate_probs_all = clf_gate.predict_proba(X_scaled)
    classes_gate = clf_gate.classes_
    try:
        idx_rare = np.where(classes_gate == rare_label_merged)[0][0]
        prob_is_rare = gate_probs_all[:, idx_rare]
    except:
        prob_is_rare = np.zeros(n_samples)

    mask_to_specialist = (prob_is_rare >= gate_threshold)
    mask_to_generalist = ~mask_to_specialist
    
    # Aggiorniamo status
    status_flags[mask_to_specialist] = "Sent_to_Specialist"

    # 2. GENERALIST (Percorso Diretto)
    if np.any(mask_to_generalist):
        X_gen = X_scaled[mask_to_generalist]
        gen_preds = clf_gen.predict(X_gen)
        gen_probs = np.max(clf_gen.predict_proba(X_gen), axis=1)
        final_preds[mask_to_generalist] = gen_preds
        final_probs[mask_to_generalist] = gen_probs

    # 3. SPECIALIST
    if np.any(mask_to_specialist):
        X_spec = X_scaled[mask_to_specialist]
        spec_preds = clf_spec.predict(X_spec)
        spec_probs_all = clf_spec.predict_proba(X_spec)
        spec_probs = np.max(spec_probs_all, axis=1)
        
        # Confidence Check
        mask_uncertain_rare = (spec_preds == rare_label_merged) & (spec_probs < spec_threshold)
        spec_preds[mask_uncertain_rare] = 99 
        
        mask_false_alarm = (spec_preds == 99)
        
        # Se era False Alarm, aggiorniamo lo status in "Rejected_by_Spec"
        # Questi sono i candidati contaminanti!
        # Dobbiamo mappare la maschera locale (mask_false_alarm) alla globale (mask_to_specialist)
        
        # Creiamo un array temporaneo di status per questo blocco
        temp_status = status_flags[mask_to_specialist]
        temp_status[mask_false_alarm] = "Rejected_by_Spec" # <--- I Contaminanti finiscono qui
        temp_status[~mask_false_alarm] = "Confirmed_Rare"
        status_flags[mask_to_specialist] = temp_status
        
        # Gestione predizioni (come prima)
        temp_preds = spec_preds.copy()
        temp_probs = spec_probs.copy()
        
        if np.any(mask_false_alarm):
            X_retry = X_spec[mask_false_alarm]
            retry_preds = clf_gen.predict(X_retry)
            retry_probs = np.max(clf_gen.predict_proba(X_retry), axis=1)
            temp_preds[mask_false_alarm] = retry_preds
            temp_probs[mask_false_alarm] = retry_probs
            
        final_preds[mask_to_specialist] = temp_preds
        final_probs[mask_to_specialist] = temp_probs

    return final_preds, final_probs, status_flags

def analyze_pipeline_components(X, y_true, models_dict, gate_thresh=0.05, spec_thresh=0.90):
    print(f"\n{'#'*60}")
    print(f"🔬 DIAGNOSTICA COMPONENTI PIPELINE")
    print(f"{'#'*60}")
    
    # Mappiamo y_true a 3 classi: 0, 1, 2 (Rare unificate)
    y_merged = y_true.copy()
    mask_rare = np.isin(y_true, [2, 3, 4])
    y_merged[mask_rare] = 2
    
    # ---------------------------------------------------------
    # 1. ANALISI GATEKEEPER (Il setaccio grosso)
    # ---------------------------------------------------------
    print(f"\n1️⃣  ANALISI GATEKEEPER (Soglia {gate_thresh})")
    clf_gate = models_dict["gatekeeper"]
    
    # Probabilità che sia classe 2 (Rara)
    gate_probs = clf_gate.predict_proba(X)
    idx_rare = np.where(clf_gate.classes_ == 2)[0][0]
    prob_is_rare = gate_probs[:, idx_rare]
    
    # Chi passa il cancello?
    gate_preds = (prob_is_rare >= gate_thresh).astype(int) # 1 = Passa (Sospetto Raro), 0 = Bloccato
    
    # Target vero binario per il Gatekeeper (1 = È Raro, 0 = È Comune)
    y_gate_target = (y_merged == 2).astype(int)
    
    print(classification_report(y_gate_target, gate_preds, target_names=["Bloccato (Comune)", "Passa (Sospetto)"]))
    
    cm = confusion_matrix(y_gate_target, gate_preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"   -> Rari Persi (FN): {fn} (GRAVE!)")
    print(f"   -> Rari Trovati (TP): {tp}")
    print(f"   -> Falsi Allarmi passati allo Specialist (FP): {fp}")

    # ---------------------------------------------------------
    # 2. ANALISI SPECIALIST (Il setaccio fine)
    # ---------------------------------------------------------
    print(f"\n2️⃣  ANALISI SPECIALIST (TabPFN - Soglia {spec_thresh})")
    
    # Analizziamo SOLO ciò che il Gatekeeper ha lasciato passare
    mask_passed = (gate_preds == 1)
    
    if np.sum(mask_passed) > 0:
        X_spec = X[mask_passed]
        y_spec_true = y_merged[mask_passed] # Label vere (0, 1, 2)
        
        clf_spec = models_dict["specialist"]
        spec_probs_all = clf_spec.predict_proba(X_spec)
        spec_probs = np.max(spec_probs_all, axis=1) # Confidenza massima
        spec_preds_raw = clf_spec.predict(X_spec)
        
        # Logica Soglia Specialist
        # Se predice 2 (Raro) MA confidenza < soglia -> diventa 99 (Comune)
        mask_uncertain = (spec_preds_raw == 2) & (spec_probs < spec_thresh)
        spec_preds_final = spec_preds_raw.copy()
        spec_preds_final[mask_uncertain] = 99
        
        # Per il report: 1 = Confermato Raro, 0 = Scartato come Comune
        spec_binary_pred = (spec_preds_final == 2).astype(int)
        
        # Target vero: Era davvero raro?
        y_spec_target = (y_spec_true == 2).astype(int)
        
        print(f"   Campioni analizzati: {len(X_spec)} (di cui {np.sum(y_spec_target)} rari veri)")
        print(classification_report(y_spec_target, spec_binary_pred, target_names=["Scartato (Falso Allarme)", "Confermato Raro"]))
        
        cm = confusion_matrix(y_spec_target, spec_binary_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"   -> Rari Salvati (TP): {tp}")
        print(f"   -> Rari Persi per troppa severità (FN): {fn}")
        print(f"   -> Falsi Allarmi Ripuliti (TN): {tn} (Bravo TabPFN!)")
        print(f"   -> Falsi Allarmi Rimasti (FP): {fp}")
    else:
        print("   Nessun campione ha passato il Gatekeeper.")

    # ---------------------------------------------------------
    # 3. ANALISI GENERALIST (0 vs 1)
    # ---------------------------------------------------------
    print(f"\n3️⃣  ANALISI GENERALIST (XGBoost 0 vs 1)")
    
    # Il generalista vede:
    # A. Quelli bloccati dal Gatekeeper
    # B. Quelli scartati dallo Specialist
    # Per semplicità, valutiamolo su TUTTE le stelle comuni vere (0 e 1)
    # per vedere quanto è bravo nel suo lavoro base.
    
    mask_common_true = (y_merged != 2)
    X_gen = X[mask_common_true]
    y_gen_true = y_merged[mask_common_true]
    
    clf_gen = models_dict["generalist"]
    gen_preds = clf_gen.predict(X_gen)
    
    print(classification_report(y_gen_true, gen_preds, target_names=["RRab", "RRc"]))

def analyze_contaminants_flow(X_noise, filter_model, filter_type, ae_threshold, models_dict, gate_thresh=0.05, spec_thresh=0.90):
    print(f"\n{'#'*60}")
    print(f"👽 ANALISI FLUSSO CONTAMINANTI ({len(X_noise)} campioni)")
    print(f"{'#'*60}")
    
    n_total = len(X_noise)
    
    # ---------------------------------------------------------
    # 1. LIVELLO FILTRO (Anomaly Detection)
    # ---------------------------------------------------------
    print(f"\n1️⃣  LIVELLO FILTRO ({filter_type})")
    
    if filter_type == 'IsolationForest':
        # -1 = Anomalia (Bloccato), 1 = Normale (Passa)
        filter_preds = filter_model.predict(X_noise)
        n_blocked = np.sum(filter_preds == -1)
        mask_passed = (filter_preds == 1)
    else:
        # Autoencoder
        is_anomaly = check_anomaly_ae(filter_model, X_noise, ae_threshold)
        n_blocked = np.sum(is_anomaly)
        mask_passed = ~is_anomaly
        
    n_passed = np.sum(mask_passed)
    print(f"   -> Bloccati: {n_blocked} ({n_blocked/n_total:.1%}) ✅")
    print(f"   -> Passati:  {n_passed} ({n_passed/n_total:.1%}) ⚠️")
    
    if n_passed == 0:
        print("\n   🎉 OTTIMO! Il filtro ha bloccato tutto. Nessun contaminante entra nella pipeline.")
        return

    # Prendiamo solo i sopravvissuti
    X_survivors = X_noise[mask_passed]
    
    # ---------------------------------------------------------
    # 2. LIVELLO GATEKEEPER (Balanced RF)
    # ---------------------------------------------------------
    print(f"\n2️⃣  LIVELLO GATEKEEPER (Soglia {gate_thresh})")
    clf_gate = models_dict["gatekeeper"]
    
    # Probabilità Raro (Classe 2)
    gate_probs = clf_gate.predict_proba(X_survivors)
    # Troviamo indice classe 2
    classes = clf_gate.classes_
    try:
        idx_rare = np.where(classes == 2)[0][0]
        prob_is_rare = gate_probs[:, idx_rare]
    except:
        prob_is_rare = np.zeros(len(X_survivors))
        
    # Maschera: Chi va dallo specialista?
    mask_to_spec = (prob_is_rare >= gate_thresh)
    n_to_spec = np.sum(mask_to_spec)
    n_to_gen = len(X_survivors) - n_to_spec
    
    print(f"   -> Mandati al Generalist (0/1): {n_to_gen} ({n_to_gen/n_total:.1%})")
    print(f"   -> Mandati allo Specialist (Sospetti): {n_to_spec} ({n_to_spec/n_total:.1%}) 🚨")
    
    if n_to_spec == 0:
        print("   Nessun contaminante è stato ritenuto 'Sospetto Raro'.")
        return

    # Prendiamo i sospetti
    X_suspects = X_survivors[mask_to_spec]

    # ---------------------------------------------------------
    # 3. LIVELLO SPECIALIST (TabPFN)
    # ---------------------------------------------------------
    print(f"\n3️⃣  LIVELLO SPECIALIST (TabPFN - Soglia {spec_thresh})")
    clf_spec = models_dict["specialist"]
    
    spec_preds = clf_spec.predict(X_suspects)
    spec_probs_all = clf_spec.predict_proba(X_suspects)
    spec_probs = np.max(spec_probs_all, axis=1)
    
    # Logica finale: È confermato Raro (2) SOLO SE prob >= soglia
    # Altrimenti viene declassato a Falso Allarme (99)
    mask_confirmed_rare = (spec_preds == 2) & (spec_probs >= spec_thresh)
    
    n_leaked = np.sum(mask_confirmed_rare)
    n_saved = len(X_suspects) - n_leaked
    
    print(f"   -> Riconosciuti come Falsi Allarmi: {n_saved} ✅")
    print(f"   -> ERRONEAMENTE CLASSIFICATI 'RARI': {n_leaked} ({n_leaked/n_total:.1%}) ❌")
    
    if n_leaked > 0:
        print(f"   ⚠️ ATTENZIONE: {n_leaked} contaminanti sono finiti nel catalogo finale delle stelle rare!")
    else:
        print(f"   🏆 SUCCESSO: TabPFN ha ripulito tutti i contaminanti residui!")

def analyze_contaminants_statistics(path, features, label_col=None):
    """
    Esegue un'analisi esplorativa sui contaminanti simile a quella fatta per il training set.
    - Gerarchia dei NaN
    - Conteggio righe perse
    - Distribuzione delle tipologie di contaminanti (se label_col è presente)
    """
    print(f"\n{'='*60}")
    print(f"🕵️  ANALISI STATISTICA DATASET CONTAMINANTI")
    print(f"{'='*60}")
    print(f"File: {path}")

    # 1. Caricamento
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"❌ Errore: File non trovato.")
        return
    
    # Verifica colonne
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        print(f"⚠️ Attenzione: Mancano le colonne: {missing_cols}")
        return

    # 2. Analisi NaN Gerarchica
    # Filtriamo solo le colonne features che hanno NaN
    cols_with_nan = [col for col in features if df[col].isna().sum() > 0]
    
    if not cols_with_nan:
        print("\n✅ Nessun NaN trovato nelle features dei contaminanti.")
    else:
        # Ordiniamo per quantità di NaN
        nan_counts = df[cols_with_nan].isna().sum().sort_values(ascending=False)
        sorted_cols = nan_counts.index.tolist()
        
        print(f"\nColonne con NaN (ordinate): {sorted_cols}")
        print("\n--- Verifica Inclusione (Gerarchia) ---")
        
        major_col = sorted_cols[0] # La colonna con più NaN
        
        for i in range(1, len(sorted_cols)):
            minor_col = sorted_cols[i]
            
            # Se minor_col è NaN, major_col è anch'essa NaN?
            rows_missing_minor = df[df[minor_col].isna()]
            is_subset = rows_missing_minor[major_col].isna().all()
            
            if is_subset:
                print(f"✅ I NaN di '{minor_col}' sono TUTTI inclusi in '{major_col}'")
            else:
                mismatch = (~rows_missing_minor[major_col].isna()).sum()
                print(f"❌ '{minor_col}' ha {mismatch} NaN che NON sono presenti in '{major_col}'")

        print("\n--- Riepilogo NaN Contaminanti ---")
        for col, count in nan_counts.items():
            perc = (count / len(df)) * 100
            print(f"Colonna '{col}': {count} mancanti ({perc:.2f}%)")

    # 3. Pulizia e Conteggio Righe
    initial_rows = len(df)
    # Puliamo basandoci solo sulle features usate dal modello
    df_clean = df.dropna(subset=features)
    final_rows = len(df_clean)
    removed = initial_rows - final_rows
    
    print("-" * 30)
    print(f"Righe originali: {initial_rows}")
    print(f"Righe rimaste (Pulite): {final_rows}")
    print(f"Righe eliminate: {removed} ({(removed/initial_rows)*100:.1f}%)")
    print("-" * 30)

    # 4. Distribuzione Tipi Contaminanti (Opzionale)
    # Se il file ha una colonna tipo 'label', 'type', 'class' che dice cos'è l'oggetto
    if label_col:
        if label_col in df.columns:
            print(f"\n--- Distribuzione Tipi Contaminanti ({label_col}) ---")
            counts = df_clean[label_col].value_counts()
            for cls, count in counts.items():
                perc = (count / final_rows) * 100
                print(f"Tipo '{cls}': {count} campioni ({perc:.2f}%)")
        else:
            print(f"\n⚠️ Colonna label '{label_col}' non trovata nel CSV.")
    else:
        # Se non specifichi la colonna label, proviamo a indovinarne alcune comuni
        possible_labels = ['label', 'class', 'type', 'bestclassification', 'bestclassification_1', 'source_type']
        found = [c for c in possible_labels if c in df.columns]
        if found:
            print(f"\n--- Distribuzione Tipi (Colonna indovinata: '{found[0]}') ---")
            print(df_clean[found[0]].value_counts().head(10)) # Top 10 tipi

# ==========================================
# 4. GESTIONE I/O (SALVATAGGIO/CARICAMENTO)
# ==========================================

def load_hierarchical_system(model_dir, exp_name):
    """
    Carica l'intero sistema: Scaler, Filtro, 3 Classificatori (Funnel), LabelEncoder.
    """
    path = os.path.join(model_dir, exp_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cartella non trovata: {path}")

    print(f"📂 Caricamento modelli da: {path}")

    # 1. Carica Metadata
    meta_path = os.path.join(path, "metadata.json")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    # 2. Carica Scaler e LabelEncoder
    scaler = joblib.load(os.path.join(path, "scaler.joblib"))
    le = joblib.load(os.path.join(path, "le.joblib"))

    # 3. Carica Filtro (Autoencoder o IsolationForest)
    filter_type = metadata.get('filter_type', 'IsolationForest')
    if filter_type == 'Autoencoder':
        ae_path = os.path.join(path, "filter_ae.keras")
        if not os.path.exists(ae_path): ae_path = os.path.join(path, "filter_ae.h5")
        
        # Caricamento sicuro custom objects
        try:
            filter_model = tf.keras.models.load_model(ae_path, compile=False)
        except:
            filter_model = tf.keras.models.load_model(ae_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    else:
        filter_model = joblib.load(os.path.join(path, "filter_if.joblib"))

    # 4. Carica i Classificatori del FUNNEL
    models_dict = {}
    try:
        models_dict['gatekeeper'] = joblib.load(os.path.join(path, "gatekeeper.joblib"))
        models_dict['specialist'] = joblib.load(os.path.join(path, "specialist.joblib"))
        models_dict['generalist'] = joblib.load(os.path.join(path, "generalist.joblib"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"❌ Mancano i file dei modelli Funnel: {e}")

    return {
        "scaler": scaler,
        "le": le,
        "filter_model": filter_model,
        "filter_type": filter_type,
        "ae_threshold": metadata.get('ae_threshold', None),
        "features": metadata['features'],
        "rare_classes": metadata.get('rare_classes', []),
        "models_dict": models_dict
    }

def predict_hierarchical_batch(X_sample, models_dict, gate_threshold=0.05, spec_threshold=0.90):
    """
    Logica a Imbuto: Gatekeeper -> Specialist (se rara) -> Generalist (fallback).
    """
    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)

    gatekeeper = models_dict['gatekeeper']
    specialist = models_dict['specialist']
    generalist = models_dict['generalist']

    # 1. Gatekeeper
    gate_probs = gatekeeper.predict_proba(X_sample)[0]
    prob_rare = gate_probs[1] # Assumiamo indice 1 = Rara

    stage = "Generalist (Gate)"
    final_class = None
    final_conf = 0.0

    if prob_rare >= gate_threshold:
        # 2. Specialist
        spec_probs = specialist.predict_proba(X_sample)[0]
        max_spec_idx = np.argmax(spec_probs)
        max_spec_conf = spec_probs[max_spec_idx]
        pred_spec_global = specialist.classes_[max_spec_idx]

        if max_spec_conf >= spec_threshold:
            final_class = pred_spec_global
            final_conf = max_spec_conf
            stage = "Specialist"
        else:
            stage = "Generalist (Fallback)"
    
    # 3. Generalist
    if stage.startswith("Generalist"):
        gen_probs = generalist.predict_proba(X_sample)[0]
        max_gen_idx = np.argmax(gen_probs)
        final_conf = gen_probs[max_gen_idx]
        final_class = generalist.classes_[max_gen_idx]

    return {
        "class": final_class,
        "confidence": final_conf,
        "stage": stage
    }

def check_anomaly_ae(autoencoder, X_sample, threshold, return_mse=False):
    reconstructed = autoencoder.predict(X_sample, verbose=0)
    mse = np.mean(np.power(X_sample - reconstructed, 2), axis=1)[0]
    is_anomaly = mse > threshold
    if return_mse:
        return is_anomaly, mse
    return [is_anomaly]

def full_pipeline_predict(raw_data, system, threshold=0.90): 
    """Wrapper Imbuto."""
    scaler = system['scaler']
    data_scaled = scaler.transform(raw_data.reshape(1, -1))
    
    # 1. Filtro
    is_unknown = False
    if system['filter_type'] == 'IsolationForest':
        if system['filter_model'].predict(data_scaled)[0] == -1: is_unknown = True
    else:
        if check_anomaly_ae(system['filter_model'], data_scaled, system['ae_threshold'])[0]: is_unknown = True
            
    if is_unknown: return "SCONOSCIUTO (Filtro)"
    
    # 2. Imbuto
    preds, confs, flags = predict_hierarchical_batch(
        data_scaled, 
        system['models_dict'],
        spec_threshold=threshold
    )
    
    pred_class = int(preds[0])
    confidence = float(confs[0])
    status = flags[0]

    label_str = str(pred_class)
    if 'le' in system and system['le']:
        try: label_str = system['le'].inverse_transform([pred_class])[0]
        except: pass

    base_msg = f"Classe {label_str} (Conf: {confidence:.2f})"
    
    if status == "Confirmed_Rare":
        return f"💎 RARA CONFERMATA: {base_msg}"
        
    if status == "Rejected_by_Spec":
        return f"⚠️ SOSPETTO (Scartato da Specialist): {base_msg}"
        
    return base_msg

import pandas as pd
import numpy as np

def filter_known_sources(df_target, train_path, contaminants_path, id_col='sourceid'):
    """
    Rimuove dal dataframe target le righe che compaiono nel training set 
    o nel file dei contaminanti, basandosi sulla colonna ID.
    
    Args:
        df_target (pd.DataFrame): Il dataframe da pulire (evaluation set).
        train_path (str): Path del file CSV di training.
        contaminants_path (str): Path del file CSV dei contaminanti.
        id_col (str): Nome della colonna identificativa (es. 'sourceid').
        
    Returns:
        pd.DataFrame: Il dataframe senza le sorgenti note.
        dict: Statistiche sulla rimozione.
    """
    print(f"\n🧹 FILTRO SORGENTI NOTE (Training & Contaminants)...")
    initial_len = len(df_target)
    
    # Assicuriamoci che la colonna ID esista nel target
    if id_col not in df_target.columns:
        print(f"⚠️ Attenzione: Colonna '{id_col}' non trovata nel dataset target. Skip filtro overlap.")
        return df_target, {"removed_train": 0, "removed_cont": 0}

    # Set degli ID da rimuovere (usiamo stringhe per evitare problemi int vs str)
    ids_to_remove = set()
    stats = {"removed_train": 0, "removed_cont": 0}

    # 1. Carica Training IDs
    if train_path and os.path.exists(train_path):
        try:
            # Leggiamo solo la colonna ID per efficienza
            df_train = pd.read_csv(train_path, usecols=[id_col])
            train_ids = set(df_train[id_col].astype(str))
            ids_to_remove.update(train_ids)
            stats["removed_train"] = len(train_ids)
            print(f"   Training Set IDs caricati: {len(train_ids)}")
        except Exception as e:
            print(f"⚠️ Errore lettura Training set: {e}")

    # 2. Carica Contaminants IDs
    if contaminants_path and os.path.exists(contaminants_path):
        try:
            df_cont = pd.read_csv(contaminants_path, usecols=[id_col])
            cont_ids = set(df_cont[id_col].astype(str))
            ids_to_remove.update(cont_ids)
            stats["removed_cont"] = len(cont_ids)
            print(f"   Contaminants IDs caricati: {len(cont_ids)}")
        except Exception as e:
            # Alcuni file contaminanti potrebbero non avere header o avere nomi diversi
            print(f"⚠️ Errore lettura Contaminants set: {e}")

    # 3. Filtro effettivo
    # Convertiamo gli ID del target in stringa per il confronto
    mask_overlap = df_target[id_col].astype(str).isin(ids_to_remove)
    df_clean = df_target[~mask_overlap].copy()
    
    removed_count = mask_overlap.sum()
    print(f"   Sorgenti rimosse (già note): {removed_count}")
    print(f"   Dataset residuo: {len(df_clean)}")
    
    return df_clean, stats


def analyze_data_quality(df, features):
    """
    Analizza la presenza di NaN e valori Infiniti nelle features specificate.
    Restituisce un report e la maschera delle righe valide.
    
    Args:
        df (pd.DataFrame): Il dataframe da analizzare.
        features (list): Lista delle features da controllare.
        
    Returns:
        pd.Series: Maschera booleana (True = riga valida).
        pd.DataFrame: Report delle features mancanti/errate.
    """
    print(f"\n📊 ANALISI QUALITÀ DATI (Features: {len(features)})...")
    
    # Verifica esistenza colonne
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        print(f"❌ ERRORE CRITICO: Mancano le colonne: {missing_cols}")
        return pd.Series([False]*len(df), index=df.index), None

    subset = df[features]
    
    # 1. Check NaN
    nans = subset.isna().sum()
    
    # 2. Check Inf
    infs = ((subset == np.inf) | (subset == -np.inf)).sum()
    
    # 3. Totale righe "sporche" per colonna
    bad_data = nans + infs
    
    # Creiamo un report
    report = pd.DataFrame({
        'NaNs': nans,
        'Infs': infs,
        'Total Bad': bad_data,
        'Percent Bad': (bad_data / len(df) * 100).round(2)
    })
    
    # Filtriamo solo quelle che hanno problemi per visualizzazione pulita
    problematic_features = report[report['Total Bad'] > 0]
    
    if not problematic_features.empty:
        print("   Features con problemi (NaN o Inf):")
        print(problematic_features.sort_values(by='Total Bad', ascending=False))
    else:
        print("   ✅ Nessun NaN o Inf trovato nelle features!")

    # 4. Creazione Maschera Righe Valide (tutte le features ok)
    # Sostituiamo inf con nan temporaneamente per fare un check unico
    temp_subset = subset.replace([np.inf, -np.inf], np.nan)
    valid_rows_mask = temp_subset.notna().all(axis=1)
    
    print(f"   Righe completamente valide: {valid_rows_mask.sum()} / {len(df)} ({valid_rows_mask.mean():.1%})")
    
    return valid_rows_mask, report