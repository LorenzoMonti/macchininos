import pandas as pd
import os

# --- CONFIGURAZIONE FILE ---
file_1_path = 'risultato_EB_BERRY 2.csv'
file_2_path = 'risultato_EB_BERRY_2.csv'
target_path = 'contaminants.csv'

def main():
    # 1. Controllo esistenza file target
    if not os.path.exists(target_path):
        print(f"Errore: Il file target '{target_path}' non esiste. Non posso eliminare nulla.")
        return

    # 2. Caricamento dei file
    try:
        df1 = pd.read_csv(file_1_path)
        df2 = pd.read_csv(file_2_path)
        df_target = pd.read_csv(target_path)
    except FileNotFoundError as e:
        print(f"Errore: File non trovato - {e}")
        return

    print(f"Righe iniziali nel file target: {len(df_target)}")

    # 3. Identificare quale file ha più righe tra i due di origine
    if len(df1) >= len(df2):
        df_big = df1
        df_small = df2
        print(f"File maggiore: {file_1_path}")
    else:
        df_big = df2
        df_small = df1
        print(f"File maggiore: {file_2_path}")

    # 4. Estrarre le righe presenti SOLO nel file grande (il "delta")
    # Merge outer con indicator=True
    merged_source = df_big.merge(df_small, how='outer', indicator=True)
    
    # Prendiamo solo quelle che stanno a sinistra (nel file grande)
    rows_to_delete = merged_source[merged_source['_merge'] == 'left_only'].drop(columns=['_merge'])

    if rows_to_delete.empty:
        print("I due file di origine sono identici o il più piccolo contiene già tutto. Nessuna riga da eliminare.")
        return

    print(f"Individuate {len(rows_to_delete)} righe dal file maggiore da rimuovere nel target.")

    # 5. Eliminazione dal file Target
    # Facciamo un merge LEFT tra il Target e le righe da cancellare.
    # Tutto ciò che matcha (indicator='both') va rimosso.
    # Tutto ciò che rimane 'left_only' (presente nel target ma NON nella lista di cancellazione) va tenuto.
    
    check_remove = df_target.merge(rows_to_delete, how='left', indicator=True)
    
    # Filtriamo tenendo solo ciò che era SOLO nel target originariamente
    df_clean = check_remove[check_remove['_merge'] == 'left_only'].drop(columns=['_merge'])

    # 6. Calcolo risultati e Salvataggio
    righe_rimosse = len(df_target) - len(df_clean)

    if righe_rimosse > 0:
        # Sovrascriviamo il file target con i dati puliti
        df_clean.to_csv(target_path, index=False)
        print(f"Operazione completata! Rimosse {righe_rimosse} righe da '{target_path}'.")
        print(f"Il file target ora contiene {len(df_clean)} righe.")
    else:
        print("Nessuna delle righe estratte era presente nel file target. Il file non è stato modificato.")

if __name__ == "__main__":
    main()