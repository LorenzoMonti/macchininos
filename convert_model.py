import tensorflow as tf
import os

# Percorso del modello che ti dà problemi
model_path = "saved_models/Funnel_Autoencoder_TabPFN/filter_ae.keras"
save_path_h5 = "saved_models/Funnel_Autoencoder_TabPFN/filter_ae.h5"

print(f"Caricamento da: {model_path}")
# Carica il modello
model = tf.keras.models.load_model(model_path)

print(f"Salvataggio in: {save_path_h5}")
# Salva in formato H5 (più compatibile)
model.save(save_path_h5, save_format='h5')

print("✅ Fatto!")