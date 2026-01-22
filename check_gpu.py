import os
import sys

print(f"{'='*60}")
print("🔍 GPU DIAGNOSTIC TOOL")
print(f"{'='*60}")

# 1. CONTROLLO VARIABILI D'AMBIENTE
print("\n1️⃣  VARIABILI D'AMBIENTE")
cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES', 'Non impostata (Vede tutto)')
print(f"   CUDA_VISIBLE_DEVICES: {cuda_env}")

# 2. CONTROLLO DRIVER DI SISTEMA (nvidia-smi)
print("\n2️⃣  CHECK DRIVER (nvidia-smi)")
try:
    exit_code = os.system('nvidia-smi -L')
    if exit_code != 0:
        print("   ⚠️  Comando nvidia-smi fallito o non trovato.")
except Exception as e:
    print(f"   ❌ Errore esecuzione nvidia-smi: {e}")

# 3. CONTROLLO TENSORFLOW
print("\n3️⃣  CHECK TENSORFLOW")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print(f"   Versione TF: {tf.__version__}")
    print(f"   GPU Rilevate da TF: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"   -> GPU {i}: {gpu}")
    
    # Test allocazione memoria
    if gpus:
        try:
            with tf.device('/device:GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print("   ✅ Test calcolo TF su GPU: RIUSCITO")
        except Exception as e:
            print(f"   ❌ Test calcolo TF FALLITO: {e}")
except ImportError:
    print("   ❌ TensorFlow non installato.")

# 4. CONTROLLO PYTORCH (Usato da TabPFN)
print("\n4️⃣  CHECK PYTORCH")
try:
    import torch
    print(f"   Versione Torch: {torch.__version__}")
    available = torch.cuda.is_available()
    print(f"   CUDA Available: {available}")
    
    if available:
        print(f"   Device Name: {torch.cuda.get_device_name(0)}")
        print(f"   Device Count: {torch.cuda.device_count()}")
        # Test calcolo
        try:
            x = torch.tensor([1.0, 2.0]).cuda()
            y = x * 2
            print("   ✅ Test calcolo Torch su GPU: RIUSCITO")
        except Exception as e:
            print(f"   ❌ Test calcolo Torch FALLITO: {e}")
    else:
        print("   ⚠️  Torch non vede la GPU (Userà CPU).")

except ImportError:
    print("   ❌ PyTorch non installato.")

print(f"\n{'='*60}")
print("FINE DIAGNOSTICA")
print(f"{'='*60}")