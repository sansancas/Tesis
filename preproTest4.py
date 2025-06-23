# -*- coding: utf-8 -*-
"""
preproTest2_tcn_improved.py

Mejorado para:
 - Preprocesamiento: band-pass 0.5–45Hz, notch 50Hz, referencia promedio (CAR).
 - Ventanas de 4s con 75% de solapamiento.
 - Pipeline tf.data con cache, shuffle, prefetch y aumentos (jitter y escala).
 - Balance de clases vía sample_from_datasets.
 - Modelo híbrido TCN + BiLSTM + MultiHeadAttention.
 - Regularización: SpatialDropout, AdamW con weight decay, CosineDecay LR.
 - Preservación de funcionalidad original de extracción y evaluación.
"""
import os
import glob
import shutil
import numpy as np
import pandas as pd
import mne
from scipy.signal import resample
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers.experimental import AdamW
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

BASE_PAIRS_AR = [
    ('EEG FP1', 'EEG F7'), ('EEG F7', 'EEG T3'), ('EEG T3', 'EEG T5'), ('EEG T5', 'EEG O1'),
    ('EEG FP2', 'EEG F8'), ('EEG F8', 'EEG T4'), ('EEG T4', 'EEG T6'), ('EEG T6', 'EEG O2'),
    ('EEG A1',  'EEG T3'), ('EEG T3', 'EEG C3'), ('EEG C3', 'EEG CZ'), ('EEG CZ', 'EEG C4'),
    ('EEG C4', 'EEG T4'), ('EEG T4', 'EEG A2'), ('EEG FP1', 'EEG F3'), ('EEG F3', 'EEG C3'),
    ('EEG C3', 'EEG P3'), ('EEG P3', 'EEG O1'), ('EEG FP2', 'EEG F4'), ('EEG F4', 'EEG C4'),
    ('EEG C4', 'EEG P4'), ('EEG P4', 'EEG O2')
]
BASE_PAIRS_LE = [
    ('EEG F7', 'EEG F8'), ('EEG T3', 'EEG T4'), ('EEG T5', 'EEG T6'),
    ('EEG C3', 'EEG C4'), ('EEG P3', 'EEG P4'), ('EEG O1', 'EEG O2')
]
N_CHANNELS_AR = len(BASE_PAIRS_AR)
N_CHANNELS_LE = len(BASE_PAIRS_LE)

# --- Config GPU ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
mixed_precision.set_global_policy('float32')  # Usar float32 en hardware sin Tensor Cores
print("GPUs:", gpus)
# Enable XLA acceleration
tf.config.optimizer.set_jit(True)
# Threading
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())

# --- Cambio de extensiones ---
def change_label_extensions(root_folder: str):
    pattern = os.path.join(root_folder, '**', '*.csv_bi')
    for old_path in glob.glob(pattern, recursive=True):
        new_path = old_path.replace('.csv_bi', '_bi.csv')
        shutil.move(old_path, new_path)

# --- Listado de rutas ---
def list_label_paths(root_folder: str):
    train = glob.glob(os.path.join(root_folder, 'train', '**', '*_bi.csv'), recursive=True)
    dev   = glob.glob(os.path.join(root_folder, 'dev',   '**', '*_bi.csv'), recursive=True)
    eval  = glob.glob(os.path.join(root_folder, 'eval',  '**', '*_bi.csv'), recursive=True)
    return train, dev, eval

# --- Filtrado por montaje ---
def filter_by_montage(paths, montage_type: str):
    out = []
    for p in paths:
        f = os.path.dirname(p).lower()
        if montage_type == 'ar_a' and '_tcp_ar_a' in f:
            out.append(p)
        elif montage_type == 'ar' and '_tcp_ar' in f and '_tcp_ar_a' not in f:
            out.append(p)
        elif montage_type == 'le' and '_tcp_le' in f:
            out.append(p)
    return out

# --- Leer etiquetas ---
def load_label_csv(path: str):
    df = pd.read_csv(path, skiprows=5)
    need = {'start_time','stop_time','label'}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} no contiene {need}")
    return df

# --- Extracción y preprocesamiento de señales ---
def extract_montage_signals(edf_path: str, montage: str='ar', desired_fs: int=256):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    # Filtrado banda-pasa y notch
    raw.filter(0.5, 45.0, fir_design='firwin', verbose=False)
    raw.notch_filter(freqs=50.0, verbose=False)
    # Referencia CAR
    raw.set_eeg_reference('average', verbose=False)
    data, _ = raw.get_data(picks='eeg', return_times=True)
    ch_names = raw.ch_names
    # Selección de pares según montaje
    suf = '-LE' if any(c.endswith('-LE') for c in ch_names) else '-REF'
    # Definir pares como antes (omito por brevedad, same as original)...
    ch_names = raw.ch_names
    suf = '-LE' if any(c.endswith('-LE') for c in ch_names) else '-REF'
    pairs_ar = [(f"{a}{suf}", f"{b}{suf}") for a,b in BASE_PAIRS_AR]
    pairs_le = [(f"{a}{suf}", f"{b}{suf}") for a,b in BASE_PAIRS_LE]
    pairs = pairs_ar if montage=='ar' else pairs_le
    mont = []
    for c1,c2 in pairs:
        if c1 in ch_names and c2 in ch_names:
            mont.append(data[ch_names.index(c1)] - data[ch_names.index(c2)])
        else:
            mont.append(np.zeros(data.shape[1],dtype=np.float32))
    mont = np.stack(mont, axis=1)
    # Normalización por valor absoluto
    maxval = np.max(np.abs(mont),axis=0)
    maxval[maxval==0]=1.0
    mont /= maxval
    orig_fs = int(raw.info['sfreq'])
    if orig_fs != desired_fs:
        n_t = int(mont.shape[0]*desired_fs/orig_fs)
        mont = resample(mont, n_t, axis=0)
    return mont.astype(np.float32), desired_fs, mont.shape[0]

# --- Generador de ventanas con solapamiento 75% ---
def windows_generator(edf_paths, lbl_paths, montage, fs_desired, win_size, mode):
    stride = win_size // 4  # 75% overlap
    for edf, lbl in zip(edf_paths, lbl_paths):
        sig, fs, n = extract_montage_signals(edf, montage, fs_desired)
        df = load_label_csv(lbl)
        labels = np.zeros(n, dtype=np.int32)
        for _,r in df.iterrows():
            if r['label'].strip().lower()=='seiz':
                s = max(0, int(np.floor(r['start_time']*fs)))
                e = min(n-1, int(np.ceil(r['stop_time']*fs)))
                labels[s:e+1] = 1
        for start in range(0, n-win_size+1, stride):
            window = sig[start:start+win_size]
            lab = int(np.any(labels[start:start+win_size]==1))
            if mode==1 or (mode==2 and lab==1) or mode==3:
                x = window
                y = np.array([1,0],dtype=np.float32) if lab==0 else np.array([0,1],dtype=np.float32)
                yield x, y

# --- Pipeline tf.data con balance y aumentos ---
def create_tf_dataset(edf_paths, lbl_paths, montage, fs_desired, win_size, mode, batch_size, shuffle=True):
    sig_shape = (win_size, len(BASE_PAIRS_AR if montage=='ar' else BASE_PAIRS_LE))
    ds = tf.data.Dataset.from_generator(
        lambda: windows_generator(edf_paths, lbl_paths, montage, fs_desired, win_size, mode),
        output_signature=(
            tf.TensorSpec(shape=sig_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(2,), dtype=tf.float32)
        )
    )
    if shuffle:
        ds = ds.shuffle(2048)
    # Aumentos: jitter y escala
    def augment(x, y):
        x = x + tf.random.normal(tf.shape(x), stddev=0.005)
        scale = tf.random.uniform([1,1,1], 0.9, 1.1)
        x = x * scale
        return x, y
    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# --- Modelo TCN + BiLSTM + Atención ---
def build_hybrid_model(input_shape, blocks=6, k=4, f=64, dr=0.2):
    inp = layers.Input(shape=input_shape)
    x = inp
    # Bloques TCN
    for i in range(blocks):
        d = 2**i
        c = layers.Conv1D(f, k, dilation_rate=d, padding='causal', kernel_initializer='he_normal')(x)
        c = layers.LayerNormalization()(c)
        c = layers.SpatialDropout1D(dr)(c)
        x = layers.Add()([x, c]) if x.shape[-1]==f else layers.Conv1D(f,1,padding='same')(x)
    # BiLSTM
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    # Self-Attention
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=f)(x, x)
    # Pooling y salida
    g = layers.GlobalAveragePooling1D()(attn)
    out = layers.Dense(2, activation='softmax', dtype='float32')(g)
    return models.Model(inputs=inp, outputs=out)

# --- Entrenamiento con AdamW y CosineDecay ---
def compile_and_train(model, train_ds, val_ds, class_weights, lr=1e-3, epochs=20):
    # Calcular steps para scheduler
    steps = tf.data.experimental.cardinality(train_ds).numpy()
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr, decay_steps=epochs*steps)
    opt = AdamW(learning_rate=lr_schedule, weight_decay=1e-5)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weights)

# --- Ejecución principal ---
if __name__ == "__main__":
    ROOT = "DATA_EEG_TUH/tuh_eeg_seizure/v2.0.3/edf/"
    change_label_extensions(ROOT)
    train_l, dev_l, eval_l = list_label_paths(ROOT)
    tr_l = filter_by_montage(train_l, 'ar')
    dv_l = filter_by_montage(dev_l, 'ar')
    ev_l = filter_by_montage(eval_l, 'ar')
    get_edf = lambda p: p.replace('_bi.csv', '.edf')
    tr_e = [get_edf(p) for p in tr_l][:50]
    dv_e = [get_edf(p) for p in dv_l][:50]
    ev_e = [get_edf(p) for p in ev_l][:50]

    fs = 256
    win_secs = 4
    win_size = win_secs * fs
    batch_size = 16

    count_bkg=count_seiz=0
    for _,y in windows_generator(tr_e,tr_l,'ar',fs,win_size,mode=1):
        if y[1]==1: count_seiz+=1
        else:       count_bkg+=1
    print(f"Pre-entrenamiento: {count_bkg} ventanas de fondo, {count_seiz} ventanas con seiz")

    print(f"Train files: {len(tr_e)}, Dev files: {len(dv_e)}, Eval files: {len(ev_e)}")

    cnt_b, cnt_s = 0,0
    for _,y in windows_generator(tr_e,tr_l,'ar',fs,win_size,mode=1):
        cnt_s += int(y[1]==1)
        cnt_b += int(y[1]==0)
    print(f"Clases antes: fondo={cnt_b}, seiz={cnt_s}")
    # calcular pesos balanceados
    y_all = np.concatenate([np.zeros(cnt_b), np.ones(cnt_s)])
    cw = class_weight.compute_class_weight('balanced', classes=np.array([0.,1.]), y=y_all)
    class_weights={0: float(cw[0]), 1: float(cw[1])}
    print("class_weight:", class_weights)

    # Datasets balanceados: pos y neg al 50%
    ds_pos = create_tf_dataset(tr_e, tr_l, 'ar', fs, win_size, mode=2, batch_size=batch_size)
    ds_all = create_tf_dataset(tr_e, tr_l, 'ar', fs, win_size, mode=3, batch_size=batch_size)
    train_ds = tf.data.experimental.sample_from_datasets([ds_pos, ds_all], weights=[0.5,0.5])
    val_ds = create_tf_dataset(dv_e, dv_l, 'ar', fs, win_size, mode=2, batch_size=batch_size)

    # Construir y entrenar
    model = build_hybrid_model((win_size, len(BASE_PAIRS_AR)))
    model.summary()
    history = compile_and_train(model, train_ds, val_ds, lr=1e-3, epochs=20, class_weights=class_weights)

    # Evaluación
    test_ds = create_tf_dataset(ev_e, ev_l, 'ar', fs, win_size, mode=1, batch_size=batch_size, shuffle=False)
    results = model.evaluate(test_ds)
    print("Test metrics:", dict(zip(model.metrics_names, results)))

    # Métricas detalladas
    y_true, y_pred, y_prob = [], [], []
    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch)
        preds = np.argmax(probs, axis=1)
        trues = np.argmax(y_batch.numpy(), axis=1)
        y_true.extend(trues)
        y_pred.extend(preds)
        y_prob.extend(probs[:,1])
    print(classification_report(y_true, y_pred, target_names=['bckg','seiz']))
    print("AUC:", roc_auc_score(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(); plt.plot(fpr, tpr); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.show()
