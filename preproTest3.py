# -*- coding: utf-8 -*-
"""
preproTest2_optimized.py

Optimizado para:
 - Uso de GPU con crecimiento de memoria y Mixed Precision.
 - Pipeline de tf.data para reducir uso de RAM.
 - Evaluación completa de entrenamiento y test.
 - Conservación de funcionalidad original.
"""

import os
import glob
import shutil
import numpy as np
import pandas as pd
import mne
from scipy.signal import resample
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import mixed_precision
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# --- Configuración GPU y Mixed Precision ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
#mixed_precision.set_global_policy('mixed_float16')

def change_label_extensions(root_folder: str):
    pattern = os.path.join(root_folder, '**', '*.csv_bi')
    for old_path in glob.glob(pattern, recursive=True):
        new_path = old_path.replace('.csv_bi', '_bi.csv')
        shutil.move(old_path, new_path)

def list_label_paths(root_folder: str):
    train = glob.glob(os.path.join(root_folder, 'train', '**', '*_bi.csv'), recursive=True)
    dev   = glob.glob(os.path.join(root_folder, 'dev',   '**', '*_bi.csv'), recursive=True)
    eval  = glob.glob(os.path.join(root_folder, 'eval',  '**', '*_bi.csv'), recursive=True)
    return train, dev, eval


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


def load_label_csv(path: str):
    df = pd.read_csv(path, skiprows=5)
    need = {'start_time','stop_time','label'}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} no contiene {need}")
    return df


def extract_montage_signals(edf_path: str, montage: str='ar', desired_fs: int=256):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    orig_fs = int(raw.info['sfreq'])
    data, _ = raw.get_data(picks='eeg', return_times=True)
    ch_names = raw.ch_names
    suf = '-LE' if any(c.endswith('-LE') for c in ch_names) else '-REF'
    pairs_ar = [(f'EEG FP1{suf}', f'EEG F7{suf}'), (f'EEG F7{suf}', f'EEG T3{suf}'),
                (f'EEG T3{suf}', f'EEG T5{suf}'), (f'EEG T5{suf}', f'EEG O1{suf}'),
                (f'EEG FP2{suf}', f'EEG F8{suf}'), (f'EEG F8{suf}', f'EEG T4{suf}'),
                (f'EEG T4{suf}', f'EEG T6{suf}'), (f'EEG T6{suf}', f'EEG O2{suf}'),
                (f'EEG A1{suf}',  f'EEG T3{suf}'), (f'EEG T3{suf}', f'EEG C3{suf}'),
                (f'EEG C3{suf}', f'EEG CZ{suf}'), (f'EEG CZ{suf}', f'EEG C4{suf}'),
                (f'EEG C4{suf}', f'EEG T4{suf}'), (f'EEG T4{suf}', f'EEG A2{suf}'),
                (f'EEG FP1{suf}', f'EEG F3{suf}'), (f'EEG F3{suf}', f'EEG C3{suf}'),
                (f'EEG C3{suf}', f'EEG P3{suf}'), (f'EEG P3{suf}', f'EEG O1{suf}'),
                (f'EEG FP2{suf}', f'EEG F4{suf}'), (f'EEG F4{suf}', f'EEG C4{suf}'),
                (f'EEG C4{suf}', f'EEG P4{suf}'), (f'EEG P4{suf}', f'EEG O2{suf}')]
    pairs_le = [(f'EEG F7{suf}', f'EEG F8{suf}'), (f'EEG T3{suf}', f'EEG T4{suf}'),
                (f'EEG T5{suf}', f'EEG T6{suf}'), (f'EEG C3{suf}', f'EEG C4{suf}'),
                (f'EEG P3{suf}', f'EEG P4{suf}'), (f'EEG O1{suf}', f'EEG O2{suf}')]
    pairs = pairs_ar if montage=='ar' else pairs_le
    mont = []
    for c1,c2 in pairs:
        if c1 in ch_names and c2 in ch_names:
            mont.append(data[ch_names.index(c1)] - data[ch_names.index(c2)])
        else:
            mont.append(np.zeros(data.shape[1],dtype=np.float32))
    mont = np.stack(mont, axis=1)
    maxval = np.max(np.abs(mont),axis=0)
    maxval[maxval==0]=1.0
    mont /= maxval
    if orig_fs!=desired_fs:
        n_t = int(mont.shape[0]*desired_fs/orig_fs)
        mont = resample(mont, n_t, axis=0)
    return mont.astype(np.float32), (desired_fs if orig_fs!=desired_fs else orig_fs), mont.shape[0]

# --- Generador de ventanas ---
def windows_generator(edf_paths, lbl_paths, montage, fs_desired, win_size, mode):
    for edf, lbl in zip(edf_paths, lbl_paths):
        sig, fs, n = extract_montage_signals(edf, montage, fs_desired)
        df = load_label_csv(lbl)
        labels = np.zeros(n,dtype=np.int32)
        for _,r in df.iterrows():
            if r['label'].strip().lower()=='seiz':
                s = max(0,int(np.floor(r['start_time']*fs)))
                e = min(n-1,int(np.ceil(r['stop_time']*fs)))
                labels[s:e+1]=1
        stride = win_size if mode in (1,2) else win_size//2
        for start in range(0, n-win_size+1, stride):
            window = sig[start:start+win_size]
            lab = 1 if np.any(labels[start:start+win_size]==1) else 0
            if mode==1 or (mode==2 and lab==1) or mode==3:
                x = window.astype(np.float32)
                y = np.array([1,0],dtype=np.float32) if lab==0 else np.array([0,1],dtype=np.float32)
                yield x, y

# --- Creación de tf.data.Dataset ---
def create_tf_dataset(edf_paths, lbl_paths, montage, fs_desired, win_size, mode, batch_size, shuffle=True):
    sig_shape = (win_size, 22)
    ds = tf.data.Dataset.from_generator(
        lambda: windows_generator(edf_paths, lbl_paths, montage, fs_desired, win_size, mode),
        output_signature=(
            tf.TensorSpec(shape=sig_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(2,), dtype=tf.float32)
        )
    )
    if shuffle:
        ds = ds.shuffle(1024)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- II. Modelo TCN ---
def build_tcn_eegnet(input_shape, blocks=10, k=4, f=68, dr=0.25):
    inp = layers.Input(shape=input_shape)
    x = inp
    for i in range(blocks):
        d = 2**i
        c1 = layers.Conv1D(f, k, dilation_rate=d, padding='causal', kernel_initializer='he_normal')(x)
        n1 = layers.LayerNormalization()(c1)
        d1 = layers.SpatialDropout1D(dr)(n1)
        c2 = layers.Conv1D(f, k, dilation_rate=d, padding='causal', kernel_initializer='he_normal')(d1)
        n2 = layers.LayerNormalization()(c2)
        r = layers.ReLU()(n2)
        d2 = layers.SpatialDropout1D(dr)(r)
        skip = x if x.shape[-1]==f else layers.Conv1D(f,1,padding='same',kernel_initializer='he_normal')(x)
        x = layers.Add()([skip, d2])
    g = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(2, activation='softmax', dtype='float32')(g)
    return models.Model(inp, out, name='TCN_EEGNet')

def compile_and_train(model, train_ds, val_ds, lr=1e-3, epochs=20):
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# --- III. Ejecución principal ---
if __name__=="__main__":
    ROOT = "DATA_EEG_TUH/tuh_eeg_seizure/v2.0.3/edf/"
    change_label_extensions(ROOT)
    train_l, dev_l, eval_l = list_label_paths(ROOT)
    tr_l = filter_by_montage(train_l, 'ar')
    dv_l = filter_by_montage(dev_l, 'ar')
    ev_l = filter_by_montage(eval_l, 'ar')
    get_edf = lambda p: p.replace('_bi.csv', '.edf')
    tr_e = [get_edf(p) for p in tr_l]
    dv_e = [get_edf(p) for p in dv_l]
    ev_e = [get_edf(p) for p in ev_l]

    desired_fs = 256
    win_secs   = 60
    win_size   = win_secs * desired_fs
    batch_size = 32

    # Limitar a 20 archivos para pruebas rápidas
    tr_e = tr_e[:100]
    dv_e = dv_e[:100]
    ev_e = ev_e[:100]

    # count_bkg=count_seiz=0
    # for _,y in windows_generator(tr_e,tr_l,'ar',desired_fs,win_size,mode=1):
    #     if y[1]==1: count_seiz+=1
    #     else:       count_bkg+=1
    # print(f"Pre-entrenamiento: {count_bkg} ventanas de fondo, {count_seiz} ventanas con seiz")

    print(f"Train files: {len(tr_e)}, Dev files: {len(dv_e)}, Eval files: {len(ev_e)}")
    
    train_ds = create_tf_dataset(tr_e, tr_l, 'ar', desired_fs, win_size, 1, batch_size)#.repeat()
    val_ds   = create_tf_dataset(dv_e, dv_l, 'ar', desired_fs, win_size, 2, batch_size)

    model = build_tcn_eegnet((win_size, 22))
    model.summary()

    history = compile_and_train(model, train_ds, val_ds, lr=1e-3, epochs=20)

    # --- IV. Evaluación en Test ---
    test_ds = create_tf_dataset(ev_e, ev_l, 'ar', desired_fs, win_size, 1, batch_size, shuffle=False)

    # Métricas globales
    eval_results = model.evaluate(test_ds)
    print("Resultados en Test:", dict(zip(model.metrics_names, eval_results)))

    # Predicciones y métricas detalladas
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

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.show()
