# -*- coding: utf-8 -*-
"""
tuh_pipeline.py

Reproducción FIEL del pipeline MATLAB:
  - Prototipo_Data_TUH.m  → Funciones de preprocesamiento (hasta antes de LSTM)
  - Desarrollo_Redes_CNN.m → Red TCN en TensorFlow/Keras

Para probarlo, coloca esta ruta en la variable ROOT (ver sección __main__).
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
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

########################################
# I. PREPROCESAMIENTO (Prototipo_Data_TUH)
########################################

def change_label_extensions(root_folder: str):
    """
    Renombra todos los archivos '*.csv_bi' a '*_bi.csv', recursivamente.
    Equivalente exacto de:
      dir("**/*.csv_bi") + movefile(..., replace(".csv_bi","_bi.csv"))
    """
    pattern = os.path.join(root_folder, '**', '*.csv_bi')
    for old_path in glob.glob(pattern, recursive=True):
        new_path = old_path.replace('.csv_bi', '_bi.csv')
        shutil.move(old_path, new_path)


def list_label_paths(root_folder: str):
    """
    Lista rutas de etiquetas BICLASE en carpetas 'train/', 'dev/', 'eval/'.
    Coincide con:
      dir(fullfile(root, "train","**/*_bi.csv")) etc.
    """
    train_paths = glob.glob(os.path.join(root_folder, 'train', '**', '*_bi.csv'), recursive=True)
    dev_paths   = glob.glob(os.path.join(root_folder, 'dev',   '**', '*_bi.csv'), recursive=True)
    eval_paths  = glob.glob(os.path.join(root_folder, 'eval',  '**', '*_bi.csv'), recursive=True)
    return train_paths, dev_paths, eval_paths


def filter_by_montage(label_paths, montage_type: str):
    """
    Filtra rutas por montaje:
      'ar'   → archivos con '_tcp_ar' pero NO '_tcp_ar_a'
      'ar_a' → archivos con '_tcp_ar_a'
      'le'   → archivos con '_tcp_le'
    Equivalente a: dir("**/*_tcp_ar*_bi.csv"), dir("**/*_tcp_ar_a*_bi.csv"), dir("**/*_tcp_le*_bi.csv")
    """
    filtered = []
    for path in label_paths:
        folder = os.path.dirname(path).lower()
        if montage_type == 'ar_a':
            if '_tcp_ar_a' in folder:
                filtered.append(path)
        elif montage_type == 'ar':
            if ('_tcp_ar' in folder) and ('_tcp_ar_a' not in folder):
                filtered.append(path)
        elif montage_type == 'le':
            if '_tcp_le' in folder:
                filtered.append(path)
    return filtered


def load_label_csv(label_path: str):
    """
    Carga el CSV de etiquetas (skiprows=5), que es idéntico a:
      lbl_data = readtable(...,"NumHeaderLines",5)
    Verifica columnas 'start_time','stop_time','label'.
    """
    df = pd.read_csv(label_path, skiprows=5)
    required = {'start_time', 'stop_time', 'label'}
    if not required.issubset(df.columns):
        raise ValueError(f"El CSV {label_path} no contiene {required}")
    return df


# def extract_montage_signals(edf_path: str,
#                             montage: str = 'ar',
#                             desired_fs: int = 256):
#     """
#     Lee un EDF y construye montaje biploar EXACTAMENTE como en MATLAB:
#       - Antes de restar, MATLAB usaba "SelectedSignals" = 
#         ["EEG FP1-REF",...,"EEG A2-REF"] que en tu EDF aparece como "EEG FP1-LE", ..., "EEG A2-LE".
#       - Construye los 22 pares (FP1-F7, F7-T3, …, P4-O2).
#       - Luego, si orig_fs ≠ desired_fs, hace resample(edf_montage,P,Q).

#     Retorna: montage_data [n_samples,22], fs (Hz), n_samples.
#     """
#     raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
#     orig_fs = int(raw.info['sfreq'])
#     data, _ = raw.get_data(picks='eeg', return_times=True)  # (22, n_samples)
#     ch_names = raw.ch_names  # ej. ["EEG FP1-LE", ..., "EEG A2-LE"]
    # print(f"\nCanales en {edf_path}:\n", ch_names, "\n")

    # # Definición textual de canales (MATLAB usaba sufijo '-REF'; aquí '-LE'):
    # bipolar_pairs_ar = [
    #     ('EEG FP1-LE', 'EEG F7-LE'),
    #     ('EEG F7-LE',  'EEG T3-LE'),
    #     ('EEG T3-LE',  'EEG T5-LE'),
    #     ('EEG T5-LE',  'EEG O1-LE'),
    #     ('EEG FP2-LE', 'EEG F8-LE'),
    #     ('EEG F8-LE',  'EEG T4-LE'),
    #     ('EEG T4-LE',  'EEG T6-LE'),
    #     ('EEG T6-LE',  'EEG O2-LE'),
    #     ('EEG A1-LE',  'EEG T3-LE'),
    #     ('EEG T3-LE',  'EEG C3-LE'),
    #     ('EEG C3-LE',  'EEG CZ-LE'),
    #     ('EEG CZ-LE',  'EEG C4-LE'),
    #     ('EEG C4-LE',  'EEG T4-LE'),
    #     ('EEG T4-LE',  'EEG A2-LE'),
    #     ('EEG FP1-LE', 'EEG F3-LE'),
    #     ('EEG F3-LE',  'EEG C3-LE'),
    #     ('EEG C3-LE',  'EEG P3-LE'),
    #     ('EEG P3-LE',  'EEG O1-LE'),
    #     ('EEG FP2-LE', 'EEG F4-LE'),
    #     ('EEG F4-LE',  'EEG C4-LE'),
    #     ('EEG C4-LE',  'EEG P4-LE'),
    #     ('EEG P4-LE',  'EEG O2-LE')
    # ]
    # bipolar_pairs_le = [
    #     ('EEG F7-LE', 'EEG F8-LE'),
    #     ('EEG T3-LE', 'EEG T4-LE'),
    #     ('EEG T5-LE', 'EEG T6-LE'),
    #     ('EEG C3-LE', 'EEG C4-LE'),
    #     ('EEG P3-LE', 'EEG P4-LE'),
    #     ('EEG O1-LE', 'EEG O2-LE'),
    #     # (MATLAB definía esto aunque no siempre se usaba)
    # ]

    # if montage == 'ar':
    #     bipolar_pairs = bipolar_pairs_ar
    # elif montage == 'le':
    #     bipolar_pairs = bipolar_pairs_le
    # else:
    #     raise ValueError(f"Montage desconocido '{montage}'. Debe ser 'ar' o 'le'.")

    # montage_data = []
    # for ch1, ch2 in bipolar_pairs:
    #     if (ch1 in ch_names) and (ch2 in ch_names):
    #         i1 = ch_names.index(ch1)
    #         i2 = ch_names.index(ch2)
    #         diff = data[i1, :] - data[i2, :]
    #     else:
    #         # Cubre caso en que algún canal falte: rellenar con ceros.
    #         n_s = data.shape[1]
    #         diff = np.zeros(n_s, dtype=np.float32)
    #     montage_data.append(diff)

    # montage_data = np.stack(montage_data, axis=1)  # (n_samples_orig, 22)

    # # Re-muestreo si orig_fs ≠ desired_fs (igual que [P,Q]=rat(...) + resample en MATLAB)
    # if orig_fs != desired_fs:
    #     n_target = int(montage_data.shape[0] * desired_fs / orig_fs)
    #     montage_data = resample(montage_data, n_target, axis=0)
    #     fs = desired_fs
    # else:
    #     fs = orig_fs

    # n_samples = montage_data.shape[0]
    # return montage_data, fs, n_samples

def extract_montage_signals(edf_path: str,
                            montage: str = 'ar',
                            desired_fs: int = 256):
    """
    Lee un EDF y construye el montaje bipolar:
      - Detecta si los canales en raw.ch_names acaban en '-LE' o en '-REF'.
      - Usa esa misma terminación para armar los 22 pares bipolares (montage 'ar').
      - Re-muestrea a desired_fs si orig_fs != desired_fs.

    Retorna:
      montage_data: np.ndarray de forma (n_samples, n_bipolar)
      fs: frecuencia de muestreo resultante
      n_samples: número de muestras tras re-muestreo
    """

    # 1) Leer el EDF con MNE
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    orig_fs = int(raw.info['sfreq'])
    data, _ = raw.get_data(picks='eeg', return_times=True)  # (n_canales, n_muestras)
    ch_names = raw.ch_names

    # 2) Para debug/fijarnos una sola vez, imprimimos los canales
    #    (luego puedes comentar esta línea si no quieres verlo cada vez)
    # print(f"\n>> Canales en {edf_path}:\n   {ch_names}\n")

    # 3) Detectar automáticamente el sufijo real: '-LE' o '-REF'
    if any(name.endswith('-LE') for name in ch_names):
        suf = '-LE'
    elif any(name.endswith('-REF') for name in ch_names):
        suf = '-REF'
    else:
        raise RuntimeError("No se detectó sufijo '-LE' ni '-REF' en raw.ch_names.")

    # 4) Definir los 22 pares bipolares para montaje 'ar' usando el sufijo detectado
    bipolar_pairs_ar = [
        (f'EEG FP1{suf}', f'EEG F7{suf}'),
        (f'EEG F7{suf}',  f'EEG T3{suf}'),
        (f'EEG T3{suf}',  f'EEG T5{suf}'),
        (f'EEG T5{suf}',  f'EEG O1{suf}'),
        (f'EEG FP2{suf}', f'EEG F8{suf}'),
        (f'EEG F8{suf}',  f'EEG T4{suf}'),
        (f'EEG T4{suf}',  f'EEG T6{suf}'),
        (f'EEG T6{suf}',  f'EEG O2{suf}'),
        (f'EEG A1{suf}',  f'EEG T3{suf}'),
        (f'EEG T3{suf}',  f'EEG C3{suf}'),
        (f'EEG C3{suf}',  f'EEG CZ{suf}'),
        (f'EEG CZ{suf}',  f'EEG C4{suf}'),
        (f'EEG C4{suf}',  f'EEG T4{suf}'),
        (f'EEG T4{suf}',  f'EEG A2{suf}'),
        (f'EEG FP1{suf}', f'EEG F3{suf}'),
        (f'EEG F3{suf}',  f'EEG C3{suf}'),
        (f'EEG C3{suf}',  f'EEG P3{suf}'),
        (f'EEG P3{suf}',  f'EEG O1{suf}'),
        (f'EEG FP2{suf}', f'EEG F4{suf}'),
        (f'EEG F4{suf}',  f'EEG C4{suf}'),
        (f'EEG C4{suf}',  f'EEG P4{suf}'),
        (f'EEG P4{suf}',  f'EEG O2{suf}')
    ]

    # 5) (Opcional) Si quisieras montaje 'le', podrías leer otros pares; 
    #    aquí solo dejo los AR porque el pipeline usa ese por defecto.
    bipolar_pairs_le = [
        (f'EEG F7{suf}',  f'EEG F8{suf}'),
        (f'EEG T3{suf}',  f'EEG T4{suf}'),
        (f'EEG T5{suf}',  f'EEG T6{suf}'),
        (f'EEG C3{suf}',  f'EEG C4{suf}'),
        (f'EEG P3{suf}',  f'EEG P4{suf}'),
        (f'EEG O1{suf}',  f'EEG O2{suf}')
    ]

    if montage == 'ar':
        bipolar_pairs = bipolar_pairs_ar
    elif montage == 'le':
        bipolar_pairs = bipolar_pairs_le
    else:
        raise ValueError(f"Montage desconocido: {montage} (debe ser 'ar' o 'le')")

    # 6) Construir la matriz bipolar (diff = canal1 – canal2)
    montage_data = []
    for ch1, ch2 in bipolar_pairs:
        if (ch1 in ch_names) and (ch2 in ch_names):
            i1 = ch_names.index(ch1)
            i2 = ch_names.index(ch2)
            diff = data[i1, :] - data[i2, :]
        else:
            # Si alguno de los canales no se encuentra, rellenar con ceros
            n_s = data.shape[1]
            diff = np.zeros(n_s, dtype=np.float32)
        montage_data.append(diff)

    montage_data = np.stack(montage_data, axis=1)  # shape = (n_samples_orig, 22)

    # 6) Normalizar cada canal bipolar a [-1, 1]
    #    Para cada columna j: divide por max(abs(columna))
    #    Si el máximo es 0, dejar todos ceros.
    abs_max = np.max(np.abs(montage_data), axis=0)  # shape = (22,)
    # Evitar división por cero
    abs_max[abs_max == 0] = 1.0
    montage_data = montage_data / abs_max  # broadcasting → cada columna en [-1,1]

    # 7) Re-muestrear a desired_fs si es necesario
    if orig_fs != desired_fs:
        n_target = int(montage_data.shape[0] * desired_fs / orig_fs)
        montage_data = resample(montage_data, n_target, axis=0)
        fs = desired_fs
    else:
        fs = orig_fs

    n_samples = montage_data.shape[0]
    return montage_data, fs, n_samples



def generate_labels_vector(label_df: pd.DataFrame, fs: int, n_samples: int):
    """
    De cada fila en label_df con 'label' == 'seiz':
       start_idx = floor(start_time * fs)
       stop_idx  = ceil(stop_time  * fs)
    Asigna 1 (seiz) a ese rango. Resto queda 0 (background).
    Equivale con:
       strt_idx = ceil(strt_lbl * Fs)+1; stop_idx = min(edf_samples, ceil(stop_lbl * Fs)+1)
    en MATLAB 1-based.
    """
    labels = np.zeros(n_samples, dtype=np.int32)
    for _, row in label_df.iterrows():
        lab = row['label'].strip().lower()
        if lab == 'seiz':
            start_idx = int(np.floor(row['start_time'] * fs))
            stop_idx = int(np.ceil(row['stop_time'] * fs))
            if start_idx < 0:
                start_idx = 0
            if stop_idx >= n_samples:
                stop_idx = n_samples - 1
            labels[start_idx : stop_idx + 1] = 1
    return labels


def process_single_study(edf_path: str, label_path: str, montage: str = 'ar', desired_fs: int = 256):
    """
    Procesa un único estudio:
      1) extract_montage_signals(...) → (montage_data, fs, n_samples)
      2) load_label_csv(...) → DataFrame
      3) generate_labels_vector(...) → vector de 0/1 de largo n_samples
    Retorna: (montage_data [n_samples,22], labels_vec [n_samples], fs)
    """
    montage_data, fs, n_samples = extract_montage_signals(edf_path, montage, desired_fs)
    df_labels = load_label_csv(label_path)
    labels_vec = generate_labels_vector(df_labels, fs, n_samples)
    return montage_data, labels_vec, fs


# def segment_windows(signals: np.ndarray, labels: np.ndarray, window_size: int, mode: int = 2):
#     """
#     Divide señales y etiquetas en ventanas no solapadas de tamaño window_size.
#     Si n_windows = floor(n_samples/window_size) = 0, devuelve arrays vacíos de forma
#     (0, window_size, 22) y (0, 2), tal como en MATLAB pre-asignar zeros(0,7680,22).

#     Para cada ventana:
#        - win_sig = signals[i*window_size:(i+1)*window_size, :]
#        - win_lab = labels[i*window_size:(i+1)*window_size]
#        - if any(win_lab==1) → etiqueta [0,1], else [1,0]
#     """
#     n_samples, n_channels = signals.shape
#     n_windows = n_samples // window_size

#     if n_windows == 0:
#         X_empty = np.zeros((0, window_size, n_channels), dtype=signals.dtype)
#         y_empty = np.zeros((0, 2), dtype=np.int32)
#         return X_empty, y_empty

#     signals_cut = signals[: n_windows * window_size, :]
#     labels_cut  = labels[:  n_windows * window_size]

#     signals_reshaped = signals_cut.reshape(n_windows, window_size, n_channels)
#     labels_reshaped  = labels_cut.reshape(n_windows, window_size)

#     X_list = []
#     y_list = []
#     # for i in range(n_windows):
#     #     win_sig = signals_reshaped[i]
#     #     win_lab = labels_reshaped[i]
#     #     if np.any(win_lab == 1):
#     #         one_hot = np.array([0, 1], dtype=np.int32)
#     #     else:
#     #         one_hot = np.array([1, 0], dtype=np.int32)
#     #     X_list.append(win_sig)
#     #     y_list.append(one_hot)
#     for i in range(n_windows):
#         win_sig = signals_reshaped[i]   # shape = (window_size, n_channels)
#         win_lab = labels_reshaped[i]    # shape = (window_size,)

#         # Determinar etiqueta de la ventana: 1 si hay al menos un '1' en win_lab
#         window_label = 1 if np.any(win_lab == 1) else 0

#         if mode == 1:
#             # Incluimos todas las ventanas
#             if window_label == 0:
#                 one_hot = np.array([1, 0], dtype=np.int32)  # bckg
#             else:
#                 one_hot = np.array([0, 1], dtype=np.int32)  # seiz
#             X_list.append(win_sig)
#             y_list.append(one_hot)

#         elif mode == 2:
#             # Solo incluimos las ventanas que tengan al menos una muestra seiz (window_label == 1)
#             if window_label == 1:
#                 one_hot = np.array([0, 1], dtype=np.int32)
#                 X_list.append(win_sig)
#                 y_list.append(one_hot)
#             # si window_label==0, no la agregamos

#         else:
#             raise ValueError(f"Mode desconocido: {mode}. Debe ser 1 o 2.")

#     # Si no quedó ninguna ventana (p. ej. en mode=2 y no había crisis),
#     # devolvemos arrays vacíos con la forma correcta
#     if len(X_list) == 0:
#         X_empty = np.zeros((0, window_size, n_channels), dtype=signals.dtype)
#         y_empty = np.zeros((0, 2), dtype=np.int32)
#         return X_empty, y_empty


#     X_windows = np.stack(X_list, axis=0)
#     y_windows = np.stack(y_list, axis=0)
#     return X_windows, y_windows

import numpy as np

def segment_windows(signals: np.ndarray, labels: np.ndarray, window_size: int, mode: int = 1):
    """
    Divide señales y etiquetas en ventanas de tamaño window_size, con tres modos:
    
    Args:
      signals    : ndarray float32, forma (n_samples, n_channels)
      labels     : ndarray int32, forma (n_samples,), valores 0=bckg, 1=seiz
      window_size: int, número de muestras por ventana (ej. 7680 para 30 s × 256 Hz)
      mode       : int
                   1 → ventanas no solapadas, todas (bckg y seiz)
                   2 → ventanas no solapadas, sólo aquellas con al menos un ‘seiz’
                   3 → ventanas con solapamiento del 50 %, todas (bckg y seiz)

    Retorna:
      X_windows: ndarray float32 con forma (n_sel_windows, window_size, n_channels)
      y_windows: ndarray int32 con forma (n_sel_windows, 2), one-hot [1,0]=bckg, [0,1]=seiz
    """
    n_samples, n_channels = signals.shape

    if mode in (1, 2):
        # Mismo comportamiento original: no solapamiento
        n_windows = n_samples // window_size
        if n_windows == 0:
            return (np.zeros((0, window_size, n_channels), dtype=signals.dtype),
                    np.zeros((0, 2), dtype=np.int32))

        signals_cut = signals[: n_windows * window_size, :]
        labels_cut  = labels[:  n_windows * window_size]

        signals_reshaped = signals_cut.reshape(n_windows, window_size, n_channels)
        labels_reshaped  = labels_cut.reshape(n_windows, window_size)

        X_list = []
        y_list = []
        for i in range(n_windows):
            win_sig = signals_reshaped[i]
            win_lab = labels_reshaped[i]
            window_label = 1 if np.any(win_lab == 1) else 0

            if mode == 1:
                # Incluir todas las ventanas
                if window_label == 0:
                    one_hot = np.array([1, 0], dtype=np.int32)  # bckg
                else:
                    one_hot = np.array([0, 1], dtype=np.int32)  # seiz
                X_list.append(win_sig)
                y_list.append(one_hot)

            else:  # mode == 2
                # Solo incluir ventanas con al menos un 'seiz'
                if window_label == 1:
                    one_hot = np.array([0, 1], dtype=np.int32)
                    X_list.append(win_sig)
                    y_list.append(one_hot)

        if len(X_list) == 0:
            return (np.zeros((0, window_size, n_channels), dtype=signals.dtype),
                    np.zeros((0, 2), dtype=np.int32))

        X_windows = np.stack(X_list, axis=0)
        y_windows = np.stack(y_list, axis=0)
        return X_windows, y_windows

    elif mode == 3:
        # Solapamiento al 50 %: stride = window_size // 2
        stride = window_size // 2
        X_list = []
        y_list = []

        # Solo tomamos ventanas completas (start + window_size <= n_samples)
        for start in range(0, n_samples - window_size + 1, stride):
            end = start + window_size
            win_sig = signals[start:end, :]
            win_lab = labels[start:end]
            window_label = 1 if np.any(win_lab == 1) else 0

            if window_label == 0:
                one_hot = np.array([1, 0], dtype=np.int32)
            else:
                one_hot = np.array([0, 1], dtype=np.int32)

            X_list.append(win_sig)
            y_list.append(one_hot)

        if len(X_list) == 0:
            return (np.zeros((0, window_size, n_channels), dtype=signals.dtype),
                    np.zeros((0, 2), dtype=np.int32))

        X_windows = np.stack(X_list, axis=0)
        y_windows = np.stack(y_list, axis=0)
        return X_windows, y_windows

    else:
        raise ValueError(f"Mode desconocido: {mode}. Debe ser 1, 2 o 3.")

########################################
# II. RED NEURONAL TCN (Desarrollo_Redes_CNN)
########################################

def build_tcn_eegnet(input_shape,
                     num_blocks: int = 10,
                     filter_size: int = 4,
                     num_filters: int = 68,
                     dropout_rate: float = 0.25):
    """
    Construye la red TCN exactamente igual que en MATLAB:
      sequenceInputLayer(22) →
      Por i=1..10:
        Conv1D(filters=68, kernel_size=4, dilation_rate=2^(i-1), padding='causal', kernel_initializer='he_normal')
        LayerNormalization
        SpatialDropout1D(0.25)
        Conv1D(filters=68, kernel_size=4, dilation_rate=2^(i-1), padding='causal', kernel_initializer='he_normal')
        LayerNormalization
        ReLU
        SpatialDropout1D(0.25)
        Residual (si inputDim≠68, Conv1D(1,68) para igualar dimensiones) + Add
      Al final: GlobalAveragePooling1D → Dense(2, softmax).
    """
    inputs = layers.Input(shape=input_shape, name='input')
    x = inputs

    for i in range(num_blocks):
        dilation_rate = 2 ** i
        block_id = i + 1

        # 1ª convolución dilatada
        conv1 = layers.Conv1D(
            filters=num_filters,
            kernel_size=filter_size,
            dilation_rate=dilation_rate,
            padding='causal',
            kernel_initializer='he_normal',
            name=f'conv1_block{block_id}'
        )(x)
        norm1 = layers.LayerNormalization(name=f'norm1_block{block_id}')(conv1)
        drop1 = layers.SpatialDropout1D(rate=dropout_rate, name=f'drop1_block{block_id}')(norm1)

        # 2ª convolución dilatada
        conv2 = layers.Conv1D(
            filters=num_filters,
            kernel_size=filter_size,
            dilation_rate=dilation_rate,
            padding='causal',
            kernel_initializer='he_normal',
            name=f'conv2_block{block_id}'
        )(drop1)
        norm2 = layers.LayerNormalization(name=f'norm2_block{block_id}')(conv2)
        relu = layers.ReLU(name=f'relu_block{block_id}')(norm2)
        drop2 = layers.SpatialDropout1D(rate=dropout_rate, name=f'drop2_block{block_id}')(relu)

        # Residual: si input channels ≠ num_filters, proyectar con Conv1D(1,num_filters)
        if x.shape[-1] != num_filters:
            x_proj = layers.Conv1D(
                filters=num_filters,
                kernel_size=1,
                padding='same',
                kernel_initializer='he_normal',
                name=f'convSkip_block{block_id}'
            )(x)
        else:
            x_proj = x

        # Suma residual
        out_block = layers.Add(name=f'add_block{block_id}')([x_proj, drop2])
        x = out_block
    gap = layers.GlobalAveragePooling1D(name='global_pool')(x) #TimeDistributed
    outputs = layers.Dense(2, activation='softmax', name='output')(gap)

    model = models.Model(inputs=inputs, outputs=outputs, name='TCN_EEGNet')
    return model


def compile_and_train(model,
                      train_dataset: tf.data.Dataset,
                      val_dataset: tf.data.Dataset,
                      learning_rate: float = 1e-3,
                      epochs: int = 20,
                      batch_size: int = 32,
                      class_weight: dict = None):
    """
    Compila con Adam(learning_rate), categorical_crossentropy y accuracy,
    y entrena en 'epochs' épocas con mini-batch de tamaño batch_size,
    validando en val_dataset. Si se provee class_weight, se pasa a model.fit.

    Args:
      model:         tf.keras.Model
      train_dataset: tf.data.Dataset que genera (X_train, y_train)
      val_dataset:   tf.data.Dataset que genera (X_val, y_val)
      learning_rate: float, tasa de aprendizaje de Adam
      epochs:        int, número de épocas a entrenar
      batch_size:    int, tamaño de mini-batch
      class_weight:  dict {clase: peso}, p.ej. {0:1.0, 1:20.0}

    Retorna:
      history: objeto History de Keras con registros de entrenamiento
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', #binary_crossentropy
        metrics=['accuracy']
    )

    history = model.fit(
        train_dataset.shuffle(1000).batch(batch_size),
        validation_data=val_dataset.batch(batch_size),
        epochs=epochs,
        class_weight=class_weight
    )
    return history

def segment_windows_overlap(signals, labels, window_size, stride):
    n_samples, n_channels = signals.shape
    X_list, y_list = [], []
    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        win_sig = signals[start:end, :]
        win_lab = labels[start:end]
        window_label = 1 if np.any(win_lab == 1) else 0
        one_hot = np.array([0,1], dtype=np.int32) if window_label else np.array([1,0], dtype=np.int32)
        X_list.append(win_sig)
        y_list.append(one_hot)
    if not X_list:
        return np.zeros((0, window_size, n_channels)), np.zeros((0, 2), dtype=np.int32)
    return np.stack(X_list, axis=0), np.stack(y_list, axis=0)

########################################
# III. EJEMPLO DE USO (equivalente al main de MATLAB)
########################################

if __name__ == "__main__":
    # 1) Definir carpeta raíz que contiene 'train/', 'dev/', 'eval/'
    ROOT = "DATA_EEG_TUH/tuh_eeg_seizure/v2.0.3/edf/"
    

    # 1.1) Cambiar extensiones .csv_bi → _bi.csv
    change_label_extensions(ROOT)

    # 1.2) Listar rutas de etiquetas en train/dev/eval
    train_lbls, dev_lbls, eval_lbls = list_label_paths(ROOT)

    # 1.3) Filtrar solo montaje 'ar' (como en dirLbl_train_ar = dir("**/*_tcp_ar*_bi.csv"))
    train_lbls_ar = filter_by_montage(train_lbls, 'ar')
    dev_lbls_ar   = filter_by_montage(dev_lbls,   'ar')
    eval_lbls_ar  = filter_by_montage(eval_lbls,  'ar')

    # 1.4) Obtener rutas EDF reemplazando "_bi.csv" por ".edf" (igual que en MATLAB: edfName = [erase(lblName,'_bi.csv'),'.edf'])
    get_edf_path = lambda lbl: lbl.replace('_bi.csv', '.edf')
    train_edfs = [get_edf_path(p) for p in train_lbls_ar]
    dev_edfs   = [get_edf_path(p) for p in dev_lbls_ar]
    eval_edfs  = [get_edf_path(p) for p in eval_lbls_ar]
    
    train_edfs = train_edfs[:200]  # Limitar a 10 para pruebas rápidas
    dev_edfs   = dev_edfs[:200]      # Limitar a 5 para pruebas rápidas
    eval_edfs  = eval_edfs[:200]     # Limitar a 5 para pruebas rápidas
    print(f"Train EDFs: {len(train_edfs)}")
    print(f"Dev EDFs: {len(dev_edfs)}")
    print(f"Eval EDFs: {len(eval_edfs)}")

    # 1.5) Definir parámetros de ventana (tal como en MATLAB: win_sec=30; window_size=30*256)
    window_secs = 60
    desired_fs  = 256
    window_size = window_secs * desired_fs  # 7680 muestras

    # 2) Preprocesar TODOS los estudios en TRAIN/DEV/EVAL y formar X,y por ventana
    X_train_list, y_train_list = [], []
    X_dev_list,   y_dev_list   = [], []
    X_eval_list,  y_eval_list  = [], []

    # 2.a) TRAIN
    for edf_path, lbl_path in zip(train_edfs, train_lbls_ar):
        sig, labs, fs = process_single_study(edf_path, lbl_path, montage='ar', desired_fs=desired_fs)
        Xw, yw = segment_windows(sig, labs, window_size, mode=1)
        if Xw.shape[0] == 0:
            # Si no hay ventanas completas → omitir, tal como en MATLAB no habría n_windows>0
            continue
        X_train_list.append(Xw)
        y_train_list.append(yw)

    if X_train_list:
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
    else:
        X_train = np.zeros((0, window_size, 22))
        y_train = np.zeros((0, 2))

    X_dev_seiz_list = []
    y_dev_seiz_list = []
    X_dev_bckg_list = []
    y_dev_bckg_list = []
    # 2.b) DEV (validación)
    for edf_path, lbl_path in zip(dev_edfs, dev_lbls_ar):
        sig, labs, fs = process_single_study(edf_path, lbl_path, montage='ar', desired_fs=desired_fs)
        # stride_secs = window_secs // 8                  # si window_secs = 60 s, entonces stride_secs = 15 s
        # stride = stride_secs * desired_fs 
        # Xw, yw = segment_windows_overlap(sig, labs, window_size, stride)
        #Xw, yw = segment_windows(sig, labs, window_size, mode=3)
        # --- PARTE A: ventanas NO solapadas que contengan al menos "seiz" (modo 2) ---
        X_mode2, y_mode2 = segment_windows(sig, labs, window_size, mode=2)
        #   - X_mode2 tiene sólo ventanas (window_size x 22) con al menos un 'seiz'
        if X_mode2.shape[0] > 0:
            X_dev_seiz_list.append(X_mode2)
            y_dev_seiz_list.append(y_mode2)

        # --- PARTE B: ventanas solapadas, luego filtrar sólo las que contienen "seiz" ---
        # Defin
        # if Xw.shape[0] == 0:
        #     continue
        # X_dev_list.append(Xw)
        # y_dev_list.append(yw)
    # --- PARTE B: ventanas solapadas, luego filtrar sólo las que contienen "seiz" ---
        # Definimos stride para solapamiento del 75 %
        # (por ejemplo, si window_secs=60, stride_secs=window_secs//4 = 15 s)
        stride_secs = window_secs // 4
        stride = stride_secs * desired_fs  # ej. 15 s × 256 Hz = 3840 muestras

        # Generar todas las ventanas solapadas (modo 3 con stride = window_size//4)
        X_ov_all, y_ov_all = segment_windows_overlap(sig, labs, window_size, stride)
        #   - y_ov_all ya está en one-hot [1,0]=bckg, [0,1]=seiz
        if X_ov_all.shape[0] > 0:
            # Filtrar únicamente las que tienen etiqueta “seiz”:
            seiz_indices = np.where(np.argmax(y_ov_all, axis=1) == 1)[0]
            if seiz_indices.size > 0:
                X_ov_seiz = X_ov_all[seiz_indices]
                y_ov_seiz = y_ov_all[seiz_indices]
                X_dev_seiz_list.append(X_ov_seiz)
                y_dev_seiz_list.append(y_ov_seiz)

        # -------------------------------------------------------------
        # PARTE C: Ventanas NO solapadas de fondo (modo 1), para balancear
        # -------------------------------------------------------------
        X_mode1, y_mode1 = segment_windows(sig, labs, window_size, mode=1)
        if X_mode1.shape[0] > 0:
            # Filtramos solo las de fondo: etiqueta [1,0] (bckg)
            bckg_indices = np.where(np.argmax(y_mode1, axis=1) == 0)[0]
            if bckg_indices.size > 0:
                X_bckg = X_mode1[bckg_indices]
                y_bckg = y_mode1[bckg_indices]
                X_dev_bckg_list.append(X_bckg)
                y_dev_bckg_list.append(y_bckg)
    #     # De esta forma, capturamos ventanas solapadas que tengan al menos un 'seiz'.
    # if X_dev_list:
    #     X_dev = np.concatenate(X_dev_list, axis=0)
    #     y_dev = np.concatenate(y_dev_list, axis=0)
    # else:
    #     X_dev = np.zeros((0, window_size, 22))
    #     y_dev = np.zeros((0, 2))
    # 2.b.1) Concatenar todas las sublistas de ventanas “seiz”
    if X_dev_seiz_list:
        X_dev_seiz = np.concatenate(X_dev_seiz_list, axis=0)
        y_dev_seiz = np.concatenate(y_dev_seiz_list, axis=0)
    else:
        X_dev_seiz = np.zeros((0, window_size, 22))
        y_dev_seiz = np.zeros((0, 2))

    # 2.b.2) Concatenar todas las sublistas de ventanas “bckg”
    if X_dev_bckg_list:
        X_dev_bckg = np.concatenate(X_dev_bckg_list, axis=0)
        y_dev_bckg = np.concatenate(y_dev_bckg_list, axis=0)
    else:
        X_dev_bckg = np.zeros((0, window_size, 22))
        y_dev_bckg = np.zeros((0, 2))

    def unique_rows(arr):
        """
        Elimina filas duplicadas de un ndarray de 3d (n_windows, window_size, n_channels)
        devolviendo solo filas únicas.
        """
        # Aplanamos cada ventana a 1d para poder usar np.unique
        n_w, w_s, n_ch = arr.shape
        flat = arr.reshape((n_w, w_s * n_ch))
        # np.unique sobre filas (axis=0) devuelve índices de únicas
        uniq_flat, idx = np.unique(flat, axis=0, return_index=True)
        return idx  # índices de las filas únicas
    if X_dev_seiz.shape[0] > 0:
        unique_idx = unique_rows(X_dev_seiz)
        X_dev_seiz = X_dev_seiz[unique_idx]
        y_dev_seiz = y_dev_seiz[unique_idx]

    # 2.b.4) Ahora tenemos:
    #         - X_dev_seiz (solo ventanas con seiz), 
    #         - X_dev_bckg (solo ventanas de fondo).
    #       Podemos seleccionar cuántas de fondo queremos para balancear. Por ejemplo:
    n_seiz = X_dev_seiz.shape[0]
    n_bckg = X_dev_bckg.shape[0]

    if n_bckg > 0:
        # Queremos como máximo el mismo número de ventanas bckg que de seiz,
        # para que el conjunto quede 50/50. Si hay menos bckg que seiz, tomamos todas las bckg.
        n_to_take = min(n_seiz, n_bckg)
        # Muestreamos aleatoriamente:
        chosen_bckg = np.random.choice(n_bckg, size=n_to_take, replace=False)
        X_dev_bckg_sel = X_dev_bckg[chosen_bckg]
        y_dev_bckg_sel = y_dev_bckg[chosen_bckg]
    else:
        # No hay ventanas de fondo → dejamos arrays vacíos
        X_dev_bckg_sel = np.zeros((0, window_size, 22))
        y_dev_bckg_sel = np.zeros((0, 2))
    # Obtener índices únicos (en caso de que una ventana solapada coincida exactamente
    # con alguna ventana de modo2; esto eliminará duplicados exactos)
    # unique_idx = unique_rows(X_dev_seiz)
    # X_dev = X_dev_seiz[unique_idx]
    # y_dev = y_dev_seiz[unique_idx]
    # 2.b.5) Concatenar “seiz” y “bckg selec.” para formar el DEV definitivo
    X_dev = np.concatenate([X_dev_seiz, X_dev_bckg_sel], axis=0)
    y_dev = np.concatenate([y_dev_seiz, y_dev_bckg_sel], axis=0)
    if X_dev.shape[0] > 0:
        perm = np.random.permutation(X_dev.shape[0])
        X_dev = X_dev[perm]
        y_dev = y_dev[perm]

    

    # 2.c) EVAL (evaluación final)
    for edf_path, lbl_path in zip(eval_edfs, eval_lbls_ar):
        sig, labs, fs = process_single_study(edf_path, lbl_path, montage='ar', desired_fs=desired_fs)
        print(f"Procesando {edf_path} con {lbl_path} → fs={fs}, n_samples={sig.shape[0]}")
        Xw, yw = segment_windows(sig, labs, window_size, mode=1)
        # stride_secs = window_secs // 8                  # si window_secs = 60 s, entonces stride_secs = 15 s
        # stride = stride_secs * desired_fs 
        # Xw, yw = segment_windows_overlap(sig, labs, window_size, stride)
        # print(f"Ventanas: {Xw}")
        if Xw.shape[0] == 0:
            continue
        X_eval_list.append(Xw)
        y_eval_list.append(yw)

    if X_eval_list:
        X_eval = np.concatenate(X_eval_list, axis=0)
        y_eval = np.concatenate(y_eval_list, axis=0)
    else:
        X_eval = np.zeros((0, window_size, 22))
        y_eval = np.zeros((0, 2))


    labels_dev = np.argmax(y_dev, axis=1)  # 0=bckg, 1=seiz
    unique, counts = np.unique(labels_dev, return_counts=True)
    print("Validación — Ventanas por etiqueta:")
    for lab, cnt in zip(unique, counts):
        nombre = "bckg" if lab == 0 else "seiz"
        print(f"  {nombre}: {cnt} ventanas ({cnt/len(labels_dev)*100:.1f} %)")

    # 3) Construir red TCN/EegNet idéntica a MATLAB
    #   Nota: input_shape = (window_size, 22)
    n_channels = X_train.shape[-1]
    model = build_tcn_eegnet(
        input_shape=(window_size, n_channels),
        num_blocks=10,
        filter_size=4,
        num_filters=68,
        dropout_rate=0.25
    )
    model.summary()  # Mostrar arquitectura (opcional)

    # 4) Preparar tf.data.Dataset (equivalente a gather + tall + cellfun(dlarray) en MATLAB)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dev_ds   = tf.data.Dataset.from_tensor_slices((X_dev,   y_dev))

    # 5) Entrenar (idéntico a trainNetwork(...) con opciones Adam, 20 épocas, batch 32)
    history = compile_and_train(
        model,
        train_dataset=train_ds,
        val_dataset=dev_ds,
        learning_rate=1e-3,
        epochs=20,
        batch_size=32,
        class_weight = {0: 1.0, 1: (95.7 / 4.3)}
    )

    # 6) Evaluar en EVAL y guardar modelo
    eval_ds = tf.data.Dataset.from_tensor_slices((X_eval, y_eval)).batch(32)
    eval_metrics = model.evaluate(eval_ds)
    print(f"Resultados en conjunto de evaluación: {dict(zip(model.metrics_names, eval_metrics))}")

    model.save('tcn_eegnet_tuh_v2.h5')
    print("Modelo guardado en 'tcn_eegnet_tuh_v2.h5'.")

    # Genera predicciones sobre X_dev
    pred_probs = model.predict(X_dev, batch_size=32)
    pred_labels = np.argmax(pred_probs, axis=1)
    true_labels = np.argmax(y_dev, axis=1)

    print(classification_report(true_labels, pred_labels, target_names=['bckg','seiz']))

    auc = roc_auc_score(true_labels, pred_probs[:,1])
    print("AUC (seiz vs bckg) en validación:", auc)

        # (1) Definición de rutas
    edf_test_path = "DATA_EEG_TUH/tuh_eeg_seizure/v2.0.3/edf/train/aaaaatvr/s001_2015/01_tcp_ar/aaaaatvr_s001_t000.edf"
    csv_test_path = "DATA_EEG_TUH/tuh_eeg_seizure/v2.0.3/edf/train/aaaaatvr/s001_2015/01_tcp_ar/aaaaatvr_s001_t000_bi.csv"  # Si no lo tienes, omite las partes de “ground truth”

    # (2) Parámetros
    desired_fs  = 256
    window_secs = 60
    window_size = desired_fs * window_secs  # 7680 muestras

    # (3) Cargar y preprocesar EDF
    #    extract_montage_signals devuelve: (montage_data, fs, n_samples)
    montage_data, fs, n_samples = extract_montage_signals(
        edf_path=edf_test_path,
        montage='ar',         # mismo tipo de montaje que entrenaste
        desired_fs=desired_fs
    )

    # (4) Segmentar en ventanas
    X_windows, _ = segment_windows(montage_data,
                                # no necesitamos las etiquetas aquí → pasamos un vector vacío
                                labels=np.zeros(n_samples, dtype=np.int32),
                                window_size=window_size,
                                mode=1)

    # Si X_windows está vacío, la grabación es <30 s
    if X_windows.shape[0] == 0:
        print("La grabación es más corta de 30 s. No se generaron ventanas completas.")
        exit()

    # # (5) Cargar modelo preentrenado
    # model = tf.keras.models.load_model("tcn_eegnet_tuh_v2.h5")

    # (6) Predecir: obtiene array de shape (n_windows, 2) con [P(bckg), P(seiz)]
    print("Input shape del modelo:", model.input_shape)
    pred_probs = model.predict(X_windows, batch_size=16)  # batch_size opcional

    # (7) Convertir a etiquetas binarias
    seq_labels = np.argmax(pred_probs, axis=1)  # 0=background, 1=seiz
    probs_seiz = pred_probs[:, 1]               # probabilidad de “seiz” en cada ventana

    # (8) Si tienes CSV de ground-truth, cargar y comparar
    try:
        df_labels = load_label_csv(csv_test_path)
        labels_true = generate_labels_vector(df_labels, fs, n_samples)
        # Comparar ventana a ventana:
        n_windows = n_samples // window_size
        tp = fp = tn = fn = 0
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx   = (i+1) * window_size
            gt_seiz = int(np.any(labels_true[start_idx:end_idx] == 1))
            pred_seiz = int(seq_labels[i] == 1)

            if gt_seiz == 1 and pred_seiz == 1:
                tp += 1
            elif gt_seiz == 0 and pred_seiz == 0:
                tn += 1
            elif gt_seiz == 0 and pred_seiz == 1:
                fp += 1
            elif gt_seiz == 1 and pred_seiz == 0:
                fn += 1

        print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"Precisión (seiz): {precision:.3f}, Recall (seiz): {recall:.3f}, F1 (seiz): {f1:.3f}")

    except FileNotFoundError:
        print("No se encontró CSV de etiquetas. Solo haré inferencia sin comparar con ground-truth.")

    # (9) Imprimir intervalos donde se predijo “seiz”
    n_windows = X_windows.shape[0]
    for i in range(n_windows):
        if seq_labels[i] == 1:
            t0 = i * window_secs
            t1 = (i + 1) * window_secs
            print(f"[VENTANA {i}] Predicción: SEIZ entre {t0:.1f}s y {t1:.1f}s  (P={probs_seiz[i]:.3f})")

    def plot_predictions_vs_true(edf_path, csv_path, model, montage='ar', desired_fs=256, window_secs=60):
        """
        Carga un EDF y su CSV de etiquetas. Preprocesa, segmenta en ventanas, 
        predice con el modelo .h5 y grafica en un solo plot la etiqueta real vs predicha.

        Args:
            edf_path (str): Ruta al archivo .edf a probar.
            csv_path (str): Ruta al archivo de etiquetas CSV (_bi.csv).
            model_path (str): Ruta al modelo Keras guardado (.h5).
            montage (str): 'ar' o 'le', igual que usado en entrenamiento.
            desired_fs (int): Frecuencia de muestreo deseada (ej. 256).
            window_secs (int): Duración de cada ventana en segundos (ej. 30).
        """
        # 1) Cargar y montar señales
        montage_data, fs, n_samples = extract_montage_signals(edf_path, montage, desired_fs)
        
        # 2) Cargar etiquetas reales y generar vector 0/1 a nivel de muestra
        df_labels = load_label_csv(csv_path)
        labels_true = generate_labels_vector(df_labels, fs, n_samples)
        
        # 3) Segmentar en ventanas para alimentar al modelo
        window_size = window_secs * fs
        X_windows, _ = segment_windows(montage_data, np.zeros(n_samples, dtype=np.int32), window_size, mode=1)
        n_windows = X_windows.shape[0]
        
        # 4) Cargar modelo y predecir
        pred_probs = model.predict(X_windows, batch_size=22)
        seq_labels = np.argmax(pred_probs, axis=1)  # 0=background, 1=seiz
        
        # 5) Expandir predicciones de ventana a nivel de muestra
        pred_samples = np.zeros(n_samples, dtype=np.int32)
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = min(n_samples, (i+1) * window_size)
            pred_samples[start_idx:end_idx] = seq_labels[i]
        
        # 6) Crear eje temporal en segundos
        time_axis = np.arange(n_samples) / fs  # cada muestra → tiempo
        
        # 7) Graficar en un solo plot: etiquetas reales vs predichas
        plt.figure(figsize=(12, 4))
        plt.step(time_axis, labels_true, where='post', label='Etiqueta Real', linewidth=1)
        plt.step(time_axis, pred_samples, where='post', label='Predicción', linewidth=1, linestyle='--')
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Label (0=BCKG, 1=SEIZ)')
        plt.title('Detección de convulsión: Real vs Predicho')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    # # Ejemplo de uso (ajusta las rutas antes de ejecutar):
    # model_path = "tcn_eegnet_tuh_v2.h5"

    plot_predictions_vs_true(edf_test_path, csv_test_path, model)
