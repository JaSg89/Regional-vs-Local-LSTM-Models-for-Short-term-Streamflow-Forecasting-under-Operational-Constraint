# -------------------------------------- #
# LIBRARY
# -------------------------------------- #

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import gc
import pandas as pd
import numpy as np
import random
import pickle
import json
import scipy.stats as stats
import statsmodels.api as sm
import math
import datetime
import tensorflow as tf
import h5py
from keras.initializers import glorot_uniform
from keras.layers import Flatten, Multiply, Reshape, Lambda, Concatenate
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Activation
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard,  ModelCheckpoint
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn import metrics
from tensorflow.keras.losses import Huber, LogCosh, MeanSquaredLogarithmicError
from math import sqrt
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad,Adadelta,Nadam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.utils import Sequence
from scipy.stats import pearsonr
from tensorflow.keras import backend as K
import CRPS.CRPS as pscore
from scipy import stats
import psutil
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

SEED = 1933
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


dir_path ='/Users/saave/Desktop/FONDEF_DATA/Entrenamiento_LOCAL/Trained_Models_TEST'
dir_path2 = '/Users/saave/Desktop/FONDEF_DATA/Entrenamiento_LOCAL/Trained_Models_TEST/Hyperparameter_tuning'
df2 = pd.read_csv('/Users/saave/Desktop/FONDEF_DATA/Entrenamiento_LOCAL/LSTM_LOCAL2.CSV') 
df2['date'] = pd.to_datetime(df2['date'])

           
lista =[1001003]
df2 = df2[df2['gauge_id'].isin(lista)]

##############################
###### PREPROCESAMIENTO ######
##############################

# Identificar los gauge_id para eliminar
ids_para_eliminar = df2[df2['caudal_mask2_p0d'] == 3]['gauge_id'].unique()
# Filtrar el DataFrame para excluir esos gauge_id
df2 = df2[~df2['gauge_id'].isin(ids_para_eliminar)]

# multiplicación de columnas
def multiply_columns(df, columns, new_column_name):
      df[new_column_name] = df[columns].prod(axis=1)
      df.drop(columns, axis=1, inplace=True)
      return df

# suma de columnas

def sum_columns(df, columns, new_column_name):
      df[new_column_name] = df[columns].sum(axis=1)
      df.drop(columns, axis=1, inplace=True)
      return df

df2 = multiply_columns(df2, ['snw_o_era5_snr_mean_b_none_d1_m7d', 'snw_o_era5_snd_mean_b_none_d1_m7d'], 'snw_o_era5_swe_mean_b_none_d1_m7d')


################# FILTRO DE DATOS #################

lista =     ['caudal_max_p0d','caudal_max_p1d',  'caudal_max_p2d',  'caudal_max_p3d',  'caudal_max_p4d']\
          + ['caudal_mean_p1d',  'caudal_mean_p2d',  'caudal_mean_p3d',  'caudal_mean_p4d']\
          + ['caudal_mask2_p1d',  'caudal_mask2_p2d',  'caudal_mask2_p3d',  'caudal_mask2_p4d']\
          + ['gl_s_dga_ga_tot_n_none_c_c', 'gl_s_dga_ga_tot_s_none_c_c', 'gl_s_dga_gta_tot_n_none_c_c', 'gl_s_dga_gta_tot_s_none_c_c',
             'gl_s_dga_rgla_tot_n_none_c_c', 'gl_s_dga_rgla_tot_s_none_c_c', 'gl_s_dga_gwe_tot_n_none_c_c', 'gl_s_dga_gwe_tot_s_none_c_c',
             'gl_s_dga_gtwe_tot_n_none_c_c', 'gl_s_dga_gtwe_tot_s_none_c_c','gl_s_dga_rglwe_tot_n_none_c_c', 'gl_s_dga_rglwe_tot_s_none_c_c',
             'gl_s_dga_galt_mean_n_none_c_c', 'gl_s_dga_galt_mean_s_none_c_c', 'gl_s_dga_gtalt_mean_n_none_c_c', 'gl_s_dga_gtalt_mean_s_none_c_c',
             'gl_s_dga_rglalt_mean_n_none_c_c', 'gl_s_dga_rglalt_mean_s_none_c_c', 'sf_s_isric_sbd_mean_b_none_c_c', 'sf_s_isric_sbd_p10_b_none_c_c',
             'sf_s_isric_sbd_p25_b_none_c_c', 'sf_s_isric_sbd_p50_b_none_c_c', 'sf_s_isric_sbd_p75_b_none_c_c', 'sf_s_isric_sbd_p90_b_none_c_c',
             'sf_s_isric_scf_mean_b_none_c_c', 'sf_s_isric_scf_p10_b_none_c_c', 'sf_s_isric_scf_p25_b_none_c_c', 'sf_s_isric_scf_p50_b_none_c_c',
             'sf_s_isric_scf_p75_b_none_c_c', 'sf_s_isric_scf_p90_b_none_c_c', 'sf_s_isric_clay_mean_b_none_c_c', 'sf_s_isric_clay_p10_b_none_c_c',
             'sf_s_isric_clay_p25_b_none_c_c', 'sf_s_isric_clay_p50_b_none_c_c', 'sf_s_isric_clay_p75_b_none_c_c', 'sf_s_isric_clay_p90_b_none_c_c',
             'sf_s_isric_sand_mean_b_none_c_c', 'sf_s_isric_sand_p10_b_none_c_c', 'sf_s_isric_sand_p25_b_none_c_c', 'sf_s_isric_sand_p50_b_none_c_c',
             'sf_s_isric_sand_p75_b_none_c_c', 'sf_s_isric_sand_p90_b_none_c_c', 'sf_s_isric_silt_mean_b_none_c_c', 'sf_s_isric_silt_p10_b_none_c_c',
             'sf_s_isric_silt_p25_b_none_c_c', 'sf_s_isric_silt_p50_b_none_c_c','sf_s_isric_silt_p75_b_none_c_c', 'sf_s_isric_silt_p90_b_none_c_c',
             'sf_s_isric_socc_mean_b_none_c_c', 'sf_s_isric_socc_p10_b_none_c_c', 'sf_s_isric_socc_p25_b_none_c_c', 'sf_s_isric_socc_p50_b_none_c_c',
             'sf_s_isric_socc_p75_b_none_c_c', 'sf_s_isric_socc_p90_b_none_c_c', 'hi_s_cam_sr_tot_b_none_c_c', 'hi_s_cam_gwr_tot_b_none_c_c',
             'idx_s_cam_arcr2_tot_b_none_c_c', 'sf_s_ornl_brd_tot_p_none_c_c', 'sf_s_glhym_por_mean_b_none_c_c', 'sf_s_glhym_perm_mean_b_none_c_c',
             'sf_s_ornl_brd_p10_b_none_c_c', 'sf_s_ornl_brd_p25_b_none_c_c', 'sf_s_ornl_brd_p50_b_none_c_c', 'sf_s_ornl_brd_p75_b_none_c_c',
             'sf_s_ornl_brd_p90_b_none_c_c', 'sf_s_ornl_brd_mean_b_none_c_c'
             ]
df2.drop(lista, axis=1, inplace=True)

############  NORMALIZACIÓN Y FACTOR DE ESCALA ##############
potencia = 4 
df2['caudal_mean_p0d'] = ((86.4 * df2['caudal_mean_p0d']) / df2['top_s_cam_area_tot_b_none_c_c']) ** (1/potencia)

############## Procesamiento de datos ##############

df1 = df2.copy()
data1 = df2.copy()
df2.set_index(['date', 'gauge_id'], inplace = True)
df2.drop(['top_s_cam_lon_none_p_none_c_c', 'top_s_cam_lat_none_p_none_c_c', 'top_s_cam_lon_mean_b_none_c_c',
           'top_s_cam_lat_mean_b_none_c_c', 'top_s_dga_lon_none_p_none_c_c','top_s_dga_lat_none_p_none_c_c',
           'caudal_mask2_p0d', 'caudal_mean_p0d'], axis=1, inplace = True)

print('Número de variables que entran al modelo:', len(df2.columns))
print('cantidad de NAN:', df2.isna().sum().sum())
print(df2.shape[1])

strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0'])  #, '/gpu:1'])
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
time_steps = 250  # Largo de la secuencia
number_of_features = df2.shape[1]  
num_simulations = 30 
num_days = 5

def KGE(sim, obs):
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    corr_coeff = np.corrcoef(sim, obs)[0,1]
    KGE = 1 - np.sqrt((corr_coeff - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return KGE, corr_coeff, alpha, beta

def compute_KGE_for_each_ID(df):
    unique_ids = df['ID'].unique()
    kge_values = []

    for unique_id in unique_ids:
        temp_df = df[df['ID'] == unique_id]
        kge, _, _, _ = KGE(temp_df['Qsim'], temp_df['Qobs'])
        kge_values.append([unique_id, kge])

    result_df = pd.DataFrame(kge_values, columns=['ID', 'KGE'])
    return result_df

# Procesar cada cuenca
for gauge_id in df1['gauge_id'].unique():
    model_path = os.path.join(dir_path2, 'Models', str(gauge_id))
    scaler_path = os.path.join(dir_path, 'Scaler', str(gauge_id))
    sequences_path = os.path.join(dir_path, 'Secuencias', str(gauge_id))

    print(f"Evaluating model for gauge_id: {gauge_id}")

    # Cargar el mejor modelo
    best_model = tf.keras.models.load_model(os.path.join(model_path, 'best_model.h5'))
    best_model.summary()

    # Cargar escaladores
    with open(os.path.join(scaler_path, f'X_scaler_{gauge_id}.pkl'), 'rb') as file:
        X_scaler = pickle.load(file)
    with open(os.path.join(scaler_path, f'y_scaler_{gauge_id}.pkl'), 'rb') as file:
        y_scaler = pickle.load(file)

    # Cargar datos de test
    with h5py.File(os.path.join(sequences_path, 'Secuencias_test', 'test.h5'), 'r') as f:
        X_test = f['X_test'][:]
        y_test = f['y_test'][:]

    num_simulations= num_simulations
    all_predictions_test = np.zeros((num_simulations, y_test.shape[0], 5 ))

    for i in range(num_simulations):
        y_pred_test = best_model.predict(X_test)
        y_pred_transformed_test = y_scaler.inverse_transform(y_pred_test)
        y_pred_transformed_test  = np.clip(y_pred_transformed_test, 0, None)
        all_predictions_test[i] = np.power(y_pred_transformed_test, potencia)  # Elevar a la cua a

    y_test_inv = np.power(y_scaler.inverse_transform(y_test), potencia)

    # Guardar predicciones con numpy

    np.save(f'{model_path}/y_test_inv.npy', y_test_inv)
    np.save(f'{model_path}/y_pred_inv_test.npy', all_predictions_test)

    # Inicialización de diccionarios para guardar los resu ados
    means = {}
    std_dev = {}
    medians = {}
    kurtosis_vals = {}
    skewness_vals = {}
    percentiles_5 = {}
    percentiles_25 = {}
    percentiles_75 = {}
    percentiles_95 = {}


    # Cálculo de estadígrafos para cada día
    for day in range(num_days):
        # Indexación para seleccionar solo las predicciones del día actual
        day_predictions = all_predictions_test[:, :, day]
        means[day] = day_predictions.mean(axis=0)
        std_dev[day] = day_predictions.std(axis=0)
        medians[day] = np.median(day_predictions, axis=0)
        kurtosis_vals[day] = stats.kurtosis(day_predictions, axis=0, nan_policy='raise')
        skewness_vals[day] = stats.skew(day_predictions, axis=0)

        # Reemplazar NaN en ku osis_vals y skewness_vals
        kurtosis_vals[day][np.isnan(kurtosis_vals[day])] = 3000
        skewness_vals[day][np.isnan(skewness_vals[day])] = 0

        percentiles_5[day] = np.percentile(day_predictions, 5, axis=0)
        percentiles_25[day] = np.percentile(day_predictions, 25, axis=0)
        percentiles_75[day] = np.percentile(day_predictions, 75, axis=0)
        percentiles_95[day] = np.percentile(day_predictions, 95, axis=0)
    # Inicializar una lista para almacenar los valores CRPS para cada día
    crps_values_days = {}

    # Iterar sobre cada día
    for day in range(num_days):
        crps_values = [] # Lista para almacenar los valores CRPS para el día actual

    # Iterar sobre cada punto de datos
        for j in range(y_test_inv.shape[0]):
           # Extraer las predicciones de las simulaciones para el punto de datos j y el día actual
            ensemble_predictions = all_predictions_test[:, j, day]
           # Calcular el CRPS para el punto de datos j usando las simulaciones
            crps, _, _ = pscore(ensemble_predictions, y_test_inv[j, day]).compute()
            crps_values.append(crps)

        # Almacenar los valores CRPS del día actual en el diccionario
        crps_values_days[day] = crps_values



    dates_array = np.load(f'{sequences_path}/Secuencias_test/dates_test.npy', allow_pickle=True)
    dates_as_datetime = pd.to_datetime(dates_array)

    # Inicializar una lista para almacenar dataframes
    dfs = []

    # Crear y almacenar dataframes para cada día de predicción
    for day in range(num_days):
    # Ajustar las fechas para el día de predicción actual
        adjusted_dates = dates_as_datetime + pd.DateOffset(days=0)
        df = pd.DataFrame({
            'ID': np.load(f'{sequences_path}/Secuencias_test/gauge_ids_test.npy'),
            'date': adjusted_dates,
            'mean': means[day].squeeze(),
            'std_dev': std_dev[day].squeeze(),
            'median': medians[day].squeeze(),
            'kurtosis': kurtosis_vals[day].squeeze(),
            'skewness': skewness_vals[day].squeeze(),
            'per_5': percentiles_5[day].squeeze(),
            'per_25': percentiles_25[day].squeeze(),
            'per_75': percentiles_75[day].squeeze(),
            'per_95': percentiles_95[day].squeeze(),
            'Qsim': means[day].squeeze(),
            'Qobs': y_test_inv[:, day].squeeze(),
            'crps': crps_values_days[day]

        })
        dfs.append(df)


    # Asignar cada dataframe a una variable específica
    summary_test0, summary_test1, summary_test2, summary_test3, summary_test4 = dfs

    ########### Merge de caudal_maks2_p0d

    dfs_merged_test = [] # Lista para almacenar los DataFrames después del merge
    dataframes = [summary_test0, summary_test1, summary_test2, summary_test3, summary_test4]

    for df in dataframes:
       # Realizamos el merge y renombramos la columna 'caudal_mask2_p0d' a 'mask' en un paso
        merged_df = df.merge(
            data1[['gauge_id', 'date', 'caudal_mask2_p0d']].rename(columns={'gauge_id': 'ID', 'caudal_mask2_p0d': 'mask'}),
            on=['ID', 'date'],
            how='left'
        )

        dfs_merged_test.append(merged_df)

    # Asignar cada dataframe resultante a una variable específica
    summary_test0, summary_test1, summary_test2, summary_test3, summary_test4 = dfs_merged_test
    summary_test0.to_csv(f'{model_path}/Qmean_LSTM_p0d.csv', index=False)
    summary_test1.to_csv(f'{model_path}/Qmean_LSTM_p1d.csv', index=False)
    summary_test2.to_csv(f'{model_path}/Qmean_LSTM_p2d.csv', index=False)
    summary_test3.to_csv(f'{model_path}/Qmean_LSTM_p3d.csv', index=False)
    summary_test4.to_csv(f'{model_path}/Qmean_LSTM_p4d.csv', index=False)
   
    #####  KGE #####

    kge_test_p0d = compute_KGE_for_each_ID(df = summary_test0[summary_test0['mask']==2])
    kge_test_p1d = compute_KGE_for_each_ID(df = summary_test1[summary_test1['mask']==2])
    kge_test_p2d = compute_KGE_for_each_ID(df = summary_test2[summary_test2['mask']==2])
    kge_test_p3d = compute_KGE_for_each_ID(df = summary_test3[summary_test3['mask']==2])
    kge_test_p4d = compute_KGE_for_each_ID(df = summary_test4[summary_test4['mask']==2])


    kge_test_p0d.to_csv(f'{model_path}/kge_test_p0d.csv', index=False)
    kge_test_p1d.to_csv(f'{model_path}/kge_test_p1d.csv', index=False)
    kge_test_p2d.to_csv(f'{model_path}/kge_test_p2d.csv', index=False)
    kge_test_p3d.to_csv(f'{model_path}/kge_test_p3d.csv', index=False)
    kge_test_p4d.to_csv(f'{model_path}/kge_test_p4d.csv', index=False)

       
       




