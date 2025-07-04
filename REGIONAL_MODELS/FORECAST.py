#################### ENTRENAMIENTO Y EVALUACIÓN #######################

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
#from keras.utils import Sequence
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad,Adadelta,Nadam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.utils import Sequence
from scipy.stats import pearsonr
from tensorflow.keras import backend as K
import CRPS.CRPS as pscore
from scipy import stats
import psutil
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Configuración de semillas para reproducibilidad
random.seed(123)
tf.random.set_seed(123)
np.random.seed(123)

def KGE(sim, obs):
    alpha = sim.std()/obs.std()
    beta = sim.mean()/obs.mean()
    corr_coeff = np.corrcoef(sim, obs)
    corr_coeff = corr_coeff[0,1]

    KGE = 1 - np.sqrt((corr_coeff-1)**2 + (alpha-1)**2 + (beta-1)**2 )
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

dir_path ='/Users/saave/Desktop/FONDEF_DATA/Entrenamiento_LOCAL/Trained_Models_TEST'
df2 = pd.read_csv('C:/Users/saave/Desktop/FONDEF_DATA/LSTM_Data/LSTM11.CSV')
df2['date'] = pd.to_datetime(df2['date'])

## Only for testing
lista =[1001002, 3825001, 5320001]
df2 = df2[df2['gauge_id'].isin(lista)]

##############################
###### PREPROCESAMIENTO ######
##############################

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
df2 = sum_columns(df2, ['gl_s_dga_ga_tot_n_none_c_c', 'gl_s_dga_rgla_tot_n_none_c_c', 'gl_s_dga_gta_tot_n_none_c_c',
                          'gl_s_dga_ga_tot_s_none_c_c', 'gl_s_dga_rgla_tot_s_none_c_c', 'gl_s_dga_gta_tot_s_none_c_c'], 'gl_s_dga_gta_tot_sum_none_c_c')

################# FILTRO DE DATOS TEMPORALES #################
lista =     ['caudal_max_p0d','caudal_max_p1d',  'caudal_max_p2d',  'caudal_max_p3d',  'caudal_max_p4d']\
          + ['caudal_mean_p1d',  'caudal_mean_p2d',  'caudal_mean_p3d',  'caudal_mean_p4d']\
          + ['caudal_mask2_p1d',  'caudal_mask2_p2d',  'caudal_mask2_p3d',  'caudal_mask2_p4d']

df2.drop(lista, axis=1, inplace=True)

############  NORMALIZACIÓN Y FACTOR DE ESCALA ##############

df2['caudal_mean_p0d'] = ((86.4 * df2['caudal_mean_p0d']) / df2['top_s_cam_area_tot_b_none_c_c']) ** (1/4)

####### Procesamiento de datos
df1 = df2.copy()
data1 = df2.copy()
df2.set_index(['date', 'gauge_id'], inplace = True)
df2.drop(['top_s_cam_lon_none_p_none_c_c', 'top_s_cam_lat_none_p_none_c_c', 'top_s_cam_lon_mean_b_none_c_c',
           'top_s_cam_lat_mean_b_none_c_c', 'top_s_dga_lon_none_p_none_c_c','top_s_dga_lat_none_p_none_c_c',
           'caudal_mask2_p0d', 'caudal_mean_p0d'], axis=1, inplace = True)

print('Número de variables que entran al modelo:', len(df2.columns))
print('cantidad de NAN:', df2.isna().sum().sum())
print(df2.shape[1])

########### CARGAR NÚMERO DE SECUENCIAS ##############

data_loaded = np.load(f'{dir_path}/num_sequences.npy')
num_sequences_test = data_loaded[2]

################ GENERADOR ###################

class DataGenerator(Sequence):
    def __init__(self, dataset_name, batch_size, data_dir, num_sequences, shuffle=True):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_sequences = num_sequences
        self.shuffle = shuffle
        self.indexes = np.arange(0, self.num_sequences)
        self.on_epoch_end()
        # Abre el archivo HDF5 para lectura
        self.h5f = h5py.File(f'{self.data_dir}/{self.dataset_name}.h5', 'r') # xr.open_dataset('file')

    # Devuelve el número total de batches en el dataset
    def __len__(self):
        return int(np.ceil(len(self.indexes) / float(self.batch_size)))

    def __getitem__(self, index):
        start_idx = index*self.batch_size
        end_idx = min((index+1)*self.batch_size, len(self.indexes)) # Este cambio es para evitar errores al final del conjunto de datos
        X = self.h5f[f'X_{self.dataset_name}'][start_idx:end_idx]
        y = self.h5f[f'y_{self.dataset_name}'][start_idx:end_idx,:] # Seleciona los 5 dias
        return X, y
          
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # PARA COMPARACIONES POSTERIORES
    def get_indexes(self):
      return self.indexes

    def __del__(self):
       self.h5f.close()
  


################################################################ 
########### EVALUACIÓN DEL MODELO EN VALIDACIÓN ###############
################################################################  

#best_model = tf.keras.models.load_model(f'{dir_path}/best_model.h5', custom_objects={'PermanentDropout': PermanentDropout})
best_model = tf.keras.models.load_model(f'{dir_path}/best_model.h5')
best_model.summary()

############ CARGAR ESCALADORES ###############

with open(f'{dir_path}/X_scaler.pkl', 'rb') as file:
    X_scaler = pickle.load(file)

with open(f'{dir_path}/y_scaler.pkl', 'rb') as file:
    y_scaler = pickle.load(file)

########### NÚMERO DE SECUENCIAS ##############

data_loaded = np.load(f'{dir_path}/num_sequences.npy')
num_sequences_test = data_loaded[2]

########### CARGAR GAUGE_ID ###############
 
gauge_ids_train = np.load(f'{dir_path}/Secuencias_train/gauge_ids_train.npy')


########### CARGAR FECHAS ###############

dates_test = np.load(f'{dir_path}/Secuencias_test/dates_test.npy', allow_pickle=True)
dates_test = pd.to_datetime(dates_test)

# Recopila las verdaderas etiquetas de validación (y_true)
with h5py.File(f'{dir_path}/Secuencias_test/test.h5', 'r') as h5f:
        y_test = h5f[f'y_test'][:]




################################################################ 
############# EVALUACIÓN DEL MODELO EN TEST ####################
################################################################  

######## HACER PREDICCIONES EN X_val_s ###########
batch =256 
data_dir_test = f'{dir_path}/Secuencias_test'
test_generator = DataGenerator(dataset_name = 'test', batch_size = batch, data_dir =  data_dir_test,
                    num_sequences= num_sequences_test, shuffle=True)

num_simulations = 100 
all_predictions_test = np.zeros((num_simulations, y_test.shape[0], 5 ))

for i in range(num_simulations):
    y_pred_test = best_model.predict(test_generator)
    y_pred_transformed_test = y_scaler.inverse_transform(y_pred_test)
    y_pred_transformed_test  = np.clip(y_pred_transformed_test, 0, None)
    all_predictions_test[i] = np.power(y_pred_transformed_test, 4)  # Elevar a la cuarta

y_test_inv = np.power(y_scaler.inverse_transform(y_test), 4)
np.save(f'{dir_path}/y_test_inv.npy', y_test_inv)
np.save(f'{dir_path}/y_pred_inv_test.npy', all_predictions_test)


#############################################################
############ MÉTRICAS DE EVALUACIÓN DEL MODELO ##############
############################################################

# Inicialización de diccionarios para guardar los resultados
means = {}
std_dev = {}
medians = {}
kurtosis_vals = {}
skewness_vals = {}
percentiles_5 = {}
percentiles_25 = {}
percentiles_75 = {}
percentiles_95 = {}

num_days = 5

# Cálculo de estadígrafos para cada día
for day in range(num_days):
    # Indexación para seleccionar solo las predicciones del día actual
    day_predictions = all_predictions_test[:, :, day]
    
    means[day] = day_predictions.mean(axis=0)
    std_dev[day] = day_predictions.std(axis=0)
    medians[day] = np.median(day_predictions, axis=0)
    kurtosis_vals[day] = stats.kurtosis(day_predictions, axis=0, nan_policy='raise')
    skewness_vals[day] = stats.skew(day_predictions, axis=0)
    
    # Reemplazar NaN en kurtosis_vals y skewness_vals
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
    crps_values = []  # Lista para almacenar los valores CRPS para el día actual
    # Iterar sobre cada punto de datos 
    for j in range(y_test_inv.shape[0]):
        # Extraer las predicciones de las simulaciones para el punto de datos j y el día actual
        ensemble_predictions = all_predictions_test[:, j, day]

        # Calcular el CRPS para el punto de datos j usando las simulaciones
        crps, _, _ = pscore(ensemble_predictions, y_test_inv[j, day]).compute()
        crps_values.append(crps)

    # Almacenar los valores CRPS del día actual en el diccionario
    crps_values_days[day] = crps_values

dates_array = np.load(f'{dir_path}/Secuencias_test/dates_test.npy', allow_pickle=True)
dates_as_datetime = pd.to_datetime(dates_array)

# Inicializar una lista para almacenar dataframes
dfs = []

# Crear y almacenar dataframes para cada día de predicción
for day in range(num_days):
    # Ajustar las fechas para el día de predicción actual
    adjusted_dates = dates_as_datetime + pd.DateOffset(days=0)
    
    df = pd.DataFrame({
        'ID': np.load(f'{dir_path}/Secuencias_test/gauge_ids_test.npy'),
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

dfs_merged_test = []  # Lista para almacenar los DataFrames después del merge
dataframes = [summary_test0, summary_test1, summary_test2, summary_test3, summary_test4]
for df in dataframes:
    # Realizamos el merge y renombramos la columna 'caudal_mask2_p0d' a 'mask' en un paso
    merged_df = df.merge(
        df1[['gauge_id', 'date', 'caudal_mask2_p0d']].rename(columns={'gauge_id': 'ID', 'caudal_mask2_p0d': 'mask'}),
        on=['ID', 'date'],
        how='left'
    )

    dfs_merged_test.append(merged_df)

# Asignar cada dataframe resultante a una variable específica
summary_test0, summary_test1, summary_test2, summary_test3, summary_test4 = dfs_merged_test

summary_test0.to_csv(f'{dir_path}/Qmean_LSTM_p0d.csv', index=False)
summary_test1.to_csv(f'{dir_path}/Qmean_LSTM_p1d.csv', index=False)
summary_test2.to_csv(f'{dir_path}/Qmean_LSTM_p2d.csv', index=False)
summary_test3.to_csv(f'{dir_path}/Qmean_LSTM_p3d.csv', index=False)
summary_test4.to_csv(f'{dir_path}/Qmean_LSTM_p4d.csv', index=False)

print(' ')
kge = compute_KGE_for_each_ID(df = summary_test0)
print('KGE MAYOR A 0.6 (p0d):', (kge['KGE'] >= 0.6).sum())
print('KGE MAYOR A 0.5 (p0d):', (kge['KGE'] >= 0.5).sum())
print('KGE MENOR QUE 0 (p0d):', (kge['KGE'] <  0).sum())
print(' ')
print('PRONOSTICO P4D')
print(' ')
kge = compute_KGE_for_each_ID(df = summary_test4)
print('KGE MAYOR A 0.6 (p4d):', (kge['KGE'] >= 0.6).sum())
print('KGE MAYOR A 0.5 (p4d):', (kge['KGE'] >= 0.5).sum())
print('KGE MENOR QUE 0 (p4d):', (kge['KGE'] <  0).sum())



