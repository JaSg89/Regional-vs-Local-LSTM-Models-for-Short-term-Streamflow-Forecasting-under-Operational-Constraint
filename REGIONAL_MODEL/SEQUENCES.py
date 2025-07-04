################# ESCALAMIENTO Y CREAR SECUENCIAS ####################

import os
import pandas as pd
import numpy as np
import random
import pickle
import json
import scipy.stats as stats
import statsmodels.api as sm
import math
import datetime
import h5py
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn import metrics
import psutil


############ PREPARACIÓN DE LOS DATOS ##############

#### Configuración de semillas para reproducibilidad
random.seed(123)
np.random.seed(123)

dir_path ='/Users/saave/Desktop/FONDEF_DATA/Entrenamiento_LOCAL/Trained_Models_TEST'
df1 = pd.read_csv('C:/Users/saave/Desktop/FONDEF_DATA/LSTM_Data/LSTM11.CSV')
df1['date'] = pd.to_datetime(df1['date'])

#lista =[1001002, 3825001, 5320001]
#df1 = df1[df1['gauge_id'].isin(lista)]

###### PREPROCESAMIENTO ######

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

df1 = multiply_columns(df1, ['snw_o_era5_snr_mean_b_none_d1_m7d', 'snw_o_era5_snd_mean_b_none_d1_m7d'], 'snw_o_era5_swe_mean_b_none_d1_m7d')
df1 = sum_columns(df1, ['gl_s_dga_ga_tot_n_none_c_c', 'gl_s_dga_rgla_tot_n_none_c_c', 'gl_s_dga_gta_tot_n_none_c_c',
                          'gl_s_dga_ga_tot_s_none_c_c', 'gl_s_dga_rgla_tot_s_none_c_c', 'gl_s_dga_gta_tot_s_none_c_c'], 'gl_s_dga_gta_tot_sum_none_c_c')

################# FILTRO DE DATOS TEMPORALES #################

lista =     ['caudal_max_p0d','caudal_max_p1d',  'caudal_max_p2d',  'caudal_max_p3d',  'caudal_max_p4d']\
          + ['caudal_mean_p1d',  'caudal_mean_p2d',  'caudal_mean_p3d',  'caudal_mean_p4d']\
          + ['caudal_mask2_p1d',  'caudal_mask2_p2d',  'caudal_mask2_p3d',  'caudal_mask2_p4d']

df1.drop(lista, axis=1, inplace=True)

######## NORMALIZACIÓN Y FACTOR DE ESCALA #########
df1['caudal_mean_p0d'] = ((86.4*df1['caudal_mean_p0d'])/df1['top_s_cam_area_tot_b_none_c_c'])**(1/4)

##### Procesamiento de datos #####
df1.set_index(['date', 'gauge_id'], inplace = True)
df2 = df1.copy()
df2.drop(['top_s_cam_lon_none_p_none_c_c', 'top_s_cam_lat_none_p_none_c_c', 'top_s_cam_lon_mean_b_none_c_c',
             'top_s_cam_lat_mean_b_none_c_c', 'top_s_dga_lon_none_p_none_c_c','top_s_dga_lat_none_p_none_c_c'], axis=1, inplace = True)


#### Separación de los conjuntos #####
train = (df2.loc[df2['caudal_mask2_p0d'].isin([0])]).drop(['caudal_mask2_p0d'], axis=1)
val = df2.loc[df2['caudal_mask2_p0d'].isin([1])].drop(['caudal_mask2_p0d'], axis=1)
test = df2.loc[df2['caudal_mask2_p0d'].isin([2])].drop([ 'caudal_mask2_p0d'], axis=1)
df2 = df2.drop(['caudal_mean_p0d', 'caudal_mask2_p0d'], axis=1)

print('NAN en el df: ',df2.isna().sum().sum())
print('número de variables que entran al modelo: ', len(df2.columns))
print(df2.shape[1])

################### ESCALAMIENTO GLOBAL ANTES DE LA SECUENCIA ####################

def scale_data(train, val, test, target, df_original):
    # Inicializar los escaladores
   
    X_scaler = MinMaxScaler(clip=True)
    y_scaler = MinMaxScaler(clip=True)

    # Obtener los índices de los DataFrames de entrenamiento
   
    train_indices = train.index
    val_indices = val.index
    test_indices = test.index
    original_indices = df_original.index

    # Ajustar el escalador de características con los datos de entrenamiento
    X_scaler.fit(train.drop(target, axis=1))

    # Ajustar el escalador de target con los datos de entrenamiento
    y_scaler.fit(train[[target]])

    # Escalar las características en los conjuntos de datos y convertirlas en DataFrame
    X_train_scaled = pd.DataFrame(X_scaler.transform(train.drop(target, axis=1)), columns=train.columns.drop(target), index=train_indices)
    X_val_scaled = pd.DataFrame(X_scaler.transform(val.drop(target, axis=1)), columns=val.columns.drop(target), index=val_indices)
    X_test_scaled = pd.DataFrame(X_scaler.transform(test.drop(target, axis=1)), columns=test.columns.drop(target), index=test_indices)

    # Escalar el target y convertirlo en DataFrame
    y_train_scaled = pd.DataFrame(y_scaler.transform(train[[target]]), columns=[target], index=train_indices)
    y_val_scaled = pd.DataFrame(y_scaler.transform(val[[target]]), columns=[target], index=val_indices)
    y_test_scaled = pd.DataFrame(y_scaler.transform(test[[target]]), columns=[target], index=test_indices)

    # Concatenar las características y el target escalados a los conjuntos de datos
    train_scaled = pd.concat([X_train_scaled, y_train_scaled], axis=1)
    val_scaled = pd.concat([X_val_scaled, y_val_scaled], axis=1)
    test_scaled = pd.concat([X_test_scaled, y_test_scaled], axis=1)

    # Escalar el df_original utilizando el escalador de características
    original_scaled = pd.DataFrame(X_scaler.transform(df_original), columns=df_original.columns, index=original_indices)


    return train_scaled, val_scaled, test_scaled, original_scaled, X_scaler, y_scaler

# ESCALAMIENTO
train_s, val_s, test_s, df2, X_scaler, y_scaler = scale_data(train, val, test, target='caudal_mean_p0d', df_original = df2)

# Guardar los escaladores
with open(f'{dir_path}/X_scaler.pkl', 'wb') as file:
    pickle.dump(X_scaler, file)

with open(f'{dir_path}/y_scaler.pkl', 'wb') as file:
    pickle.dump(y_scaler, file)

###### FUNCIÓN QUE CREA LAS SECUENCIAS #######

## CUENTA LAS SECUENCIAS VALIDAS

def count_sequences(df_filtrado, time_steps, prediction_days):
    conteo_valido = 0
    conteo_no_valido = 0
    
    first_date = df_filtrado.index.get_level_values('date').min()    
    df_filtrado = df_filtrado[df_filtrado.index.get_level_values('date') >= first_date + pd.DateOffset(days=time_steps-1)]
    
    # Crear un diccionario que almacena los conjuntos de fechas para cada gauge_id.
    # Utilizamos conjuntos ya que la operación "issubset" (determinar si un conjunto es un subconjunto de otro)
    date_sets = {}
    
    # Para cada gauge_id único en el DataFrame guardamos en el diccionario 'date_sets' las fechas asociadas a ese gauge_id
    for gauge_id in df_filtrado.index.get_level_values('gauge_id').unique():
        date_sets[gauge_id] = set(df_filtrado.loc[pd.IndexSlice[:, gauge_id], :].index.get_level_values('date'))

    # Ahora, iteramos sobre las combinaciones únicas de fecha y gauge_id en el DataFrame.
    for end_date, gauge_id in df_filtrado.index.unique():
        pred_dates = {end_date + pd.DateOffset(days=i) for i in range(prediction_days)}
        
        if pred_dates.issubset(date_sets[gauge_id]):
            conteo_valido += 1
        else:
            conteo_no_valido += 1

    return conteo_valido

############## CREACIÓN DE SECUENCIAS PARA 5 DIAS  ##############

def create_and_save_dataset(df_original, df_filtrado, time_steps, prediction_days, dataset_name, dir_path):

    # Contar el número de secuencias válidas que se pueden formar en el DataFrame filtrado excluyendo las de time_steps
    valid_sequences = count_sequences(df_filtrado, time_steps, prediction_days)

    # Si el directorio donde se guardarán las secuencias no existe, se crea
    if not os.path.exists(f'{dir_path}/Secuencias_{dataset_name}/'):
        os.makedirs(f'{dir_path}/Secuencias_{dataset_name}/')

    # Determinar la fecha de inicio del DataFrame filtrado
    first_date = df_filtrado.index.get_level_values('date').min()

    # Filtrar las fechas del DataFrame filtrado para considerar solo aquellas fechas válidas
    df_filtrado = df_filtrado[df_filtrado.index.get_level_values('date') >= first_date + pd.DateOffset(days=time_steps-1)]

    conteo_omitido = 0  # Contador para las secuencias que no son válidas y se omiten
    conteo_creado = 0  # Contador para las secuencias que son válidas y se crean

    # Creación del archivo h5 donde se guardarán las secuencias
    with h5py.File(f'{dir_path}/Secuencias_{dataset_name}/{dataset_name}.h5', 'w') as hf:

        # Inicializar conjuntos de datos para almacenar las secuencias de entrada y objetivo
        X_data = hf.create_dataset(f'X_{dataset_name}', (valid_sequences, time_steps, df_original.shape[1]), dtype='float64')
        y_data = hf.create_dataset(f'y_{dataset_name}', (valid_sequences, prediction_days), dtype='float64')

        # Inicializar una lista de tamaño 'valid_sequences' para almacenar los gauge_id temporales
        temp_gauge_ids = [-1] * valid_sequences
        temp_dates = [None] * valid_sequences

        # Agrupar el DataFrame filtrado por 'gauge_id'
        for gauge_id, group in df_filtrado.groupby(level=1):

            # Ordenar y extraer las fechas del grupo actual
            date_indices = group.index.get_level_values('date').sort_values()


            for end_date in date_indices:

                # Crear una lista de fechas de predicción
                pred_dates = [end_date + pd.DateOffset(days=i) for i in range(prediction_days)]

                # Si alguna fecha de predicción no está presente, omitir la secuencia
                if not all(d in date_indices for d in pred_dates):
                    conteo_omitido += 1
                    continue

                # Extraer los valores objetivo para las fechas de predicción
                pred_values = group.loc[(slice(pred_dates[0], pred_dates[-1]), gauge_id), 'caudal_mean_p0d'].values

                # Determinar la fecha de inicio para la secuencia de entrada
                start_date = end_date - pd.DateOffset(days=time_steps-1)

                # Extraer la secuencia de entrada
                temp_df = df_original.loc[(slice(start_date, end_date), gauge_id), :]

                # Si la secuencia de entrada no tiene la longitud correcta, omitirla
                if temp_df.shape[0] != time_steps:
                    conteo_omitido += 1
                    print('REVISAR')
                    continue

                # Guardar la secuencia de entrada y los valores objetivo en los conjuntos de datos h5
                X_data[conteo_creado, :, :] = temp_df.values
                y_data[conteo_creado, :] = pred_values

                # Guardar el gauge_id en la lista temporal
                temp_gauge_ids[conteo_creado] = gauge_id
                temp_dates[conteo_creado] = end_date

                # Incrementar el contador de secuencias creadas
                conteo_creado += 1

        #Guardar gauge_id y dates para despues hacer las metricas
        np.save(f'{dir_path}/Secuencias_{dataset_name}/gauge_ids_{dataset_name}.npy', np.array(temp_gauge_ids))
        np.save(f'{dir_path}/Secuencias_{dataset_name}/dates_{dataset_name}.npy', np.array(temp_dates))

        # Imprimir estadísticas sobre las secuencias creadas y omitidas
        print('=' * 25)
        print(' '*10 + f'{dataset_name}')
        print('=' * 25)
        print(f"Número de gauge_id: {len(temp_gauge_ids)}")
        print(f"Número de fechas: {len(temp_dates)}")
        print(f"Número de secuencias omitidas: {conteo_omitido}")
        print(f"Número de secuencias creadas: {conteo_creado}")
        print('shape de y:', y_data.shape)
        if -1 in temp_gauge_ids:
          print("Hay valores no actualizados (-1) en temp_gauge_ids!")
        else:
          print("Todos los valores en temp_gauge_ids han sido actualizados correctamente.")

    return conteo_creado

# Largo de la secuencia
time_steps = 250 

# Guardar secuencias y obtener la cantidad de secuencias
num_sequences_train = create_and_save_dataset(df_original = df2, df_filtrado = train_s, time_steps = time_steps, prediction_days = 5, dataset_name = 'train', dir_path = dir_path)
num_sequences_val = create_and_save_dataset(df_original = df2, df_filtrado = val_s, time_steps = time_steps, prediction_days = 5, dataset_name = 'val', dir_path= dir_path)
num_sequences_test = create_and_save_dataset(df_original = df2, df_filtrado = test_s, time_steps = time_steps, prediction_days = 5, dataset_name = 'test', dir_path= dir_path)

############## GUARDA EL NÚMERO DE SECUENCIAS QUE DESPUES SE OCUPARÁ EN EL GENERADOR ###########
data_to_save = np.array([num_sequences_train, num_sequences_val, num_sequences_test])
np.save(f'{dir_path}/num_sequences.npy', data_to_save)

