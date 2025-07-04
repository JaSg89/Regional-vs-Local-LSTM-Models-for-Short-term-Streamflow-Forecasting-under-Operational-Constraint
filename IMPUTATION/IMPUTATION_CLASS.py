import os
import pandas as pd
import time
from scipy.interpolate import interp1d
from scipy import interpolate
import math
import warnings
from sklearn.linear_model import LinearRegression
import numpy as np
import scipy.stats as stats
import pickle
import json
import  h5py
import warnings
warnings.filterwarnings('ignore')
from scipy.interpolate import interp1d
import shutil
import dotenv

inicio = time.time()
dotenv.load_dotenv()
# Variables de entorno
HIDROCL_ROOT_PATH = os.getenv('HIDROCL_ROOT_PATH')
DIC_PATH = os.getenv('DIC_PATH')
DATA_PATH = os.getenv('DATA_PATH')
IMPUTED_PATH = os.getenv('IMPUTED_PATH')
UPDATED_PATH = os.getenv('UPDATED_PATH')

# Expandir las rutas al definirlas
path = IMPUTED_PATH
path_save = UPDATED_PATH

#######################################
######## Imputation gfs path ##########
#######################################


TSNE = os.getenv('Tsne')
dir_path = os.getenv('dir_path')
Path = os.getenv('Path')
ruta_2 = os.getenv('ruta_2')

Tsne = pd.read_csv(TSNE, sep=';')
end_date = pd.Timestamp.now().normalize() 
fecha_maxima = pd.Timestamp.now().normalize()
print(end_date)

###########################################################################
########################### Se Borran los df ##############################
###########################################################################

'''
def borrar_subfolders(path_save_base):
    subfolders = ['observed', 'forecasted']
    for subfolder in subfolders:
        folder_path = os.path.join(path_save_base, subfolder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"La carpeta {folder_path} ha sido borrada.")
        else:
            print(f"La carpeta {folder_path} no existe y no necesita ser borrada.")


# Llamar a la función de borrado antes de iniciar la imputación
borrar_subfolders(UPDATED_PATH)
'''

def borrar_y_crear_subfolders(path_save_base):
    subfolders = ['observed', 'forecasted']
    for subfolder in subfolders:
        folder_path = os.path.join(path_save_base, subfolder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"La carpeta {folder_path} ha sido borrada.")
        else:
            print(f"La carpeta {folder_path} no existe y no necesita ser borrada.")
        
        # Crear la carpeta nuevamente
        os.makedirs(folder_path)
        print(f"La carpeta {folder_path} ha sido creada nuevamente.")

# Llamar a la función de borrado y creación antes de iniciar la imputación
borrar_y_crear_subfolders(UPDATED_PATH)

def fechas_faltantes(df, columna_fecha, fecha_inicio, fecha_fin):

    # Crea una lista de todas las fechas en el rango deseado
    date_range = pd.date_range(start=fecha_inicio, end=fecha_fin)

    # Convierte la columna de fecha en un objeto de fecha y ordena los valores
    df[columna_fecha] = pd.to_datetime(df[columna_fecha])
    df = df.sort_values(columna_fecha)
    # Obtiene la lista de fechas en el DataFrame y elimina la hora
    df_fechas = df[columna_fecha].dt.date.tolist()
    # Crea una lista de fechas faltantes comparando la lista de todas las fechas con la lista de fechas en el DataFrame
    fechas_faltantes = list(set(date_range.date) - set(df_fechas))
    # Ordena la lista de fechas faltantes
    fechas_faltantes = sorted(fechas_faltantes)
    # Crea un DataFrame con las fechas faltantes en formato 'yyyy-mm-dd'
    fechas_faltantes_df = pd.DataFrame({'Fechas faltantes': fechas_faltantes})
    # Cuenta la cantidad de fechas faltantes
    cantidad_fechas_faltantes = len(fechas_faltantes)
    print('Fechas faltantes:', cantidad_fechas_faltantes)

    return fechas_faltantes_df

def complete_dates(df, start_date, end_date):
    try:
    # Crea un rango de fechas entre start_date y end_date
        date_range = pd.date_range(start = start_date, end=end_date)
        # Convierte la columna "date" en una columna de tipo fecha
        df['date'] = pd.to_datetime(df['date'])
        # Crea un marco de datos con las fechas en el rango
        all_dates_df = pd.DataFrame({'date': date_range})
        # Une los dos marcos de datos en un solo marco de datos utilizando una unión externa
        result = pd.merge(all_dates_df, df, on='date', how='left')
        # Rellena los valores faltantes con NaN
        result = result.fillna(value=np.nan)
        # Recupera las filas originales
        result = pd.concat([result, df[df['date'] > end_date]])
        # Ordena el marco de datos por la columna "date"
        result = result.sort_values(by='date')
        # Reasigna los índices del marco de datos
        result = result.reset_index(drop=True)

        return result
    except KeyError as e:
        print("An error occurred:", e)

#Funcion de transformación
def quantile_func(observed_data, forecast_data, num_quantile):

    # Datos mayores que cero en ambos conjuntos
    data0 = (observed_data > 0)  & (forecast_data > 0)
    #data1 = (observed_data == 0) | (forecast_data == 0)

    # Crear dos nuevos  NaN
    obs = observed_data
    obs[forecast_data.isnull()] = np.nan
    pred = forecast_data
    pred[observed_data.isnull()] = np.nan

    # Quantiles
    quantiles_observed = np.nanquantile(obs[data0], q=np.linspace(0, 1, num=num_quantile))
    quantiles_forecast = np.nanquantile(pred[data0], q=np.linspace(0, 1, num=num_quantile))

    # Buscar los índices de los cuantiles repetidos en quantiles_observed
    unique_indexes_observed = np.where(np.diff(quantiles_observed) != 0)[0]
    unique_indexes_observed = np.concatenate((unique_indexes_observed, [len(quantiles_observed) - 1]))

    # Agregar el 0 a ambos cuantiles
    quantiles_forecast_unique = np.concatenate([[0], quantiles_forecast[unique_indexes_observed]])
    quantiles_observed_unique = np.concatenate([[0], quantiles_observed[unique_indexes_observed]])

    # Se encuentra la función que mapea los cuantiles de una distribución a los cuantiles de la otra
    transform_func = interp1d(quantiles_observed_unique, quantiles_forecast_unique, kind='linear', bounds_error=False, fill_value='extrapolate')

    # Se aplica la función de transformación a los datos observados
    data_transform = pd.DataFrame(transform_func(obs))
    data_transform[data_transform < 0] = 0

    # Renombrar las columnas de data_transform
    data_transform.rename(columns=dict(zip(data_transform.columns, obs.columns)), inplace=True)
    quantiles_observed_transformados = np.quantile(data_transform, q=np.linspace(0, 1, num=num_quantile))

    return data_transform, transform_func

# P(pronosticado = x | observado = 0)

def prob_condicional2(df_obs, df_pron, x):
    # Convertir los datos a arreglos NumPy y eliminar los valores NaN
    obs = df_obs.to_numpy()
    pred = df_pron.to_numpy()
    nan_indices = np.logical_or(np.isnan(obs), np.isnan(pred))
    obs = obs[~nan_indices]
    pred = pred[~nan_indices]

    if x in pred:
        # Si x está en datos_pronosticados, calcular normalmente
        num_0_x = np.count_nonzero(np.logical_and(obs == 0, pred == x))
        num_0 = np.count_nonzero(obs == 0)

        if num_0 > 0:
            p_x_0 = num_0_x / num_0
        else:
            p_x_0 = 0
    else:
        # Si x no está en datos_pronosticados, realizar interpolación
        # Encontrar los dos valores más cercanos a x en datos_pronosticados
        diff = np.abs(pred - x)
        # Ordenar las diferencias absolutas de menor a mayor (indices)
        idx = np.argsort(diff)
        # Selecciona los índices de los dos valores más cercanos a x
        closest_indices = idx[:2]
        closest_values = pred[closest_indices]

        # Calcular las probabilidades condicionales para los valores más cercanos a x
        num_0_x = np.count_nonzero(np.logical_and(obs == 0, pred == closest_values[:, None]), axis=1)
        num_0 = np.count_nonzero(obs == 0)
        closest_probabilities = np.where(num_0 > 0, num_0_x / num_0, 0)

        # Interpolar la probabilidad condicional de x como una fracción lineal entre las probabilidades de los valores más cercanos
        p_x_0 = np.interp(x, closest_values, closest_probabilities)

    return p_x_0

#  P( pronosticado = 0 | observado = x ) = (P(observado = x | pronosticado = 0)* P(pronosticado = 0))/P(observado = x)

def prob_condicional_con_bayes(df_obs, df_pron, x):
    # Convertir los datos a arreglos NumPy y eliminar los valores NaN
    obs = df_obs.to_numpy()
    pred = df_pron.to_numpy()
    nan_indices = np.logical_or(np.isnan(obs), np.isnan(pred))
    obs = obs[~nan_indices]
    pred = pred[~nan_indices]

    if x in obs:
        # Si x está en df_obs, calcular normalmente
        num_0 = np.count_nonzero(pred == 0)
        num_x_0 = np.count_nonzero(np.logical_and(obs == x, pred == 0))
        num_x = np.count_nonzero(obs == x)

        if num_x > 0:
            p_x = num_x / len(obs)
            p_x_0 = num_x_0 / num_0
            p_0_x = (p_x_0 * num_0) / (p_x * len(pred))
        else:
            p_0_x = 0
    else:
        # Si x no está en df_obs, realizar interpolación
        # Encontrar los dos valores más cercanos a x en datos_observados
        diff = np.abs(obs - x)
        # Ordenar las diferencias absolutas de menor a mayor (indices)
        idx = np.argsort(diff)
        # Selecciona los índices de los dos valores más cercanos a x
        closest_indices = idx[:2]
        closest_values = obs[closest_indices]

        # Calcular P(pronosticado = 0), P(observado = y ∩ pronosticado = 0) y P(observado = z ∩ pronosticado = 0) (y < x < z)
        num_0 = np.count_nonzero(pred == 0)
        num_y_0 = np.count_nonzero(np.logical_and(obs == closest_values[0], pred == 0))
        num_z_0 = np.count_nonzero(np.logical_and(obs == closest_values[1], pred == 0))
        p_0_y = num_y_0 / num_0
        p_0_z = num_z_0 / num_0

        # Calcular P(observado = x | pronosticado = 0) utilizando interpolación lineal
        closest_probabilities = [p_0_y, p_0_z]
        p_y = np.count_nonzero(obs == closest_values[0]) / len(obs)
        p_z = np.count_nonzero(obs == closest_values[1]) / len(obs)
        p_x = np.interp(x, closest_values, [p_y, p_z]) # P(observado = x)
        p_x_0 = np.interp(x, closest_values, closest_probabilities) # P(observado = x ∩ pronosticado = 0)

        # Calcular P(pronosticado = 0 | observado = x) utilizando la regla de Bayes
        if p_x > 0:
            p_0_x = (p_x_0 * num_0) / (p_x * len(pred))
        else:
            p_0_x = 0

    return p_0_x

# P(pronosticado = x | observado = 0)
def generar_valores1(df_obs, df_pron, df_transformed):
    np.random.seed(33)
    # Crear una máscara para seleccionar las observaciones con valor 0
    obs_mask = (df_obs == 0).to_numpy()

    # Pronosticados = x (todos los posibles valores de x)

    #values_f = np.linspace(start=int(df_p0d.min().min()), stop=int(df_p0d.max().max()), num=(int(df_p0d.max().max()) - int(df_p0d.min().min()) + 1), dtype=int)
    values_f = np.arange(0, int(df_pron.max().max()) + 1)
    #values_f = np.arange(0, 100)

    # Calcular P(pronosticado = x | observado = 0) para cada valor en datos_pronosticados
    p_x_0 = {}
    for x in values_f:
        if x not in p_x_0:
            p_x_0[x] = prob_condicional2(df_obs, df_pron, x)

    # Generar valores aleatorios con base en las probabilidades condicionales;
    probs = [p_x_0[x] if x in p_x_0 else 0 for x in values_f]
    probs = np.array(probs) / np.sum(probs)  # normalizar las probabilidades
    random_values = np.random.choice(values_f, size=obs_mask.sum(), p=probs)

    # Reemplazar los valores generados en el dataframe df_transformed
    df_transformed = df_transformed.copy() # Hacer una copia para evitar modificar el dataframe original
    df_transformed.values[obs_mask] = random_values

    return df_transformed

# P( pronosticado = 0 | observado = x )

def generar_valores2(df_obs, df_pron, df_transformed):
    np.random.seed(64)
    values_obs = np.arange(0, int(df_obs.max().max()) + 1)
    #values_obs = np.arange(1, 100).astype(float)

    # Calcular P(pronosticado = 0 | observado = x) para cada valor x en values_obs
    p_x_0 = {x: prob_condicional_con_bayes(df_obs, df_pron, x) for x in values_obs}
    # Generar un diccionario con las probabilidades condicionales para cada valor de datos_observados
    probs = {value: p_x_0[value] for value in values_obs if value in np.array(df_obs).flatten()}

    df_transformed = df_transformed.copy()

    # Generar valores aleatorios y reemplazar los valores en df_transformed
    for i, value in np.ndenumerate(df_obs):
      if value in probs and not np.isnan(value) and value != 0:
        if probs[value] > np.random.uniform():
            print(np.random.uniform())
            # Reemplazar el valor en df_transformed por cero
            df_transformed.iloc[i[0], i[1]] = 0

    return df_transformed

def generar_valores1_test(df_obs, df_pron, df_obs_test, df_transformed):
    np.random.seed(33)
    # Crear una máscara para seleccionar las observaciones con valor 0
    obs_mask = (df_obs_test == 0).to_numpy()

    # Pronosticados = x (todos los posibles valores de x)
    #values_f = np.linspace(start=0, stop=int(df_p0d.max().max()), num=(int(df_p0d.max().max()) - int(df_p0d.min().min()) + 1), dtype=int)
    values_f = np.arange(0, int(df_pron.max().max()) + 1).astype(float)
    #values_f = np.arange(0, 10000).astype(float)

    # Calcular P(pronosticado = x | observado = 0) para cada valor en datos_pronosticados
    p_x_0 = {}
    for x in values_f:
        if x not in p_x_0:
            p_x_0[x] = prob_condicional2(df_obs, df_pron, x)

    # Generar valores aleatorios con base en las probabilidades condicionales
    probs = [p_x_0[x] if x in p_x_0 else 0 for x in values_f]

    probs = np.array(probs) / np.sum(probs)  # normalizar las probabilidades
    random_values = np.random.choice(values_f, size=obs_mask.sum(), p=probs)

    # Reemplazar los valores generados en el dataframe df_transformed
    df_transformed = df_transformed.copy() # Hacer una copia para evitar modificar el dataframe original
    df_transformed.values[obs_mask] = random_values

    return df_transformed

# P( pronosticado = 0 | observado = x ) PARA TEST

def generar_valores2_test(df_obs, df_pron, df_obs_test, df_transformed):
    np.random.seed(64)

    values_obs = np.arange(0, int(df_obs.max().max()) + 1)
    #values_obs = np.arange(1, 10000).astype(float)

    # Calcular P(pronosticado = 0 | observado = x) para cada valor x en values_obs
    p_x_0 = {x: prob_condicional_con_bayes(df_obs, df_pron, x) for x in values_obs}
    # Generar un diccionario con las probabilidades condicionales para cada valor de datos_observados
    probs = {value: p_x_0[value] for value in values_obs if value in np.array(df_obs).flatten()}
    df_transformed = df_transformed.copy()

    # Generar valores aleatorios y reemplazar los valores en df_transformed
    for i, value in np.ndenumerate(df_obs_test):
      if value in probs and not np.isnan(value) and value != 0:
        if probs[value] > np.random.uniform():
            # Reemplazar el valor en df_transformed por cero
            df_transformed.iloc[i[0], i[1]] = 0

    return df_transformed

# Seleccion de cuencas por zonas geográficas
def seleccion_por_zonas(dataframes, df_names, Tsne):
    zonas = {
        'NG': 'Norte Grande',
        'NC': 'Norte Chico',
        'ZC': 'Zona Central',
        'ZS': 'Zona Sur',
        'ZA': 'Zona Austral'
    }

    for zona, nombre_zona in zonas.items():
        columnas = Tsne.loc[Tsne['Zona'] == nombre_zona, 'gauge_id'].astype('str')
        for df_name, df in zip(df_names, dataframes):
            df_zona = df.loc[:, df.columns.isin(columnas)]
            globals()[f"{df_name}_{zona}"] = df_zona

#Se convierten a df y rename
def convert_and_rename(data, column_names):
    df = pd.DataFrame(data)
    df.rename(columns=dict(zip(df.columns, column_names)), inplace=True)
    return df

# Función LLenado

def llenar_nan(df1, df2):

    # Identificar las columnas comunes entre df1 y df2
    columnas_comunes = df1.columns.intersection(df2.columns)

    # Llenar los NaN en las columnas comunes de df1 con los valores correspondientes en df2
    for columna in columnas_comunes:
        df1[columna].fillna(df2[columna], inplace=True)

    return df1

def guardar_df(dict_dataframes, ruta_drive):
    for nombre, df in dict_dataframes.items():
        ruta_archivo = f"{ruta_drive}/{nombre}.csv"
        df.to_csv(ruta_archivo, index=False)
        print(f"Archivo guardado en: {ruta_archivo}")

def filtrar_hasta_fecha(df, fecha_maxima):
    """
    Esta función filtra el DataFrame para incluir solo los datos hasta una fecha máxima.
    """
    df =  df[df['date'] <= fecha_maxima]

    return df

def apply_lag(df, lag_columns):
    for col in lag_columns:
        if col != 'date':
            df[col] = df[col].shift(-1)
    return df

def imputar(df):

    # Asegurarse de que 'columna_fecha' no sea modificada
    columns_to_interpolate = df.columns.difference(['date'])  
    # Realizar la interpolación lineal
    df[columns_to_interpolate] = df[columns_to_interpolate].interpolate(method='linear', limit_direction='both')    
    # Rellenar los NaN restantes al principio y al final
    df[columns_to_interpolate] = df[columns_to_interpolate].fillna(method='bfill').fillna(method='ffill')
    
    return df

# FUNCIÓN QUE REALIZA LA IMPUTACIÓN SOLO PARA PRECIPITACIÖN
def procesar_datos(p_sel, df1 = None, df2 = None):
    if p_sel in ['p1d', 'p2d', 'p3d','p4d']:
      df_era5 = apply_lag(df1, lag_columns = df1.columns)
      df_era5_rell = apply_lag(df2, lag_columns = df2.columns)

    seleccion_por_zonas(dataframes, df_names, Tsne)
    zonas = ['NG', 'NC', 'ZC', 'ZS', 'ZA']
    num_sel = p_sel[1:]  # Extrae el número de la cadena p_sel

    df_era5_conv = {}
    transform_func = {}

    # Función Quantil-Quantil 2006-2016
    for zona in zonas:
        _, transform_func[f'{num_sel}{zona}'] = quantile_func(globals()[f'df_era5_{zona}'], globals()[f'df_{p_sel}_{zona}'], num_quantile=101)

    df_era5_conv_rell = {}

    # Función transformación 2000-2021 (con rename)
    for zona in zonas:
        df_era5_conv_rell[f'{p_sel}_rell_{zona}'] = transform_func[f'{num_sel}{zona}'](globals()[f'df_era5_rell_{zona}'])

    # Convierte en df y renombra las columnas
    for zona in zonas:
        df_era5_conv_rell[zona] = convert_and_rename(df_era5_conv_rell[f'{p_sel}_rell_{zona}'], globals()[f'df_era5_{zona}'].columns)

    # Generación de Datos
    for zona in zonas:
        # P(pronosticado = x | observado = 0)
        df_era5_conv_rell[zona] = generar_valores1_test(df_obs=globals()[f'df_era5_{zona}'], df_pron=globals()[f'df_{p_sel}_{zona}'], df_obs_test=globals()[f'df_era5_rell_{zona}'], df_transformed=df_era5_conv_rell[zona])

        # P(pronosticado = 0 | observado = x)
        df_era5_conv_rell[zona] = generar_valores2_test(df_obs=globals()[f'df_era5_{zona}'], df_pron=globals()[f'df_{p_sel}_{zona}'], df_obs_test=globals()[f'df_era5_rell_{zona}'], df_transformed=df_era5_conv_rell[zona])

    return df_era5_conv_rell



####################################################################a
####################### Precipitación media #########################
#####################################################################

df_era5_rell = pd.read_csv(f'{Path}/pp_o_era5_pp_mean_b_none_d1_p0d.csv') # SE LE APLICA LA FUNCIÓN TRANSFORMACIÓN
df_era5 = df_era5_rell.copy() # SE UTILIZA PARA CREAR LA FUNCIÓN TRANSFORMACIÓN

# Completar fechas para era5
df_era5_rell = complete_dates(df_era5_rell, '2000-01-01', end_date)
df_era5 = complete_dates(df_era5, '2000-01-01', end_date)

# Filtrar Fechas
df_era5_rell = df_era5_rell[(df_era5_rell['date'] >= '2000-01-01') & (df_era5_rell['date'] <= end_date)].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)
df_era5 = df_era5[(df_era5['date'] >= '2006-10-10') & (df_era5['date'] <= '2016-12-31')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)

# Cargar los archivos csv en una lista de dataframes
dataframes = [pd.read_csv(f'{dir_path}/pp_f_gfs_pp_mean_b_none_d1_p0d.csv'),
              pd.read_csv(f'{dir_path}/pp_f_gfs_pp_mean_b_none_d1_p1d.csv'),
              pd.read_csv(f'{dir_path}/pp_f_gfs_pp_mean_b_none_d1_p2d.csv'),
              pd.read_csv(f'{dir_path}/pp_f_gfs_pp_mean_b_none_d1_p3d.csv'),
              pd.read_csv(f'{dir_path}/pp_f_gfs_pp_mean_b_none_d1_p4d.csv')]

# Eliminar name_id
for df in dataframes:
    df.drop('name_id', axis=1, inplace=True)

# Completar fechas ausentes
dataframes = [complete_dates(df, '2000-01-01', end_date) for df in dataframes]

# Asignar los dataframes actualizados a las variables originales (Estos dfs serán imputados)
pp_f_gfs_pp_mean_b_none_d1_p0d, pp_f_gfs_pp_mean_b_none_d1_p1d, \
pp_f_gfs_pp_mean_b_none_d1_p2d, pp_f_gfs_pp_mean_b_none_d1_p3d, \
pp_f_gfs_pp_mean_b_none_d1_p4d = dataframes

# Con estos dfs se creara la función para imputar
df_p0d, df_p1d, df_p2d, df_p3d, df_p4d = dataframes

# Filtrar las fechas 2006-2016

def filter_df(df):
    df_filtered = df[(df['date'] >= '2006-10-10') & (df['date'] <= '2016-12-31')].iloc[:, 1:].reset_index(drop=True)
    return df_filtered

df_p0d = filter_df(df_p0d)
df_p1d = filter_df(df_p1d)
df_p2d = filter_df(df_p2d)
df_p3d = filter_df(df_p3d)
df_p4d = filter_df(df_p4d)

dataframes = [df_era5, df_era5_rell, df_p0d, df_p1d, df_p2d, df_p3d, df_p4d]
df_names = ['df_era5', 'df_era5_rell', 'df_p0d', 'df_p1d', 'df_p2d', 'df_p3d', 'df_p4d']

#######################################################################
#######################################################################

df0 = procesar_datos('p0d')

# Unir df
df0 = [df0['NG'], df0['NC'], df0['ZC'], df0['ZS'], df0['ZA']]
df_era5_conv_p0d_rellU = pd.concat(df0, axis=1)
# LLenar df
pp_f_gfs_pp_mean_b_none_d1_p0d = llenar_nan(df1 = pp_f_gfs_pp_mean_b_none_d1_p0d, df2 = df_era5_conv_p0d_rellU)

#######################################################################

df1 = procesar_datos('p1d', df1 = df_era5, df2 = df_era5_rell)

# Unir df
df1 = [df1['NG'], df1['NC'], df1['ZC'], df1['ZS'], df1['ZA']]
df_era5_conv_p1d_rellU = pd.concat(df1, axis=1)
# LLenar df
pp_f_gfs_pp_mean_b_none_d1_p1d = llenar_nan(df1 = pp_f_gfs_pp_mean_b_none_d1_p1d, df2 = df_era5_conv_p1d_rellU)
pp_f_gfs_pp_mean_b_none_d1_p1d  = imputar(df = pp_f_gfs_pp_mean_b_none_d1_p1d)

#######################################################################

df2 = procesar_datos('p2d', df1 = df_era5, df2 = df_era5_rell)

# Unir df
df2 = [df2['NG'], df2['NC'], df2['ZC'], df2['ZS'], df2['ZA']]
df_era5_conv_p2d_rellU = pd.concat(df2, axis=1)
# LLenar df
pp_f_gfs_pp_mean_b_none_d1_p2d = llenar_nan(df1 = pp_f_gfs_pp_mean_b_none_d1_p2d, df2 = df_era5_conv_p2d_rellU)
pp_f_gfs_pp_mean_b_none_d1_p2d  = imputar(df = pp_f_gfs_pp_mean_b_none_d1_p2d)
#######################################################################

df3 = procesar_datos('p3d', df1 = df_era5, df2 = df_era5_rell)

# Unir df
df3 = [df3['NG'], df3['NC'], df3['ZC'], df3['ZS'], df3['ZA']]
df_era5_conv_p3d_rellU = pd.concat(df3, axis=1)
# LLenar df
pp_f_gfs_pp_mean_b_none_d1_p3d = llenar_nan(df1 = pp_f_gfs_pp_mean_b_none_d1_p3d, df2 = df_era5_conv_p3d_rellU)
pp_f_gfs_pp_mean_b_none_d1_p3d  = imputar(df = pp_f_gfs_pp_mean_b_none_d1_p3d)

#######################################################################

df4 = procesar_datos('p4d', df1 = df_era5, df2 = df_era5_rell)

# Unir df
df4 = [df4['NG'], df4['NC'], df4['ZC'], df4['ZS'], df4['ZA']]
df_era5_conv_p4d_rellU = pd.concat(df4, axis=1)
# LLenar df
pp_f_gfs_pp_mean_b_none_d1_p4d = llenar_nan(df1 = pp_f_gfs_pp_mean_b_none_d1_p4d, df2 = df_era5_conv_p4d_rellU)
pp_f_gfs_pp_mean_b_none_d1_p4d  = imputar(df = pp_f_gfs_pp_mean_b_none_d1_p4d)

dict_dataframes = {
   "pp_f_gfs_pp_mean_b_none_d1_p0d": pp_f_gfs_pp_mean_b_none_d1_p0d,
   "pp_f_gfs_pp_mean_b_none_d1_p1d": pp_f_gfs_pp_mean_b_none_d1_p1d,
   "pp_f_gfs_pp_mean_b_none_d1_p2d": pp_f_gfs_pp_mean_b_none_d1_p2d,
   "pp_f_gfs_pp_mean_b_none_d1_p3d": pp_f_gfs_pp_mean_b_none_d1_p3d,
   "pp_f_gfs_pp_mean_b_none_d1_p4d": pp_f_gfs_pp_mean_b_none_d1_p4d
}

# Filtrar cada DataFrame en el diccionario
dict_dataframes = {nombre: filtrar_hasta_fecha(df, fecha_maxima)
                             for nombre, df in dict_dataframes.items()}

guardar_df(dict_dataframes, ruta_2)


#####################################################################
####################### Precipitación maxima ########################
#####################################################################

df_era5_rell = pd.read_csv(f'{Path}/pp_o_era5_maxpp_mean_b_none_d1_p0d.csv') # df_era5_rell tiene como objetivo aplicarle la fución trasformación del 2000 al 2016 y de esta manera llear los NaN's
df_era5 = df_era5_rell.copy()

# Completar fechas para era5
df_era5_rell = complete_dates(df_era5_rell, '2000-01-01', end_date)
df_era5 = complete_dates(df_era5, '2000-01-01', end_date)

# Filtrar Fechas
df_era5_rell = df_era5_rell[(df_era5_rell['date'] >= '2000-01-01') & (df_era5_rell['date'] <= end_date)].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)
df_era5 = df_era5[(df_era5['date'] >= '2006-10-10') & (df_era5['date'] <= '2016-12-31')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)

# Cargar los archivos csv en una lista de dataframes

dataframes = [pd.read_csv(f'{dir_path}/pp_f_gfs_pp_max_b_none_d1_p0d.csv'),
              pd.read_csv(f'{dir_path}/pp_f_gfs_pp_max_b_none_d1_p1d.csv'),
              pd.read_csv(f'{dir_path}/pp_f_gfs_pp_max_b_none_d1_p2d.csv'),
              pd.read_csv(f'{dir_path}/pp_f_gfs_pp_max_b_none_d1_p3d.csv'),
              pd.read_csv(f'{dir_path}/pp_f_gfs_pp_max_b_none_d1_p4d.csv')]

# Eliminar gauge_id
for df in dataframes:
    df.drop('name_id', axis=1, inplace=True)

# Completar fechas ausentes
dataframes = [complete_dates(df, '2000-01-01', end_date) for df in dataframes]

# Asignar los dataframes actualizados a las variables originales
pp_f_gfs_pp_max_b_none_d1_p0d, pp_f_gfs_pp_max_b_none_d1_p1d, \
pp_f_gfs_pp_max_b_none_d1_p2d, pp_f_gfs_pp_max_b_none_d1_p3d, \
pp_f_gfs_pp_max_b_none_d1_p4d = dataframes

# Con estos dfs se creara la función para imputar
df_p0d, df_p1d, df_p2d, df_p3d, df_p4d = dataframes


# Filtrar las fechas 2006-2016

def filter_df(df):
    df_filtered = df[(df['date'] >= '2006-10-10') & (df['date'] <= '2016-12-31')].iloc[:, 1:].reset_index(drop=True)
    return df_filtered

df_p0d = filter_df(df_p0d)
df_p1d = filter_df(df_p1d)
df_p2d = filter_df(df_p2d)
df_p3d = filter_df(df_p3d)
df_p4d = filter_df(df_p4d)

dataframes = [df_era5, df_era5_rell, df_p0d, df_p1d, df_p2d, df_p3d, df_p4d]
df_names = ['df_era5', 'df_era5_rell', 'df_p0d', 'df_p1d', 'df_p2d', 'df_p3d', 'df_p4d']

#######################################################################
#######################################################################
df0 = procesar_datos('p0d')

# Unir df
df0 = [df0['NG'], df0['NC'], df0['ZC'], df0['ZS'], df0['ZA']]
df_era5_conv_p0d_rellU = pd.concat(df0, axis=1)
# LLenar df
pp_f_gfs_pp_max_b_none_d1_p0d = llenar_nan(df1 = pp_f_gfs_pp_max_b_none_d1_p0d, df2 = df_era5_conv_p0d_rellU)

#######################################################################
#######################################################################

df1 = procesar_datos('p1d', df1 = df_era5, df2 = df_era5_rell)

# Unir df
df1 = [df1['NG'], df1['NC'], df1['ZC'], df1['ZS'], df1['ZA']]
df_era5_conv_p1d_rellU = pd.concat(df1, axis=1)
# LLenar df
pp_f_gfs_pp_max_b_none_d1_p1d = llenar_nan(df1 = pp_f_gfs_pp_max_b_none_d1_p1d, df2 = df_era5_conv_p1d_rellU)
pp_f_gfs_pp_max_b_none_d1_p1d = imputar(df = pp_f_gfs_pp_max_b_none_d1_p1d)

#######################################################################

df2 = procesar_datos('p2d', df1 = df_era5, df2 = df_era5_rell)

# Unir df
df2 = [df2['NG'], df2['NC'], df2['ZC'], df2['ZS'], df2['ZA']]
df_era5_conv_p2d_rellU = pd.concat(df2, axis=1)
# LLenar df
pp_f_gfs_pp_max_b_none_d1_p2d = llenar_nan(df1 = pp_f_gfs_pp_max_b_none_d1_p2d, df2 = df_era5_conv_p2d_rellU)
pp_f_gfs_pp_max_b_none_d1_p2d = imputar(df = pp_f_gfs_pp_max_b_none_d1_p2d)

#######################################################################
#######################################################################

df3 = procesar_datos('p3d', df1 = df_era5, df2 = df_era5_rell)

# Unir df
df3 = [df3['NG'], df3['NC'], df3['ZC'], df3['ZS'], df3['ZA']]
df_era5_conv_p3d_rellU = pd.concat(df3, axis=1)
# LLenar df
pp_f_gfs_pp_max_b_none_d1_p3d = llenar_nan(df1 = pp_f_gfs_pp_max_b_none_d1_p3d, df2 = df_era5_conv_p3d_rellU)
pp_f_gfs_pp_max_b_none_d1_p3d = imputar(df = pp_f_gfs_pp_max_b_none_d1_p3d)

#######################################################################

df4 = procesar_datos('p4d', df1 = df_era5, df2 = df_era5_rell)

# Unir df
df4 = [df4['NG'], df4['NC'], df4['ZC'], df4['ZS'], df4['ZA']]
df_era5_conv_p4d_rellU = pd.concat(df4, axis=1)
# LLenar df
pp_f_gfs_pp_max_b_none_d1_p4d = llenar_nan(df1 = pp_f_gfs_pp_max_b_none_d1_p4d, df2 = df_era5_conv_p4d_rellU)
pp_f_gfs_pp_max_b_none_d1_p4d = imputar(df = pp_f_gfs_pp_max_b_none_d1_p4d)

dict_dataframes = {
   "pp_f_gfs_pp_max_b_none_d1_p0d": pp_f_gfs_pp_max_b_none_d1_p0d,
   "pp_f_gfs_pp_max_b_none_d1_p1d": pp_f_gfs_pp_max_b_none_d1_p1d,
   "pp_f_gfs_pp_max_b_none_d1_p2d": pp_f_gfs_pp_max_b_none_d1_p2d,
   "pp_f_gfs_pp_max_b_none_d1_p3d": pp_f_gfs_pp_max_b_none_d1_p3d,
   "pp_f_gfs_pp_max_b_none_d1_p4d": pp_f_gfs_pp_max_b_none_d1_p4d
}

# Filtrar cada DataFrame en el diccionario
dict_dataframes = {nombre: filtrar_hasta_fecha(df, fecha_maxima)
                             for nombre, df in dict_dataframes.items()}
guardar_df(dict_dataframes, ruta_2)

#####################################################################
####################### Precipitación PLEN #############################
#####################################################################

df_era5_rell = pd.read_csv(f'{Path}/pp_o_era5_plen_mean_b_none_d1_p0d.csv') # df_era5_rell tiene como objetivo aplicarle la fución trasformación del 2000 al 2021 y de esta maera llear los NaN's
df_era5 = df_era5_rell.copy()

# Completar fechas para era5
df_era5_rell = complete_dates(df_era5_rell, '2000-01-01', end_date)
df_era5 = complete_dates(df_era5, '2000-01-01', end_date)

# Filtrar Fechas
df_era5_rell = df_era5_rell[(df_era5_rell['date'] >= '2000-01-01') & (df_era5_rell['date'] <= end_date)].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)
df_era5 = df_era5[(df_era5['date'] >= '2006-10-10') & (df_era5['date'] <= '2016-12-31')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)

# Cargar los archivos csv en una lista de dataframes
dataframes = [pd.read_csv(f'{dir_path}/pp_f_gfs_plen_mean_b_none_d1_p0d.csv'),
              pd.read_csv(f'{dir_path}/pp_f_gfs_plen_mean_b_none_d1_p1d.csv'),
              pd.read_csv(f'{dir_path}/pp_f_gfs_plen_mean_b_none_d1_p2d.csv'),
              pd.read_csv(f'{dir_path}/pp_f_gfs_plen_mean_b_none_d1_p3d.csv'),
              pd.read_csv(f'{dir_path}/pp_f_gfs_plen_mean_b_none_d1_p4d.csv')]

# Eliminar gauge_id
for df in dataframes:
    df.drop('name_id', axis=1, inplace=True)

# Completar fechas ausentes
dataframes = [complete_dates(df, '2000-01-01', end_date) for df in dataframes]

# Asignar los dataframes actualizados a las variables originales
pp_f_gfs_plen_mean_b_none_d1_p0d, pp_f_gfs_plen_mean_b_none_d1_p1d, \
pp_f_gfs_plen_mean_b_none_d1_p2d, pp_f_gfs_plen_mean_b_none_d1_p3d, \
pp_f_gfs_plen_mean_b_none_d1_p4d = dataframes

# Con estos dfs se creara la función para imputar
df_p0d, df_p1d, df_p2d, df_p3d, df_p4d = dataframes

# Filtrar las fechas 2006-2016

def filter_df(df):
    df_filtered = df[(df['date'] >= '2006-10-10') & (df['date'] <= '2016-12-31')].iloc[:, 1:].reset_index(drop=True)
    return df_filtered

df_p0d = filter_df(df_p0d)
df_p1d = filter_df(df_p1d)
df_p2d = filter_df(df_p2d)
df_p3d = filter_df(df_p3d)
df_p4d = filter_df(df_p4d)
dataframes = [df_era5, df_era5_rell, df_p0d, df_p1d, df_p2d, df_p3d, df_p4d]
df_names = ['df_era5', 'df_era5_rell', 'df_p0d', 'df_p1d', 'df_p2d', 'df_p3d', 'df_p4d']

#######################################################################
df0 = procesar_datos('p0d')

# Unir df
df0 = [df0['NG'], df0['NC'], df0['ZC'], df0['ZS'], df0['ZA']]
df_era5_conv_p0d_rellU = pd.concat(df0, axis=1)
# LLenar df
pp_f_gfs_plen_mean_b_none_d1_p0d = llenar_nan(df1 = pp_f_gfs_plen_mean_b_none_d1_p0d, df2 = df_era5_conv_p0d_rellU)

###########################################################

df1 = procesar_datos('p1d', df1 = df_era5, df2 = df_era5_rell)

# Unir df
df1 = [df1['NG'], df1['NC'], df1['ZC'], df1['ZS'], df1['ZA']]
df_era5_conv_p1d_rellU = pd.concat(df1, axis=1)
# LLenar df
pp_f_gfs_plen_mean_b_none_d1_p1d = llenar_nan(df1 = pp_f_gfs_plen_mean_b_none_d1_p1d, df2 = df_era5_conv_p1d_rellU)
pp_f_gfs_plen_mean_b_none_d1_p1d = imputar(df = pp_f_gfs_plen_mean_b_none_d1_p1d)

###########################################################

df2 = procesar_datos('p2d', df1 = df_era5, df2 = df_era5_rell)

# Unir df
df2 = [df2['NG'], df2['NC'], df2['ZC'], df2['ZS'], df2['ZA']]
df_era5_conv_p2d_rellU = pd.concat(df2, axis=1)
# LLenar df
pp_f_gfs_plen_mean_b_none_d1_p2d = llenar_nan(df1 = pp_f_gfs_plen_mean_b_none_d1_p2d, df2 = df_era5_conv_p2d_rellU)
pp_f_gfs_plen_mean_b_none_d1_p2d = imputar(df = pp_f_gfs_plen_mean_b_none_d1_p2d)

###########################################################

df3 = procesar_datos('p3d', df1 = df_era5, df2 = df_era5_rell)

# Unir df
df3 = [df3['NG'], df3['NC'], df3['ZC'], df3['ZS'], df3['ZA']]
df_era5_conv_p3d_rellU = pd.concat(df3, axis=1)
# LLenar df
pp_f_gfs_plen_mean_b_none_d1_p3d = llenar_nan(df1 = pp_f_gfs_plen_mean_b_none_d1_p3d, df2 = df_era5_conv_p3d_rellU)
pp_f_gfs_plen_mean_b_none_d1_p3d = imputar(df = pp_f_gfs_plen_mean_b_none_d1_p3d)

###########################################################

df4 = procesar_datos('p4d', df1 = df_era5, df2 = df_era5_rell)

# Unir df
df4 = [df4['NG'], df4['NC'], df4['ZC'], df4['ZS'], df4['ZA']]
df_era5_conv_p4d_rellU = pd.concat(df4, axis=1)
# LLenar df
pp_f_gfs_plen_mean_b_none_d1_p4d = llenar_nan(df1 = pp_f_gfs_plen_mean_b_none_d1_p4d, df2 = df_era5_conv_p4d_rellU)
pp_f_gfs_plen_mean_b_none_d1_p4d = imputar(df = pp_f_gfs_plen_mean_b_none_d1_p4d)

dict_dataframes = {
   "pp_f_gfs_plen_mean_b_none_d1_p0d": pp_f_gfs_plen_mean_b_none_d1_p0d,
   "pp_f_gfs_plen_mean_b_none_d1_p1d": pp_f_gfs_plen_mean_b_none_d1_p1d,
   "pp_f_gfs_plen_mean_b_none_d1_p2d": pp_f_gfs_plen_mean_b_none_d1_p2d,
   "pp_f_gfs_plen_mean_b_none_d1_p3d": pp_f_gfs_plen_mean_b_none_d1_p3d,
   "pp_f_gfs_plen_mean_b_none_d1_p4d": pp_f_gfs_plen_mean_b_none_d1_p4d
}

# Filtrar cada DataFrame en el diccionario
dict_dataframes = {nombre: filtrar_hasta_fecha(df, fecha_maxima)
                             for nombre, df in dict_dataframes.items()}

guardar_df(dict_dataframes, ruta_2)

#####################################################################
######################## TEMPERATURA MEAN ###########################
#####################################################################

#Funcion de transformación para variables que no sean precipitación

def quantile_func(observed_data, forecast_data, num_quantile):

    obs = observed_data
    obs[forecast_data.isnull()] = np.nan
    pred = forecast_data
    pred[observed_data.isnull()] = np.nan

    # Quantiles
    quantiles_observed = np.nanquantile(obs, q=np.linspace(0, 1, num=num_quantile))
    quantiles_forecast = np.nanquantile(pred, q=np.linspace(0, 1, num=num_quantile))

    # Se encuentra la función que mapea los cuantiles de una distribución a los cuantiles de la otra
    transform_func = interp1d(quantiles_observed, quantiles_forecast, kind='linear', bounds_error=False, fill_value='extrapolate')

    # Se aplica la función de transformación a los datos observados
    data_transform = pd.DataFrame(transform_func(obs))

    # Renombrar las columnas de data_transform
    data_transform.rename(columns=dict(zip(data_transform.columns, obs.columns)), inplace=True)
    quantiles_observed_transformados = np.quantile(data_transform, q=np.linspace(0, 1, num=num_quantile))

    return data_transform, transform_func

# Cargar los archivos csv en una lista de dataframes

# Observed
tmp_era5 = pd.read_csv(f'{Path}/tmp_o_era5_tmp_mean_b_none_d1_p0d.csv')
tmp_era5_rell = tmp_era5.copy()

# Filtrar Fechas
tmp_era5_rell = tmp_era5_rell[(tmp_era5_rell['date'] >= '2000-01-01') & (tmp_era5_rell['date'] <= 'end_date')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)
tmp_era5 = tmp_era5[(tmp_era5['date'] >= '2006-10-10') & (tmp_era5['date'] <= '2016-12-31')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)

# Forecasted
dataframes = [pd.read_csv(f'{dir_path}/tmp_f_gfs_tmp_mean_b_none_d1_p0d.csv'),
              pd.read_csv(f'{dir_path}/tmp_f_gfs_tmp_mean_b_none_d1_p1d.csv'),
              pd.read_csv(f'{dir_path}/tmp_f_gfs_tmp_mean_b_none_d1_p2d.csv'),
              pd.read_csv(f'{dir_path}/tmp_f_gfs_tmp_mean_b_none_d1_p3d.csv'),
              pd.read_csv(f'{dir_path}/tmp_f_gfs_tmp_mean_b_none_d1_p4d.csv')]

# Eliminar name_id
for df in dataframes:
    df.drop('name_id', axis=1, inplace=True)

# Completar fechas ausentes
dataframes = [complete_dates(df, '2000-01-01', end_date) for df in dataframes]

# Asignar los dataframes actualizados a las variables originales
tmp_f_gfs_tmp_mean_b_none_d1_p0d, tmp_f_gfs_tmp_mean_b_none_d1_p1d, \
tmp_f_gfs_tmp_mean_b_none_d1_p2d, tmp_f_gfs_tmp_mean_b_none_d1_p3d, \
tmp_f_gfs_tmp_mean_b_none_d1_p4d = dataframes

# Con estos dfs se creara la función para imputar
tmp_p0d, tmp_p1d, tmp_p2d, tmp_p3d, tmp_p4d = dataframes

# Filtrar las fechas 
def filter_df(df):
    df_filtered = df[(df['date'] >= '2006-10-10') & (df['date'] <= '2016-12-31')].iloc[:, 1:].reset_index(drop=True)
    return df_filtered

tmp_p0d = filter_df(tmp_p0d)
tmp_p1d = filter_df(tmp_p1d)
tmp_p2d = filter_df(tmp_p2d)
tmp_p3d = filter_df(tmp_p3d)
tmp_p4d = filter_df(tmp_p4d)

def procesar_datos_tmp(prefix, df1 = None, df2 = None):
    if prefix in ['p1d', 'p2d', 'p3d','p4d']:
      tmp_era5 = apply_lag(df1, lag_columns = df1.columns)
      tmp_era5_rell = apply_lag(df2, lag_columns = df2.columns)

    seleccion_por_zonas(dataframes, df_names, Tsne)
    # Función Quantil-Quantil 2006-2016
    zonas = ['NG', 'NC', 'ZC', 'ZS', 'ZA']
    tmp_era5_conv = {}
    transform_func = {}

    for zona in zonas:
        _, transform_func[zona] = quantile_func(globals()[f'tmp_era5_{zona}'], globals()[f'tmp_{prefix}_{zona}'], num_quantile=101)

    # Función Quantil-Quantil 2000-2021 y transformación a df (con rename)
    tmp_era5_conv_rell = {}
    for zona in zonas:
        tmp_era5_conv_rell[zona] = transform_func[zona](globals()[f'tmp_era5_rell_{zona}'])# todo tmp_era5_rell se le aplica ña función transformación y los lleva al dominio de los datos pronosticados

    tmp_era5_conv_rell_df = {zona: convert_and_rename(tmp_era5_conv_rell[zona], globals()[f'tmp_era5_{zona}'].columns) for zona in zonas}

    # Unir df
    tmp = [tmp_era5_conv_rell_df[zona] for zona in zonas]
    tmp_era5_conv_rellU = pd.concat(tmp, axis=1)
    # LLenar df
    tmp_f_gfs_tmp_mean_b_none_d1 = llenar_nan(df1=globals()[f'tmp_f_gfs_tmp_mean_b_none_d1_{prefix}'], df2=tmp_era5_conv_rellU)

    return tmp_f_gfs_tmp_mean_b_none_d1

dataframes = [tmp_era5, tmp_era5_rell, tmp_p0d, tmp_p1d, tmp_p2d, tmp_p3d, tmp_p4d]
df_names = ['tmp_era5', 'tmp_era5_rell','tmp_p0d', 'tmp_p1d', 'tmp_p2d', 'tmp_p3d', 'tmp_p4d']

##################################################################################################

tmp_f_gfs_tmp_mean_b_none_d1_p0d = procesar_datos_tmp('p0d')

tmp_f_gfs_tmp_mean_b_none_d1_p1d = procesar_datos_tmp('p1d', df1 = tmp_era5, df2 = tmp_era5_rell)
tmp_f_gfs_tmp_mean_b_none_d1_p1d  = imputar(df = tmp_f_gfs_tmp_mean_b_none_d1_p1d)

tmp_f_gfs_tmp_mean_b_none_d1_p2d = procesar_datos_tmp('p2d', df1 = tmp_era5, df2 = tmp_era5_rell)
tmp_f_gfs_tmp_mean_b_none_d1_p2d  = imputar(df = tmp_f_gfs_tmp_mean_b_none_d1_p2d)

tmp_f_gfs_tmp_mean_b_none_d1_p3d = procesar_datos_tmp('p3d', df1 = tmp_era5, df2 = tmp_era5_rell)
tmp_f_gfs_tmp_mean_b_none_d1_p3d  = imputar(df = tmp_f_gfs_tmp_mean_b_none_d1_p3d)

tmp_f_gfs_tmp_mean_b_none_d1_p4d = procesar_datos_tmp('p4d', df1 = tmp_era5, df2 = tmp_era5_rell)
tmp_f_gfs_tmp_mean_b_none_d1_p4d  = imputar(df = tmp_f_gfs_tmp_mean_b_none_d1_p4d)

dict_dataframes = {
    "tmp_f_gfs_tmp_mean_b_none_d1_p0d": tmp_f_gfs_tmp_mean_b_none_d1_p0d,
    "tmp_f_gfs_tmp_mean_b_none_d1_p1d": tmp_f_gfs_tmp_mean_b_none_d1_p1d,
    "tmp_f_gfs_tmp_mean_b_none_d1_p2d": tmp_f_gfs_tmp_mean_b_none_d1_p2d,
    "tmp_f_gfs_tmp_mean_b_none_d1_p3d": tmp_f_gfs_tmp_mean_b_none_d1_p3d,
    "tmp_f_gfs_tmp_mean_b_none_d1_p4d": tmp_f_gfs_tmp_mean_b_none_d1_p4d
}

# Filtrar cada DataFrame en el diccionario
dict_dataframes = {nombre: filtrar_hasta_fecha(df, fecha_maxima)
                             for nombre, df in dict_dataframes.items()}

guardar_df(dict_dataframes, ruta_2)


#####################################################################
######################### TEMPERATURA MAX ###########################
#####################################################################

# Cargar los archivos csv en una lista de dataframes

tmp_era5 = pd.read_csv(f'{Path}/tmp_o_era5_tmax_mean_b_none_d1_p0d.csv')
tmp_era5_rell = tmp_era5.copy() 

# Filtrar Fechas
tmp_era5_rell = tmp_era5_rell[(tmp_era5_rell['date'] >= '2000-01-01') & (tmp_era5_rell['date'] <='end_date')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)
tmp_era5 = tmp_era5[(tmp_era5['date'] >= '2006-10-10') & (tmp_era5['date'] <= '2016-12-31')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)

# Forecasted
dataframes = [pd.read_csv(f'{dir_path}/tmp_f_gfs_tmp_max_b_none_d1_p0d.csv'),
              pd.read_csv(f'{dir_path}/tmp_f_gfs_tmp_max_b_none_d1_p1d.csv'),
              pd.read_csv(f'{dir_path}/tmp_f_gfs_tmp_max_b_none_d1_p2d.csv'),
              pd.read_csv(f'{dir_path}/tmp_f_gfs_tmp_max_b_none_d1_p3d.csv'),
              pd.read_csv(f'{dir_path}/tmp_f_gfs_tmp_max_b_none_d1_p4d.csv')]

# Eliminar gauge_id
for df in dataframes:
    df.drop('name_id', axis=1, inplace=True)

# Completar fechas ausentes
dataframes = [complete_dates(df, '2000-01-01', end_date) for df in dataframes]

# Asignar los dataframes actualizados a las variables originales
tmp_f_gfs_tmp_max_b_none_d1_p0d, tmp_f_gfs_tmp_max_b_none_d1_p1d, \
tmp_f_gfs_tmp_max_b_none_d1_p2d, tmp_f_gfs_tmp_max_b_none_d1_p3d, \
tmp_f_gfs_tmp_max_b_none_d1_p4d = dataframes

# Con estos dfs se creara la función para imputar
tmp_p0d, tmp_p1d, tmp_p2d, tmp_p3d, tmp_p4d = dataframes

# Filtrar las fechas 
def filter_df(df):
    df_filtered = df[(df['date'] >= '2006-10-10') & (df['date'] <= '2016-12-31')].iloc[:, 1:].reset_index(drop=True)
    return df_filtered

tmp_p0d = filter_df(tmp_p0d)
tmp_p1d = filter_df(tmp_p1d)
tmp_p2d = filter_df(tmp_p2d)
t3d = filter_df(tmp_p3d)
tmp_p4d = filter_df(tmp_p4d)

def procesar_datos_tmp_max(prefix, df1 = None, df2 = None):
    if prefix in ['p1d', 'p2d', 'p3d','p4d']:
      tmp_era5 = apply_lag(df1, lag_columns = df1.columns)
      tmp_era5_rell = apply_lag(df2, lag_columns = df2.columns)

    seleccion_por_zonas(dataframes, df_names, Tsne)
    # Función Quantil-Quantil 2006-2016
    zonas = ['NG', 'NC', 'ZC', 'ZS', 'ZA']
    tmp_era5_conv = {}
    transform_func = {}

    for zona in zonas:
        tmp_era5_conv[zona], transform_func[zona] = quantile_func(globals()[f'tmp_era5_{zona}'], globals()[f'tmp_{prefix}_{zona}'], num_quantile=101)

    # Función Quantil-Quantil 2000-2021 y transformación a df (con rename)
    tmp_era5_conv_rell = {}
    for zona in zonas:
        tmp_era5_conv_rell[zona] = transform_func[zona](globals()[f'tmp_era5_rell_{zona}'])

    tmp_era5_conv_rell_df = {zona: convert_and_rename(tmp_era5_conv_rell[zona], globals()[f'tmp_era5_{zona}'].columns) for zona in zonas}

    # Unir df
    tmp = [tmp_era5_conv_rell_df[zona] for zona in zonas]
    tmp_era5_conv_rellU = pd.concat(tmp, axis=1)
    # LLenar df
    tmp_f_gfs_tmp_max_b_none_d1 = llenar_nan(df1=globals()[f'tmp_f_gfs_tmp_max_b_none_d1_{prefix}'], df2=tmp_era5_conv_rellU)

    return tmp_f_gfs_tmp_max_b_none_d1

# División en zonas geográficas

dataframes = [tmp_era5, tmp_era5_rell, tmp_p0d, tmp_p1d, tmp_p2d, tmp_p3d, tmp_p4d]
df_names = ['tmp_era5', 'tmp_era5_rell','tmp_p0d', 'tmp_p1d', 'tmp_p2d', 'tmp_p3d', 'tmp_p4d']

##################################################################################################
# Aplicación de la función

tmp_f_gfs_tmp_max_b_none_d1_p0d = procesar_datos_tmp_max('p0d')

tmp_f_gfs_tmp_max_b_none_d1_p1d = procesar_datos_tmp_max('p1d', df1 = tmp_era5, df2 = tmp_era5_rell)
tmp_f_gfs_tmp_max_b_none_d1_p1d  = imputar(df = tmp_f_gfs_tmp_max_b_none_d1_p1d)

tmp_f_gfs_tmp_max_b_none_d1_p2d = procesar_datos_tmp_max('p2d', df1 = tmp_era5, df2 = tmp_era5_rell)
tmp_f_gfs_tmp_max_b_none_d1_p2d  = imputar(df = tmp_f_gfs_tmp_max_b_none_d1_p2d)

tmp_f_gfs_tmp_max_b_none_d1_p3d = procesar_datos_tmp_max('p3d', df1 = tmp_era5, df2 = tmp_era5_rell)
tmp_f_gfs_tmp_max_b_none_d1_p3d  = imputar(df = tmp_f_gfs_tmp_max_b_none_d1_p3d)

tmp_f_gfs_tmp_max_b_none_d1_p4d = procesar_datos_tmp_max('p4d', df1 = tmp_era5, df2 = tmp_era5_rell)
tmp_f_gfs_tmp_max_b_none_d1_p4d  = imputar(df = tmp_f_gfs_tmp_max_b_none_d1_p4d)

dict_dataframes = {
    "tmp_f_gfs_tmp_max_b_none_d1_p0d": tmp_f_gfs_tmp_max_b_none_d1_p0d,
    "tmp_f_gfs_tmp_max_b_none_d1_p1d": tmp_f_gfs_tmp_max_b_none_d1_p1d,
    "tmp_f_gfs_tmp_max_b_none_d1_p2d": tmp_f_gfs_tmp_max_b_none_d1_p2d,
    "tmp_f_gfs_tmp_max_b_none_d1_p3d": tmp_f_gfs_tmp_max_b_none_d1_p3d,
    "tmp_f_gfs_tmp_max_b_none_d1_p4d": tmp_f_gfs_tmp_max_b_none_d1_p4d
}

# Filtrar cada DataFrame en el diccionario
dict_dataframes = {nombre: filtrar_hasta_fecha(df, fecha_maxima)
                             for nombre, df in dict_dataframes.items()}

guardar_df(dict_dataframes, ruta_2)

#####################################################################
######################### TEMPERATURA MIN ###########################
#####################################################################

# Observed
tmp_era5 = pd.read_csv(f'{Path}/tmp_o_era5_tmin_mean_b_none_d1_p0d.csv')
tmp_era5_rell = tmp_era5.copy()

# Filtrar Fechas
tmp_era5_rell = tmp_era5_rell[(tmp_era5_rell['date'] >= '2000-01-01') & (tmp_era5_rell['date'] <= 'end_date')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)
tmp_era5 = tmp_era5[(tmp_era5['date'] >= '2006-10-10') & (tmp_era5['date'] <= '2016-12-31')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)

# Forecasted
dataframes = [pd.read_csv(f'{dir_path}/tmp_f_gfs_tmp_min_b_none_d1_p0d.csv'),
              pd.read_csv(f'{dir_path}/tmp_f_gfs_tmp_min_b_none_d1_p1d.csv'),
              pd.read_csv(f'{dir_path}/tmp_f_gfs_tmp_min_b_none_d1_p2d.csv'),
              pd.read_csv(f'{dir_path}/tmp_f_gfs_tmp_min_b_none_d1_p3d.csv'),
              pd.read_csv(f'{dir_path}/tmp_f_gfs_tmp_min_b_none_d1_p4d.csv')
              ]

# Eliminar gauge_id
for df in dataframes:
    df.drop('name_id', axis=1, inplace=True)

# Completar fechas ausentes
dataframes = [complete_dates(df, '2000-01-01', end_date) for df in dataframes]

# Asignar los dataframes actualizados a las variables originales
tmp_f_gfs_tmp_min_b_none_d1_p0d, tmp_f_gfs_tmp_min_b_none_d1_p1d, \
tmp_f_gfs_tmp_min_b_none_d1_p2d, tmp_f_gfs_tmp_min_b_none_d1_p3d, \
tmp_f_gfs_tmp_min_b_none_d1_p4d = dataframes

# Con estos dfs se creara la función para imputar
tmp_p0d, tmp_p1d, tmp_p2d, tmp_p3d, tmp_p4d = dataframes

# Filtrar las fechas 
def filter_df(df):
    df_filtered = df[(df['date'] >= '2006-10-10') & (df['date'] <= '2016-12-31')].iloc[:, 1:].reset_index(drop=True)
    return df_filtered

tmp_p0d = filter_df(tmp_p0d)
tmp_p1d = filter_df(tmp_p1d)
tmp_p2d = filter_df(tmp_p2d)
tmp_p3d = filter_df(tmp_p3d)
tmp_p4d = filter_df(tmp_p4d)

# Función ajuste

def procesar_datos_tmp_min(prefix, df1 = None, df2 = None):
    if prefix in ['p1d', 'p2d', 'p3d','p4d']:
      tmp_era5 = apply_lag(df1, lag_columns = df1.columns)
      tmp_era5_rell = apply_lag(df2, lag_columns = df2.columns)

    seleccion_por_zonas(dataframes, df_names, Tsne)
    # Función Quantil-Quantil 2006-2016
    zonas = ['NG', 'NC', 'ZC', 'ZS', 'ZA']
    tmp_era5_conv = {}
    transform_func = {}

    for zona in zonas:
        tmp_era5_conv[zona], transform_func[zona] = quantile_func(globals()[f'tmp_era5_{zona}'], globals()[f'tmp_{prefix}_{zona}'], num_quantile=101)

    # Función Quantil-Quantil 2000-2021 y transformación a df (con rename)
    tmp_era5_conv_rell = {}
    for zona in zonas:
        tmp_era5_conv_rell[zona] = transform_func[zona](globals()[f'tmp_era5_rell_{zona}'])

    tmp_era5_conv_rell_df = {zona: convert_and_rename(tmp_era5_conv_rell[zona], globals()[f'tmp_era5_{zona}'].columns) for zona in zonas}

    # Unir df
    tmp = [tmp_era5_conv_rell_df[zona] for zona in zonas]
    tmp_era5_conv_rellU = pd.concat(tmp, axis=1)
    # LLenar df
    tmp_f_gfs_tmp_min_b_none_d1 = llenar_nan(df1=globals()[f'tmp_f_gfs_tmp_min_b_none_d1_{prefix}'], df2=tmp_era5_conv_rellU)

    return tmp_f_gfs_tmp_min_b_none_d1

# División en zonas geográficas
dataframes = [tmp_era5, tmp_era5_rell, tmp_p0d, tmp_p1d, tmp_p2d, tmp_p3d, tmp_p4d]
df_names = ['tmp_era5', 'tmp_era5_rell','tmp_p0d', 'tmp_p1d', 'tmp_p2d', 'tmp_p3d', 'tmp_p4d']

##################################################################################################
# Aplicación de la función
tmp_f_gfs_tmp_min_b_none_d1_p0d = procesar_datos_tmp_min('p0d')

tmp_f_gfs_tmp_min_b_none_d1_p1d = procesar_datos_tmp_min('p1d', df1 = tmp_era5, df2 = tmp_era5_rell)
tmp_f_gfs_tmp_min_b_none_d1_p1d  = imputar(df = tmp_f_gfs_tmp_min_b_none_d1_p1d)

tmp_f_gfs_tmp_min_b_none_d1_p2d = procesar_datos_tmp_min('p2d', df1 = tmp_era5, df2 = tmp_era5_rell)
tmp_f_gfs_tmp_min_b_none_d1_p2d  = imputar(df = tmp_f_gfs_tmp_min_b_none_d1_p2d)

tmp_f_gfs_tmp_min_b_none_d1_p3d = procesar_datos_tmp_min('p3d', df1 = tmp_era5, df2 = tmp_era5_rell)
tmp_f_gfs_tmp_min_b_none_d1_p3d  = imputar(df = tmp_f_gfs_tmp_min_b_none_d1_p3d)

tmp_f_gfs_tmp_min_b_none_d1_p4d = procesar_datos_tmp_min('p4d', df1 = tmp_era5, df2 = tmp_era5_rell)
tmp_f_gfs_tmp_min_b_none_d1_p4d  = imputar(df = tmp_f_gfs_tmp_min_b_none_d1_p4d)


dict_dataframes = {
    "tmp_f_gfs_tmp_min_b_none_d1_p0d": tmp_f_gfs_tmp_min_b_none_d1_p0d,
    "tmp_f_gfs_tmp_min_b_none_d1_p1d": tmp_f_gfs_tmp_min_b_none_d1_p1d,
    "tmp_f_gfs_tmp_min_b_none_d1_p2d": tmp_f_gfs_tmp_min_b_none_d1_p2d,
    "tmp_f_gfs_tmp_min_b_none_d1_p3d": tmp_f_gfs_tmp_min_b_none_d1_p3d,
    "tmp_f_gfs_tmp_min_b_none_d1_p4d": tmp_f_gfs_tmp_min_b_none_d1_p4d
}


# Filtrar cada DataFrame en el diccionario
dict_dataframes = {nombre: filtrar_hasta_fecha(df, fecha_maxima)
                             for nombre, df in dict_dataframes.items()}
guardar_df(dict_dataframes, ruta_2)

#####################################################################
########################## ATMOSPHERIC UW ###########################
#####################################################################

# Observed

atm_uw = pd.read_csv(f'{Path}/atm_o_era5_uw_mean_b_none_d1_p0d.csv')
atm_uw_rell = atm_uw.copy()

# Filtrar Fechas
atm_uw_rell = atm_uw_rell[(atm_uw_rell['date'] >= '2000-01-01') & (atm_uw_rell['date'] <= 'end_date')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)
atm_uw = atm_uw[(atm_uw['date'] >= '2006-10-10') & (atm_uw['date'] <= '2016-12-31')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)

# Cargar los archivos csv en una lista de dataframes
dataframes = [pd.read_csv(f'{dir_path}/atm_f_gfs_uw_mean_b_none_d1_p0d.csv'),
              pd.read_csv(f'{dir_path}/atm_f_gfs_uw_mean_b_none_d1_p1d.csv'),
              pd.read_csv(f'{dir_path}/atm_f_gfs_uw_mean_b_none_d1_p2d.csv'),
              pd.read_csv(f'{dir_path}/atm_f_gfs_uw_mean_b_none_d1_p3d.csv'),
              pd.read_csv(f'{dir_path}/atm_f_gfs_uw_mean_b_none_d1_p4d.csv')]

# Eliminar gauge_id
for df in dataframes:
    df.drop('name_id', axis=1, inplace=True)

# Completar fechas ausentes
dataframes = [complete_dates(df, '2000-01-01', end_date) for df in dataframes]

# Asignar los dataframes actualizados a las variables originales
atm_f_gfs_uw_mean_b_none_d1_p0d, atm_f_gfs_uw_mean_b_none_d1_p1d, \
atm_f_gfs_uw_mean_b_none_d1_p2d, atm_f_gfs_uw_mean_b_none_d1_p3d, \
atm_f_gfs_uw_mean_b_none_d1_p4d = dataframes

# Con estos dfs se creara la función para imputar
atm_p0d, atm_p1d, atm_p2d, atm_p3d, atm_p4d = dataframes

# Filtrar las fechas 
def filter_df(df):
    df_filtered = df[(df['date'] >= '2006-10-10') & (df['date'] <= '2016-12-31')].iloc[:, 1:].reset_index(drop=True)
    return df_filtered

atm_p0d = filter_df(atm_p0d)
atm_p1d = filter_df(atm_p1d)
atm_p2d = filter_df(atm_p2d)
atm_p3d = filter_df(atm_p3d)
atm_p4d = filter_df(atm_p4d)

# Función ajuste

def procesar_datos_atm(prefix, df1 = None, df2 = None):
    if prefix in ['p1d', 'p2d', 'p3d','p4d']:
      atm_uw = apply_lag(df1, lag_columns = df1.columns)
      atm_uw_rell = apply_lag(df2, lag_columns = df2.columns)

    seleccion_por_zonas(dataframes, df_names, Tsne)
    # Función Quantil-Quantil 2006-2016
    zonas = ['NG', 'NC', 'ZC', 'ZS', 'ZA']
    atm_uw_conv = {}
    transform_func = {}

    for zona in zonas:
        _, transform_func[zona] = quantile_func(globals()[f'atm_uw_{zona}'], globals()[f'atm_{prefix}_{zona}'], num_quantile=101)

    # Función Quantil-Quantil 2000-2021 y transformación a df (con rename)
    atm_uw_conv_rell = {}
    for zona in zonas:
        atm_uw_conv_rell[zona] = transform_func[zona](globals()[f'atm_uw_rell_{zona}'])

    atm_uw_conv_rell_df = {zona: convert_and_rename(atm_uw_conv_rell[zona], globals()[f'atm_uw_{zona}'].columns) for zona in zonas}

    # Unir df
    atm = [atm_uw_conv_rell_df[zona] for zona in zonas]
    atm_uw_conv_rellU = pd.concat(atm, axis=1)
    # LLenar df
    atm_f_gfs_uw_mean_b_none_d1 = llenar_nan(df1=globals()[f'atm_f_gfs_uw_mean_b_none_d1_{prefix}'], df2=atm_uw_conv_rellU)

    return atm_f_gfs_uw_mean_b_none_d1

# División en zonas geográficas

dataframes = [atm_uw, atm_uw_rell, atm_p0d, atm_p1d, atm_p2d, atm_p3d, atm_p4d]
df_names = ['atm_uw', 'atm_uw_rell','atm_p0d', 'atm_p1d', 'atm_p2d', 'atm_p3d', 'atm_p4d']

##################################################################################################
# Aplicación de la función

atm_f_gfs_uw_mean_b_none_d1_p0d = procesar_datos_atm('p0d')

atm_f_gfs_uw_mean_b_none_d1_p1d = procesar_datos_atm('p1d', df1 = atm_uw, df2 = atm_uw_rell)
atm_f_gfs_uw_mean_b_none_d1_p1d  = imputar(df = atm_f_gfs_uw_mean_b_none_d1_p1d)

atm_f_gfs_uw_mean_b_none_d1_p2d = procesar_datos_atm('p2d', df1 = atm_uw, df2 = atm_uw_rell)
atm_f_gfs_uw_mean_b_none_d1_p2d  = imputar(df = atm_f_gfs_uw_mean_b_none_d1_p2d)

atm_f_gfs_uw_mean_b_none_d1_p3d = procesar_datos_atm('p3d', df1 = atm_uw, df2 = atm_uw_rell)
atm_f_gfs_uw_mean_b_none_d1_p3d  = imputar(df = atm_f_gfs_uw_mean_b_none_d1_p3d)

atm_f_gfs_uw_mean_b_none_d1_p4d = procesar_datos_atm('p4d', df1 = atm_uw, df2 = atm_uw_rell)
atm_f_gfs_uw_mean_b_none_d1_p4d  = imputar(df = atm_f_gfs_uw_mean_b_none_d1_p4d)

dict_dataframes = {
    "atm_f_gfs_uw_mean_b_none_d1_p0d": atm_f_gfs_uw_mean_b_none_d1_p0d,
    "atm_f_gfs_uw_mean_b_none_d1_p1d": atm_f_gfs_uw_mean_b_none_d1_p1d,
    "atm_f_gfs_uw_mean_b_none_d1_p2d": atm_f_gfs_uw_mean_b_none_d1_p2d,
    "atm_f_gfs_uw_mean_b_none_d1_p3d": atm_f_gfs_uw_mean_b_none_d1_p3d,
    "atm_f_gfs_uw_mean_b_none_d1_p4d": atm_f_gfs_uw_mean_b_none_d1_p4d
}


# Filtrar cada DataFrame en el diccionario
dict_dataframes = {nombre: filtrar_hasta_fecha(df, fecha_maxima)
                             for nombre, df in dict_dataframes.items()}
guardar_df(dict_dataframes, ruta_2)

#####################################################################
#########################  ATMOSPHERIC VW ###########################
#####################################################################

# Observed
atm_vw = pd.read_csv(f'{Path}/atm_o_era5_vw_mean_b_none_d1_p0d.csv')
atm_vw_rell = atm_vw.copy()

# Filtrar Fechas
# Filtrar Fechas
atm_vw_rell = atm_vw_rell[(atm_vw_rell['date'] >= '2000-01-01') & (atm_vw_rell['date'] <= 'end_date')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)
atm_vw = atm_vw[(atm_vw['date'] >= '2006-10-10') & (atm_vw['date'] <= '2016-12-31')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)

# Cargar los archivos csv en una lista de dataframes
dataframes = [pd.read_csv(f'{dir_path}/atm_f_gfs_vw_mean_b_none_d1_p0d.csv'),
              pd.read_csv(f'{dir_path}/atm_f_gfs_vw_mean_b_none_d1_p1d.csv'),
              pd.read_csv(f'{dir_path}/atm_f_gfs_vw_mean_b_none_d1_p2d.csv'),
              pd.read_csv(f'{dir_path}/atm_f_gfs_vw_mean_b_none_d1_p3d.csv'),
              pd.read_csv(f'{dir_path}/atm_f_gfs_vw_mean_b_none_d1_p4d.csv')]

# Eliminar gauge_id
for df in dataframes:
    df.drop('name_id', axis=1, inplace=True)

# Completar fechas ausentes
dataframes = [complete_dates(df, '2000-01-01', end_date) for df in dataframes]

# Asigns dataframes actualizados a las variables originales
atm_f_gfs_vw_mean_b_none_d1_p0d, atm_f_gfs_vw_mean_b_none_d1_p1d, \
atm_f_gfs_vw_mean_b_none_d1_p2d, atm_f_gfs_vw_mean_b_none_d1_p3d, \
atm_f_gfs_vw_mean_b_none_d1_p4d = dataframes

## Con estos dfs se creara la función para imputar
atm_p0d, atm_p1d, atm_p2d, atm_p3d, atm_p4d = dataframes

# Filtrar las fechas 
def filter_df(df):
    df_filtered = df[(df['date'] >= '2006-10-10') & (df['date'] <= '2016-12-31')].iloc[:, 1:].reset_index(drop=True)
    return df_filtered

atm_p0d = filter_df(atm_p0d)
atm_p1d = filter_df(atm_p1d)
atm_p2d = filter_df(atm_p2d)
atm_p3d = filter_df(atm_p3d)
atm_p4d = filter_df(atm_p4d)

# Función ajuste

def procesar_datos_atm(prefix, df1 = None, df2 = None):
    if prefix in ['p1d', 'p2d', 'p3d','p4d']:
      atm_vw = apply_lag(df1, lag_columns = df1.columns)
      atm_vw_rell = apply_lag(df2, lag_columns = df2.columns)

    seleccion_por_zonas(dataframes, df_names, Tsne)
    # Función Quantil-Quantil 2006-2016
    zonas = ['NG', 'NC', 'ZC', 'ZS', 'ZA']
    atm_vw_conv = {}
    transform_func = {}

    for zona in zonas:
        _, transform_func[zona] = quantile_func(globals()[f'atm_vw_{zona}'], globals()[f'atm_{prefix}_{zona}'], num_quantile=101)

    # Función Quantil-Quantil 2000-2021 y transformación a df (con rename)
    atm_vw_conv_rell = {}
    for zona in zonas:
        atm_vw_conv_rell[zona] = transform_func[zona](globals()[f'atm_vw_rell_{zona}'])

    atm_vw_conv_rell_df = {zona: convert_and_rename(atm_vw_conv_rell[zona], globals()[f'atm_vw_{zona}'].columns) for zona in zonas}

    # Unir df
    atm = [atm_vw_conv_rell_df[zona] for zona in zonas]
    atm_vw_conv_rellU = pd.concat(atm, axis=1)
    # LLenar df
    atm_f_gfs_vw_mean_b_none_d1 = llenar_nan(df1=globals()[f'atm_f_gfs_vw_mean_b_none_d1_{prefix}'], df2=atm_vw_conv_rellU)

    return atm_f_gfs_vw_mean_b_none_d1

# División en zonas geográficas

dataframes = [atm_vw, atm_vw_rell, atm_p0d, atm_p1d, atm_p2d, atm_p3d, atm_p4d]
df_names = ['atm_vw', 'atm_vw_rell','atm_p0d', 'atm_p1d', 'atm_p2d', 'atm_p3d', 'atm_p4d']

##################################################################################################
# Aplicación de la función

atm_f_gfs_vw_mean_b_none_d1_p0d = procesar_datos_atm('p0d')
atm_f_gfs_vw_mean_b_none_d1_p1d = procesar_datos_atm('p1d', df1 = atm_vw, df2 = atm_vw_rell)
atm_f_gfs_vw_mean_b_none_d1_p1d  = imputar(df = atm_f_gfs_vw_mean_b_none_d1_p1d)
atm_f_gfs_vw_mean_b_none_d1_p2d = procesar_datos_atm('p2d', df1 = atm_vw, df2 = atm_vw_rell)
atm_f_gfs_vw_mean_b_none_d1_p2d  = imputar(df = atm_f_gfs_vw_mean_b_none_d1_p2d)
atm_f_gfs_vw_mean_b_none_d1_p3d = procesar_datos_atm('p3d', df1 = atm_vw, df2 = atm_vw_rell)
atm_f_gfs_vw_mean_b_none_d1_p3d  = imputar(df = atm_f_gfs_vw_mean_b_none_d1_p3d)
atm_f_gfs_vw_mean_b_none_d1_p4d = procesar_datos_atm('p4d', df1 = atm_vw, df2 = atm_vw_rell)
atm_f_gfs_vw_mean_b_none_d1_p4d  = imputar(df = atm_f_gfs_vw_mean_b_none_d1_p4d)

dict_dataframes = {
    "atm_f_gfs_vw_mean_b_none_d1_p0d": atm_f_gfs_vw_mean_b_none_d1_p0d,
    "atm_f_gfs_vw_mean_b_none_d1_p1d": atm_f_gfs_vw_mean_b_none_d1_p1d,
    "atm_f_gfs_vw_mean_b_none_d1_p2d": atm_f_gfs_vw_mean_b_none_d1_p2d,
    "atm_f_gfs_vw_mean_b_none_d1_p3d": atm_f_gfs_vw_mean_b_none_d1_p3d,
    "atm_f_gfs_vw_mean_b_none_d1_p4d": atm_f_gfs_vw_mean_b_none_d1_p4d
}

# Filtrar cada DataFrame en el diccionario
dict_dataframes = {nombre: filtrar_hasta_fecha(df, fecha_maxima)
                             for nombre, df in dict_dataframes.items()}
guardar_df(dict_dataframes, ruta_2)

#####################################################################
######################### ATMOSPHERIC GH  ###########################
#####################################################################

# Observed
atm_gh = pd.read_csv(f'{Path}/atm_o_era5_z_mean_b_none_d1_p0d.csv')
atm_gh_rell = atm_gh.copy()

# Filtrar Fechas
atm_gh_rell = atm_gh_rell[(atm_gh_rell['date'] >= '2000-01-01') & (atm_gh_rell['date'] <='end_date')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)
atm_gh = atm_gh[(atm_gh['date'] >= '2006-10-10') & (atm_gh['date'] <= '2016-12-31')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)

# Filtrar Fechas
dataframes = [pd.read_csv(f'{dir_path}/atm_f_gfs_gh_mean_b_none_d1_p0d.csv'),
              pd.read_csv(f'{dir_path}/atm_f_gfs_gh_mean_b_none_d1_p1d.csv'),
              pd.read_csv(f'{dir_path}/atm_f_gfs_gh_mean_b_none_d1_p2d.csv'),
              pd.read_csv(f'{dir_path}/atm_f_gfs_gh_mean_b_none_d1_p3d.csv'),
              pd.read_csv(f'{dir_path}/atm_f_gfs_gh_mean_b_none_d1_p4d.csv')]

# Eliminar gauge_id
for df in dataframes:
    df.drop('name_id', axis=1, inplace=True)

# Completar fechas ausentes
dataframes = [complete_dates(df, '2000-01-01', end_date) for df in dataframes]

# Asignar los dataframes actualizados a las variables originales
atm_f_gfs_gh_mean_b_none_d1_p0d, atm_f_gfs_gh_mean_b_none_d1_p1d, \
atm_f_gfs_gh_mean_b_none_d1_p2d, atm_f_gfs_gh_mean_b_none_d1_p3d, \
atm_f_gfs_gh_mean_b_none_d1_p4d = dataframes

## Con estos dfs se creara la función para imputar
atm_p0d, atm_p1d, atm_p2d, atm_p3d, atm_p4d = dataframes

# Filtrar las fechas 
def filter_df(df):
    df_filtered = df[(df['date'] >= '2006-10-10') & (df['date'] <= '2016-12-31')].iloc[:, 1:].reset_index(drop=True)
    return df_filtered

atm_p0d = filter_df(atm_p0d)
atm_p1d = filter_df(atm_p1d)
atm_p2d = filter_df(atm_p2d)
atm_p3d = filter_df(atm_p3d)
atm_p4d = filter_df(atm_p4d)

# Función ajuste

def procesar_datos_atm(prefix, df1 = None, df2 = None):
    if prefix in ['p1d', 'p2d', 'p3d','p4d']:
      atm_gh = apply_lag(df1, lag_columns = df1.columns)
      atm_gh_rell = apply_lag(df2, lag_columns = df2.columns)

    seleccion_por_zonas(dataframes, df_names, Tsne)
    # Función Quantil-Quantil 2006-2016
    zonas = ['NG', 'NC', 'ZC', 'ZS', 'ZA']
    atm_gh_conv = {}
    transform_func = {}

    for zona in zonas:
        _, transform_func[zona] = quantile_func(globals()[f'atm_gh_{zona}'], globals()[f'atm_{prefix}_{zona}'], num_quantile=101) #(observed_data, forecast_data, num_quantile)

    # Función Quantil-Quantil 2000-2021 y transformación a df (con rename)
    atm_gh_conv_rell = {}
    for zona in zonas:
        atm_gh_conv_rell[zona] = transform_func[zona](globals()[f'atm_gh_rell_{zona}'])

    atm_gh_conv_rell_df = {zona: convert_and_rename(atm_gh_conv_rell[zona], globals()[f'atm_gh_{zona}'].columns) for zona in zonas}

    # Unir df
    atm_gh = [atm_gh_conv_rell_df[zona] for zona in zonas]
    atm_gh_conv_rellU = pd.concat(atm_gh, axis=1)

    # LLenar df
    atm_f_gfs_gh_mean_b_none_d1 = llenar_nan(df1=globals()[f'atm_f_gfs_gh_mean_b_none_d1_{prefix}'], df2 = atm_gh_conv_rellU)

    return atm_f_gfs_gh_mean_b_none_d1

# División en zonas geográficas

dataframes = [atm_gh, atm_gh_rell, atm_p0d, atm_p1d, atm_p2d, atm_p3d, atm_p4d]
df_names = ['atm_gh', 'atm_gh_rell','atm_p0d', 'atm_p1d', 'atm_p2d', 'atm_p3d', 'atm_p4d']

##################################################################################################
# Aplicación de la func
atm_f_gfs_gh_mean_b_none_d1_p0d = procesar_datos_atm('p0d')
atm_f_gfs_gh_mean_b_none_d1_p1d = procesar_datos_atm('p1d', df1 = atm_gh, df2 = atm_gh_rell)
atm_f_gfs_gh_mean_b_none_d1_p1d  = imputar(df = atm_f_gfs_gh_mean_b_none_d1_p1d)
atm_f_gfs_gh_mean_b_none_d1_p2d = procesar_datos_atm('p2d', df1 = atm_gh, df2 = atm_gh_rell)
atm_f_gfs_gh_mean_b_none_d1_p2d  = imputar(df = atm_f_gfs_gh_mean_b_none_d1_p2d)
atm_f_gfs_gh_mean_b_none_d1_p3d = procesar_datos_atm('p3d', df1 = atm_gh, df2 = atm_gh_rell)
atm_f_gfs_gh_mean_b_none_d1_p3d  = imputar(df = atm_f_gfs_gh_mean_b_none_d1_p3d)
atm_f_gfs_gh_mean_b_none_d1_p4d = procesar_datos_atm('p4d', df1 = atm_gh, df2 = atm_gh_rell)
atm_f_gfs_gh_mean_b_none_d1_p4d  = imputar(df = atm_f_gfs_gh_mean_b_none_d1_p4d)

dict_dataframes = {
    "atm_f_gfs_gh_mean_b_none_d1_p0d": atm_f_gfs_gh_mean_b_none_d1_p0d,
    "atm_f_gfs_gh_mean_b_none_d1_p1d": atm_f_gfs_gh_mean_b_none_d1_p1d,
    "atm_f_gfs_gh_mean_b_none_d1_p2d": atm_f_gfs_gh_mean_b_none_d1_p2d,
    "atm_f_gfs_gh_mean_b_none_d1_p3d": atm_f_gfs_gh_mean_b_none_d1_p3d,
    "atm_f_gfs_gh_mean_b_none_d1_p4d": atm_f_gfs_gh_mean_b_none_d1_p4d
}

# Filtrar cada DataFrame en el diccionario
dict_dataframes = {nombre: filtrar_hasta_fecha(df, fecha_maxima)
                             for nombre, df in dict_dataframes.items()}
guardar_df(dict_dataframes, ruta_2)

#####################################################################
##################### awc_f_gfs_rh_mean_b_none_d1_p  ################
#####################################################################

# Observed
awc = pd.read_csv(f'{Path}/awc_o_era5_rh_mean_b_none_d1_p0d.csv')
awc_rell = awc.copy()

# Filtrar Fechas
awc_rell = awc_rell[(awc_rell['date'] >= '2000-01-01') & (awc_rell['date'] <= 'end_date')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)
awc = awc[(awc['date'] >= '2006-10-10') & (awc['date'] <= '2016-12-31')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)

# Filtrar Fechas
# Cargar los archivos csv en una lista de dataframes

dataframes = [pd.read_csv(f'{dir_path}/awc_f_gfs_rh_mean_b_none_d1_p0d.csv'),
              pd.read_csv(f'{dir_path}/awc_f_gfs_rh_mean_b_none_d1_p1d.csv'),
              pd.read_csv(f'{dir_path}/awc_f_gfs_rh_mean_b_none_d1_p2d.csv'),
              pd.read_csv(f'{dir_path}/awc_f_gfs_rh_mean_b_none_d1_p3d.csv'),
              pd.read_csv(f'{dir_path}/awc_f_gfs_rh_mean_b_none_d1_p4d.csv')]

# Eliminar gauge_id
for df in dataframes:
    df.drop('name_id', axis=1, inplace=True)

# Completar fechas ausentes
dataframes = [complete_dates(df, '2000-01-01', end_date) for df in dataframes]

# Asignar los dataframes actualizados a las variables originales
awc_f_gfs_rh_mean_b_none_d1_p0d, awc_f_gfs_rh_mean_b_none_d1_p1d, \
awc_f_gfs_rh_mean_b_none_d1_p2d, awc_f_gfs_rh_mean_b_none_d1_p3d, \
awc_f_gfs_rh_mean_b_none_d1_p4d = dataframes

## Con estos dfs se creara la función para imputar
awc_p0d, awc_p1d, awc_p2d, awc_p3d, awc_p4d = dataframes

# Filtrar las fechas 
def filter_df(df):
    df_filtered = df[(df['date'] >= '2006-10-10') & (df['date'] <= '2016-12-31')].iloc[:, 1:].reset_index(drop=True)
    return df_filtered

awc_p0d = filter_df(awc_p0d)
awc_p1d = filter_df(awc_p1d)
awc_p2d = filter_df(awc_p2d)
awc_p3d = filter_df(awc_p3d)
awc_p4d = filter_df(awc_p4d)

# Función ajuste
def procesar_datos_awc(prefix, df1 = None, df2 = None):
    if prefix in ['p1d', 'p2d', 'p3d','p4d']:
      awc = apply_lag(df1, lag_columns = df1.columns)
      awc_rell = apply_lag(df2, lag_columns = df2.columns)

    seleccion_por_zonas(dataframes, df_names, Tsne)
    # Función Quantil-Quantil 2006-2016
    zonas = ['NG', 'NC', 'ZC', 'ZS', 'ZA']
    awc_conv = {}
    transform_func = {}

    for zona in zonas:
        _, transform_func[zona] = quantile_func(globals()[f'awc_{zona}'], globals()[f'awc_{prefix}_{zona}'], num_quantile=101) # quantile_func(observed_data, forecast_data, num_quantile):

    # Función Quantil-Quantil 2000-2021 y transformación a df (con rename)
    awc_conv_rell = {}
    for zona in zonas:
        awc_conv_rell[zona] = transform_func[zona](globals()[f'awc_rell_{zona}'])

    awc_conv_rell_df = {zona: convert_and_rename(awc_conv_rell[zona], globals()[f'awc_{zona}'].columns) for zona in zonas}

    # Unir df
    awcc = [awc_conv_rell_df[zona] for zona in zonas]
    awc_conv_rellU = pd.concat(awcc, axis=1)
    # LLenar df
    awc_f_gfs_rh_mean_b_none_d1 = llenar_nan(df1=globals()[f'awc_f_gfs_rh_mean_b_none_d1_{prefix}'], df2 = awc_conv_rellU)

    return awc_f_gfs_rh_mean_b_none_d1

# División en zonas geográficas

dataframes = [awc, awc_rell, awc_p0d, awc_p1d, awc_p2d, awc_p3d, awc_p4d]
df_names = ['awc', 'awc_rell','awc_p0d', 'awc_p1d', 'awc_p2d', 'awc_p3d', 'awc_p4d']

##################################################################################################
# Aplicación de la función

awc_f_gfs_rh_mean_b_none_d1_p0d = procesar_datos_awc('p0d')
awc_f_gfs_rh_mean_b_none_d1_p1d = procesar_datos_awc('p1d', df1 = awc, df2 = awc_rell)
awc_f_gfs_rh_mean_b_none_d1_p1d  = imputar(df = awc_f_gfs_rh_mean_b_none_d1_p1d)
awc_f_gfs_rh_mean_b_none_d1_p2d = procesar_datos_awc('p2d', df1 = awc, df2 = awc_rell)
awc_f_gfs_rh_mean_b_none_d1_p2d  = imputar(df = awc_f_gfs_rh_mean_b_none_d1_p2d)
awc_f_gfs_rh_mean_b_none_d1_p3d = procesar_datos_awc('p3d', df1 = awc, df2 = awc_rell)
awc_f_gfs_rh_mean_b_none_d1_p3d  = imputar(df = awc_f_gfs_rh_mean_b_none_d1_p3d)
awc_f_gfs_rh_mean_b_none_d1_p4d = procesar_datos_awc('p4d', df1 = awc, df2 = awc_rell)
awc_f_gfs_rh_mean_b_none_d1_p4d  = imputar(df = awc_f_gfs_rh_mean_b_none_d1_p4d)

#funcion para que los datos esten en el rango

def adjust_values(dataframes):
    adjusted_dataframes = []
    for df in dataframes:
        df_scaled = df.copy()  # Crear una copia para no modificar el DataFrame original
        df_scaled[df_scaled.select_dtypes(include=[np.number]).columns] /= 10  # Escalar todos los valores numéricos
        for column in df_scaled.columns:
            if column != 'date':
                df_scaled[column] = df_scaled[column].clip(lower=0, upper=100)  # Reemplazar valores negativos por 0 y mayores que 100 por 100
        adjusted_dataframes.append(df_scaled)
    return adjusted_dataframes

dataframes = [awc_f_gfs_rh_mean_b_none_d1_p0d, awc_f_gfs_rh_mean_b_none_d1_p1d, awc_f_gfs_rh_mean_b_none_d1_p2d, awc_f_gfs_rh_mean_b_none_d1_p3d, awc_f_gfs_rh_mean_b_none_d1_p4d]
df = adjust_values(dataframes)

dict_dataframes = {
    "awc_f_gfs_rh_mean_b_none_d1_p0d": awc_f_gfs_rh_mean_b_none_d1_p0d,
    "awc_f_gfs_rh_mean_b_none_d1_p1d": awc_f_gfs_rh_mean_b_none_d1_p1d,
    "awc_f_gfs_rh_mean_b_none_d1_p2d": awc_f_gfs_rh_mean_b_none_d1_p2d,
    "awc_f_gfs_rh_mean_b_none_d1_p3d": awc_f_gfs_rh_mean_b_none_d1_p3d,
    "awc_f_gfs_rh_mean_b_none_d1_p4d": awc_f_gfs_rh_mean_b_none_d1_p4d
}

# Filtrar cada DataFrame en el diccionario
dict_dataframes = {nombre: filtrar_hasta_fecha(df, fecha_maxima)
                             for nombre, df in dict_dataframes.items()}
guardar_df(dict_dataframes, ruta_2)

########################################################
###################### DICCIONARIO ######################
#########################################################

class HidroCL_Dictionary:
    def __init__(self):
        self.data = {} # Crea una variable de instancia vacía llamada "data" para almacenar el diccionario

    def readDiccionario(self, filename): # Lee el archivo CSV especificado utilizando Pandas
        df = pd.read_csv(filename, sep=';', skiprows=2, header=None, encoding='latin-1')
        keys = ["Group", "Type", "Source", "Variable", "SpatialAgg", "Coverage", "TemporalAgg", "Period", "ValidTime"] # Define una lista de nombres de columnas
        for i in range(0, len(keys)): # Para cada columna, almacena las claves y valores en el diccionario "data"
            self.data[keys[i]] = {'key': df[2*i + 1], 'value': df[2*i]}
            self.data[keys[i]]['key'] = self.data[keys[i]]['key'].ffill()
            self.data[keys[i]]['value'] = self.data[keys[i]]['value'].ffill()  
            #self.data[keys[i]]['key'] = self.data[keys[i]]['key'].fillna(method='ffill')  # rellenar los valores NaN con el valor de la fila anterior
            #self.data[keys[i]]['value'] = self.data[keys[i]]['value'].fillna(method='ffill')
            self.data[keys[i]] = dict(zip(self.data[keys[i]]["key"], self.data[keys[i]]["value"])) # combina las claves y valores en un solo diccionario

    def __getitem__(self, key):
        return self.data[key]

#################################################################
###################### IMPUTACIÓN DE DATOS ######################
#################################################################

class HidroCL_Complete:
    def __init__(self, varcod, pathdicc = DIC_PATH, varpath = DATA_PATH):
        self.start_date = '2023-12-31'
        self.varpath = varpath
        self.varcod = varcod
        self.pathdicc = pathdicc
        self.data = None
        self.data_list = []
        self.fields = {}
        fields_list = ['Group', 'Type', 'Source', 'Variable', 'SpatialAgg', 'Coverage', 'TemporalAgg', 'Period', 'ValidTime']
        for i, val in enumerate(varcod.split('_')):
            self.fields[fields_list[i]] = val
        self.path = self.__getMetadata__('path')
        self.Latency = self.__getMetadata__('Latency')
        self.Scale_Factor = self.__getMetadata__('Scale_Factor')
        self.Unit = self.__getMetadata__('Unit')
        self.__dicc__ = HidroCL_Dictionary()
        self.__dicc__.readDiccionario(pathdicc)

    # Obtiene los metadatos asociados con la variable
    def __getMetadata__(self, attribute):
      if attribute in self.fields:
        return self.__dicc__[attribute].get(self.fields[attribute])
      else:
        Dic_Var = pd.read_csv(self.varpath, sep = ',')
        if self.varcod in Dic_Var['New_code'].values:
            row = Dic_Var[Dic_Var['New_code'] == self.varcod]
            if attribute == 'New_code':
                return row['New_code'].values[0]
            elif attribute == 'path':
                return row['Relative_path'].values[0]
            elif attribute == 'Latency':
                return row['Latencia'].values[0]
            elif attribute == 'Scale_Factor':
                return row['Scale_Factor'].values[0]
            elif attribute == 'Unit':
                return row['Unit'].values[0]
            else:
                print(f'El atributo {attribute} no existe')
                return None
        else:
          if (self.fields["TemporalAgg"] in ["sum", "mean"] or
             (self.fields["Period"].startswith("d") and int(self.fields["Period"][1:]) > 1) or
             (self.fields["ValidTime"].startswith("p") and int(self.fields["ValidTime"][1:].rstrip("d")) in range(0,5)) or
             (self.fields["ValidTime"].startswith("m") and int(self.fields["ValidTime"][1:].rstrip("d")) in range(1,int(self.fields["ValidTime"][1:].rstrip("d")) + 1)) or
             (self.fields["ValidTime"].isdigit() and int(self.fields["ValidTime"]) > 1)):
             fields_mod = self.fields.copy()
             fields_mod["TemporalAgg"] = "none"
             fields_mod["Period"] = "d1"
             fields_mod["ValidTime"] = "p0d"
             varcod_mod = "_".join(fields_mod.values())
             #print(f"Variable actual: {self.varcod}, variable modificada: {varcod_mod}")
             return HidroCL_Variable(varcod_mod, pathdicc=self.pathdicc, varpath=self.varpath).__getMetadata__(attribute)
          else:
                print(f'La variable {self.varcod} no se encuentra en el archivo')
                return None
    '''
    ### verifica la existencia de la ruta del archivo, lee el archivo y eliminar columnas name_id ###
    start_date = '2023-12-31'
    def __read_data__(self, start_date = start_date):
        if self.path is not None:
            self.path = os.path.join(os.path.dirname(self.varpath), self.path)
            try:
                self.data = pd.read_csv(self.path)

                # Eliminar columna 'Unnamed: 0' si existe (ya que en algunos df está presente)
                if 'Unnamed: 0' in self.data.columns:
                    self.data = self.data.drop(columns=['Unnamed: 0'])

                # Convertir 'date' a datetime y filtrar por fecha si start_date es proporcionado
                if 'date' in self.data.columns:
                    self.data['date'] = pd.to_datetime(self.data['date'])
                    # self.data = self.data.loc[self.data['date'] >= start_date
                    self.data = self.data[self.data['date'] >= pd.to_datetime(start_date)]

                    # Verifica si 'Type' es 'o' o 'f' y 'name_id' está en las columnas
                    if  'name_id' in self.data.columns:
                    # Eliminar columna 'name_id'
                      self.data = self.data.drop(columns=['name_id'])


                    # Reiniciar índice después de filtrar
                    self.data.reset_index(inplace=True, drop=True)

                return self.data
            except FileNotFoundError:
                print(f"No se pudo encontrar el archivo en la ruta {self.path}")
                return None
        else:
            print(f"No se puede leer el archivo debido a que no se encontró la ruta para la variable {self.varcod}")
            return None

     ### COMPLETA LAS FECHAS FALTANTES ###
    '''
       
    def agrega_columnas(self, df2):
       df1 = self.data
       df_completo = df1.combine_first(df2)
       df_completo = df_completo[df2.columns]  # Reordena las columnas
       return df_completo    
     
    def __read_data__(self, start_date= None):
        if start_date is None:
           start_date = self.start_date
        if self.path is not None:
            self.path = os.path.join(os.path.dirname(self.varpath), self.path)
            try:
                self.data = pd.read_csv(self.path)
                self.data['date'] = pd.to_datetime(self.data['date'])
                self.data = self.data.sort_values('date')
                self.data = self.data.reset_index(drop=True)
                # Eliminar columna 'Unnamed: 0' si existe (ya que en algunos df está presente)
                if 'Unnamed: 0' in self.data.columns:
                   self.data = self.data.drop(columns=['Unnamed: 0'])

                 # Convertir 'date' a datetime
                if 'date' in self.data.columns:
                    self.data['date'] = pd.to_datetime(self.data['date'])
                    self.data = self.data.sort_values('date')
                    self.data = self.data.reset_index(drop=True)                     
                   # dates_to_remove = [pd.Timestamp('2024-06-30'), pd.Timestamp('2024-07-01')]
                   # self.data = self.data[~self.data['date'].isin(dates_to_remove)]
                   # Verificar si la fecha de hoy está presente
                    today =  pd.Timestamp.now().normalize()  
                    if  self.fields['Type'] == 'f' and today not in self.data['date'].values:
                        raise ValueError("La fecha de hoy no está presente en los datos.")

                    # Verificar si existe alguna fecha igual o posterior a start_date (esto se hace porque hay variables que tienen fechas cada 5 años)
                    if not (self.data['date'] >= pd.to_datetime(start_date)).any():
                        # Buscar la fecha más cercana inferior a start_date
                        closest_date = self.data[self.data['date'] < pd.to_datetime(start_date)]['date'].max()
                        if pd.isna(closest_date):
                            raise ValueError("No hay fechas disponibles antes de start_date.")
                            return None
                        else:
                            start_date = closest_date
                 
                    # Filtrar por la fecha de inicio ajustada
                    self.data = self.data[self.data['date'] >= pd.to_datetime(start_date)]
                
               # Verifica si 'Type' es 'o' o 'f' y 'name_id' está en las columnas
                    if  'name_id' in self.data.columns:
                        # Eliminar columna 'name_id'
                        self.data = self.data.drop(columns=['name_id'])
                 
                    # Reiniciar índice después de filtrar
                    self.data.reset_index(inplace=True, drop=True)
                    
                            # Verificación adicional para la variable específica
                if self.varcod == 'veg_o_modis_agr_mean_b_none_d1_p0d':
                    # Crear una instancia para veg_o_modis_evi_mean_b_none_d1_p0d
                    evi_instance = HidroCL_Complete('veg_o_modis_evi_mean_b_none_d1_p0d', self.pathdicc, self.varpath)
                    df_evi = evi_instance.__read_data__()
                   
                    # Si df_evi no es None, realizar la agregación
                    if df_evi is not None:
                        self.data = self.agrega_columnas(df2=df_evi)           
                        
                return self.data
            
            except FileNotFoundError:
                print(f"No se pudo encontrar el archivo en la ruta {self.path}")
                return None
        else:
            print(f"No se puede leer el archivo debido a que no se encontró la ruta para la variable {self.varcod}")
    
            return None 

############### FUNCIÓN PARA IMPUTAR GFS EN CASO DE NAN (FILA ANTERIOR) ###################################

    def __rellenar_fila_con_datos__(self, valid_time = None, fila_nan_idx = None, dias_offset = None):
            if valid_time == None or fila_nan_idx == None or dias_offset == None:
              print('Uno de los atributos en __rellenar_fila_con_datos__ es None.')
              pass
            else:
                varcod = '_'.join([val if key != 'ValidTime' else valid_time for key, val in self.fields.items()])
                instance = HidroCL_Complete(varcod, self.pathdicc, self.varpath)
                instance.__read_data__()
                instance.__completeDates__()
                data_instance = instance.data
                data_instance['date'] = pd.to_datetime(data_instance['date'])
                data_instance = data_instance.sort_values('date')
                data_instance.reset_index(inplace=True, drop=True)
                fila_nan_idx = fila_nan_idx-1   
                nan_date = self.data[self.data.isna().any(axis=1)]['date'].iloc[fila_nan_idx]
                print(nan_date)
                target_date = nan_date - pd.DateOffset(days=dias_offset)

                if target_date in data_instance['date'].values:
                    self.data.loc[self.data['date'] == nan_date] = self.data.loc[self.data['date'] == nan_date].fillna(data_instance.loc[data_instance['date'] == target_date].squeeze())
                    self.data.reset_index(inplace=True, drop=True)
                    return self.data
                else:
                  print('Las fechas no estan en el otro df.')

############### COMPLETA LAS FECHAS FALTANTES ##############

    def __completeDates__(self):
        if self.data is None:
            print("No hay datos para verificar.")
            return

        if self.fields['Type'] == 's':
            print(f"Las fechas no necesitan ser ordenadas para el tipo 's' en la variable {self.varcod}.")
            return

        # Convertir la columna de fechas a formato datetime
        self.data['date'] = pd.to_datetime(self.data['date'])

        # Ordenar las fechas si es necesario
        if not self.data['date'].is_monotonic_increasing:
            self.data.sort_values(by='date', inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            print(self.data) 
        # Establecer el rango de fechas desde start_date hasta la fecha actual
        start_date = self.data['date'].min()
        #end_date = self.data['date'].max()
        end_date = pd.Timestamp.now().normalize()  # Fecha actual sin la hora, minuto y segundo

        # Rellenar fechas faltantes
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        missing_dates = pd.Index(date_range).difference(self.data['date'])
        if not missing_dates.empty:
            # Crear un DataFrame para las fechas faltantes con todas las columnas necesarias
            missing_data = pd.DataFrame(index=missing_dates, columns=self.data.columns)
            missing_data['date'] = missing_data.index
            missing_data = missing_data.reset_index(drop=True)

            # Concatenar con el DataFrame original
            self.data = pd.concat([self.data, missing_data], ignore_index=True).sort_values(by='date')
            self.data.reset_index(drop=True, inplace=True)
            print(f"{len(missing_dates)} fechas han sido agregadas con valores NaN en la variable {self.varcod}.")
        else:
            print(f"Las fechas están completas en la variable {self.varcod}.")

       # return self.data
    def __rellenar_fila_con_datos_p4d__(self, valid_time = None, fila_nan_idx = None, dias_offset = None):

            if valid_time == None and fila_nan_idx == None and dias_offset == None:
              #print('Uno de los atributos en __rellenar_fila_con_datos_p4d__ es None.')
              pass
            else:
                varcod = '_'.join([val if key != 'ValidTime' else valid_time for key, val in self.fields.items()])
                instance = HidroCL_Complete(varcod, self.pathdicc, self.varpath)
                instance.__read_data__()
                instance.__completeDates__()
                data_instance = instance.data
                data_instance['date'] = pd.to_datetime(data_instance['date'])
                fila_nan_idx = fila_nan_idx-1
                nan_date = self.data[self.data.isna().any(axis=1)]['date'].iloc[fila_nan_idx]
                target_date = nan_date + pd.DateOffset(days=dias_offset)

                if target_date in data_instance['date'].values:
                    self.data.loc[self.data['date'] == nan_date] = self.data.loc[self.data['date'] == nan_date].fillna(data_instance.loc[data_instance['date'] == target_date].squeeze())
                    self.data.reset_index(inplace=True, drop=True)
            #        return self.data

    ### IMPUTA LOS VALORES FALTANTES ###
    def __fill_missing_values__(self):

     # Verifica si hay datos cargados en el DataFrame
       if self.data is not None:
           self.data['date'] = pd.to_datetime(self.data['date'])
           n_added = 0  # Contador para el número total de valores NaN agregados

           # Itera sobre cada columna, excepto 'date'
           for col in self.data.columns:
               if col != "date":
                  # Convierte la columna a numérico y agrega NaNs donde hay errores
                   self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                   # Suma la cantidad de NaNs recién agregados al contador
                   n_added += self.data[col].isna().sum()

                   # En 'pp' reemplaza los valores negativos por 0 (a veces habian valores negativos en pp)
                   if self.fields['Group'] == 'pp':
                      negative_values_count = (self.data[col] < 0).sum()
                      self.data[col] = self.data[col].apply(lambda x: 0 if x < 0 else x)
                      n_added += negative_values_count
           
           # Si el tipo de datos es 'f' (pronosticado), retorna el DataFrame sin cambios (Aquí se debe hacer la imputación propuesta para gfs)
           if self.fields['Type'] == 'f':      
                  # Verifica secuencias de días consecutivos con NaNs
                  indices_con_nan = self.data.index[self.data.isna().any(axis=1)]
                  diferencias = indices_con_nan.to_series().diff() - 1 #se utiliza para identificar secuencias de valores NaN consecutivos

                  contador_consecutivos = 0
                  for diferencia in diferencias:
                      if diferencia == 0:
                          contador_consecutivos += 1
                      else:
                          if contador_consecutivos >= 4:
                              raise ValueError("Existe más de 4 días consecutivos con datos faltantes")
                          contador_consecutivos = 0
                  # Verificar la última secuencia
                  if contador_consecutivos >= 4:
                      raise ValueError("Existe más de 4 días consecutivos con datos faltantes")
                  
                  else:
                      if not self.data.isna().any().any():
                        return self.data
                      else:
                        # self.data = self.data.tail(6)
                      ###### p0d #####
                         if self.fields["ValidTime"] == 'p0d': # and self.data.isna().any().any():
                                      num_nan_rows = self.data.isna().any(axis=1).sum()
                                      if num_nan_rows in [1, 2, 3, 4]:
                                          for i in range(num_nan_rows):
                                              self.data = self.__rellenar_fila_con_datos__(f'p{i+1}d', i, i+1)

                                          return self.data

                      ###### p1d #####
                         elif self.fields["ValidTime"] == 'p1d' :
                                      num_nan_rows = self.data.isna().any(axis=1).sum()
                                      if num_nan_rows in [1, 2, 3]:
                                          for i in range(num_nan_rows):
                                              self.__rellenar_fila_con_datos__(f'p{i+2}d', i, i+1)
                                          return self.data
                                      elif num_nan_rows == 4:
                                          for i in range(3):
                                              self.__rellenar_fila_con_datos__(f'p{i+2}d', i, i+1)
                                          # Procesar la cuarta fila utilizando 'p0d' del día siguiente
                                          self.__rellenar_fila_con_datos_p4d__('p0d', 3, 1)
                                          return self.data

                      ###### p2d #####
                         elif self.fields["ValidTime"] == 'p2d':
                              num_nan_rows = self.data.isna().any(axis=1).sum()

                              if num_nan_rows in [1, 2]:
                                  for i in range(num_nan_rows):
                                      self.__rellenar_fila_con_datos__(f'p{i+3}d', i, i+1)
                                  return self.data
                              elif num_nan_rows in [3, 4]:
                                  for i in range(2):
                                      self.__rellenar_fila_con_datos__(f'p{i+3}d', i, i+1)
                                  # Rellenar la tercera fila con 'p1d' del día siguiente
                                  if num_nan_rows == 3:
                                    self.__rellenar_fila_con_datos_p4d__('p1d', 2, 1)
                                  else:
                                    self.__rellenar_fila_con_datos_p4d__('p0d', 2, 2)
                                  # Si hay una cuarta fila, rellenarla con 'p1d' del día siguiente
                                  if num_nan_rows == 4:
                                      self.__rellenar_fila_con_datos_p4d__('p1d', 3, 1)
                                  return self.data

                      ###### p3d #####
                         elif self.fields["ValidTime"] == 'p3d':
                              num_nan_rows = self.data.isna().any(axis=1).sum()

                              # Rellenar la primera fila siempre con 'p4d'
                              if num_nan_rows >= 1:
                                  self.__rellenar_fila_con_datos__('p4d', 0, 1)

                              # Valid times y offsets de días ajustados para las filas restantes
                              elif num_nan_rows == 2:
                                  self.__rellenar_fila_con_datos_p4d__('p2d', 1, 1)
                              elif num_nan_rows == 3:
                                  self.__rellenar_fila_con_datos_p4d__('p1d', 1, 2)
                                  self.__rellenar_fila_con_datos_p4d__('p2d', 2, 1)
                              elif num_nan_rows == 4:
                                  self.__rellenar_fila_con_datos_p4d__('p0d', 1, 3)
                                  self.__rellenar_fila_con_datos_p4d__('p1d', 2, 2)
                                  self.__rellenar_fila_con_datos_p4d__('p2d', 3, 1)
                              return self.data

                    ###### p4d #####
                         elif self.fields["ValidTime"] == 'p4d':
                              num_nan_rows = self.data.isna().any(axis=1).sum()

                              # Valid times y offsets de días ajustados para cada caso
                              if num_nan_rows == 1:
                                  self.__rellenar_fila_con_datos_p4d__('p3d', 0, 1)
                              elif num_nan_rows == 2:
                                  self.__rellenar_fila_con_datos_p4d__('p2d', 0, 2)
                                  self.__rellenar_fila_con_datos_p4d__('p3d', 1, 1)
                              elif num_nan_rows == 3:
                                  self.__rellenar_fila_con_datos_p4d__('p1d', 0, 3)
                                  self.__rellenar_fila_con_datos_p4d__('p2d', 1, 2)
                                  self.__rellenar_fila_con_datos_p4d__('p3d', 2, 1)
                              elif num_nan_rows == 4:
                                  self.__rellenar_fila_con_datos_p4d__('p0d', 0, 4)
                                  self.__rellenar_fila_con_datos_p4d__('p1d', 1, 3)
                                  self.__rellenar_fila_con_datos_p4d__('p2d', 2, 2)
                                  self.__rellenar_fila_con_datos_p4d__('p3d', 3, 1)

                              return self.data
              
           else:

                   # Rellena los valores NaN dependiendo del tipo de datos y grupo
                        if self.fields['Type'] == 'o':
                              if self.fields['Group'] != 'pp':
                                  self.data.ffill( inplace=True)
                              else:
                                  self.data.fillna(0, inplace=True)

                              return self.data

                        # Imprime el número total de valores NaN agregados o reemplazados
                        print(f"Se agregaron {n_added} valores")
       else:
               # Si no hay datos cargados, imprime un mensaje de error y retorna None
               print(f"No se puede leer el archivo debido a que no se encontraron datos para la variable {self.varcod}")
               return None

#################################################################################################################################

    #### REEMPLAZA LAS PRIMERAS FECHAS NAN CON LOS DATOS DE LA MISMA FECHA PERO DEL SIGUIENTE AÑO ####
    def __impute_next_year__(self):
        df_copy = self.data.copy()
        for column in df_copy.columns.drop(['date']):
            nan_indices = df_copy[df_copy[column].isna()].index
            for nan_index in nan_indices:
                nan_date = df_copy.loc[nan_index, 'date']
                next_year_date = nan_date + pd.DateOffset(years=1)
                if next_year_date in df_copy['date'].values:
                    next_year_value = df_copy.loc[df_copy['date'] == next_year_date, column].values[0]
                    df_copy.loc[nan_index, column] = next_year_value
        return df_copy
   
    '''
    ##########################################################
    ########## BORRA LAS CARPETAS DE DFs_ACTUALIZADOS ########
    ##########################################################

    def __borrar_subfolders__(self, path_save_base):
        subfolders = ['observed', 'forecasted', 'other']  
        for subfolder in subfolders:
            folder_path = os.path.join(path_save_base, subfolder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(f"La carpeta {folder_path} ha sido borrada.")
            else:
                print(f"La carpeta {folder_path} no existe y no necesita ser borrada.")
    '''

    ############################################
    ### Hace la imputación a toda las listas ###
    ############################################

    def get_imputation(self, varcodes_list, path_save_base, path_read):
       # self.__borrar_subfolders__(path_save_base)
        processed_dataframes = {}
        for varcod in varcodes_list:
            instance = HidroCL_Complete(varcod, self.pathdicc, self.varpath)
            instance.__read_data__()
            instance.__completeDates__()
            new_df = instance.__fill_missing_values__()

            if new_df is not None and new_df.isna().any().any():
                new_df = instance.__impute_next_year__()

            if new_df is not None:
                new_df['date'] = pd.to_datetime(new_df['date'])
                new_df.set_index('date', inplace=True)

                subfolder = {'o': 'observed', 'f': 'forecasted', 's': 'static'}.get(instance.fields['Type'], 'other')
                read_path = os.path.join(path_read, subfolder, f"{varcod}.csv")  # Ruta para leer el archivo existente.
                save_path = os.path.join(path_save_base, subfolder, f"{varcod}.csv")  # Ruta para guardar el archivo actualizado.
                os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Crear las carpetas necesarias.

                if os.path.exists(read_path):
                    existing_df = pd.read_csv(read_path)
                    existing_df['date'] = pd.to_datetime(existing_df['date'])
                    existing_df.set_index('date', inplace=True)
                    if existing_df.index.duplicated().any():
                        raise Exception(f"El DataFrame existente tiene índices duplicados para la variable {varcod}")
                    if new_df.index.duplicated().any():
                        raise Exception(f"El DataFrame nuevo tiene índices duplicados para la variable {varcod}")
                   
                   # Actualizar los valores existentes
                    existing_df.update(new_df)

                    # Concatenar filas nuevas
                    combined_df = pd.concat([existing_df, new_df[~new_df.index.isin(existing_df.index)]])

                    # Restablecer el índice y guardar el DataFrame
                    combined_df.reset_index(inplace=True)
                    combined_df.to_csv(save_path, index=False)
                    print(f"Archivo actualizado y guardado en: {save_path}")

                    processed_dataframes[varcod] = combined_df
                else:
                    raise FileNotFoundError(f'Error: La variable a actualizar no se encuentra en la carpeta para {varcod}.')

            else:
                raise Exception(f"No se pudo procesar el archivo para la variable {varcod}")

        #return processed_dataframes


####################################################
###### Crear la instancia de HidroCL_Complete ######
####################################################

variable = HidroCL_Complete('tmp_f_gfs_tmp_min_b_none_d1_p0d')

list_observed  = ['pp_o_era5_pp_mean_b_none_d1_p0d', 'atm_o_era5_uw_mean_b_none_d1_p0d', 'atm_o_era5_vw_mean_b_none_d1_p0d', 'tmp_o_era5_tmp_mean_b_none_d1_p0d', 'tmp_o_era5_tmin_mean_b_none_d1_p0d', 'tmp_o_era5_tmax_mean_b_none_d1_p0d',
                 'lulc_o_modis_dnf_sum_b_none_d1_p0d', 'awc_o_era5_rh_mean_b_none_d1_p0d', 'lulc_o_modis_crp_sum_b_none_d1_p0d', 'lulc_o_modis_csh_sum_b_none_d1_p0d', 'lulc_o_modis_cvm_sum_b_none_d1_p0d', 'lulc_o_modis_dbf_sum_b_none_d1_p0d',
                 'lulc_o_modis_pwt_sum_b_none_d1_p0d', 'lulc_o_modis_urb_sum_b_none_d1_p0d', 'lulc_o_modis_wat_sum_b_none_d1_p0d', 'pp_o_imerg_pp_mean_b_none_d1_p0d', 'pp_o_pdir_pp_mean_b_none_d1_p0d', 'lulc_o_modis_snw_sum_b_none_d1_p0d',
                 'lulc_o_modis_mxf_sum_b_none_d1_p0d', 'lulc_o_modis_ebf_sum_b_none_d1_p0d', 'lulc_o_modis_enf_sum_b_none_d1_p0d', 'lulc_o_modis_osh_sum_b_none_d1_p0d', 'lulc_o_modis_wsv_sum_b_none_d1_p0d', 'snw_o_era5_sca_mean_b_none_d1_p0d',
                 'snw_o_era5_snd_mean_b_none_d1_p0d', 'veg_o_modis_lai_mean_b_none_d1_p0d', 'veg_o_modis_evi_mean_b_none_d1_p0d', 'lulc_o_modis_svn_sum_b_none_d1_p0d', 'veg_o_modis_fpar_mean_b_none_d1_p0d', 'lulc_o_modis_brn_sum_b_none_d1_p0d',
                 'lulc_o_modis_grs_sum_b_none_d1_p0d', 'et_o_era5_eta_mean_b_none_d1_p0d', 'veg_o_modis_ndvi_mean_b_none_d1_p0d', 'veg_o_modis_nbr_mean_b_none_d1_p0d', 'veg_o_modis_agr_mean_b_none_d1_p0d', 'et_o_era5_eto_mean_b_none_d1_p0d',
                 'swc_o_era5_sm_mean_b_none_d1_p0d', 'atm_o_era5_pres_mean_b_none_d1_p0d', 'atm_o_era5_z_mean_b_none_d1_p0d', 'snw_o_era5_sna_mean_b_none_d1_p0d', 'hi_o_gww_rs_tot_b_none_d1_p0d', 'snw_o_modis_sca_tot_n_none_d1_p0d',
                 'snw_o_modis_sca_tot_s_none_d1_p0d', 'pp_o_era5_maxpp_mean_b_none_d1_p0d', 'tmp_o_era5_dew_mean_b_none_d1_p0d', 'lulc_o_modis_mxf_frac_b_none_d1_p0d', 'lulc_o_modis_osh_frac_b_none_d1_p0d', 'lulc_o_modis_dbf_frac_b_none_d1_p0d',
                 'lulc_o_modis_dnf_frac_b_none_d1_p0d', 'lulc_o_modis_urb_frac_b_none_d1_p0d', 'lulc_o_modis_crp_frac_b_none_d1_p0d', 'lulc_o_modis_snw_frac_b_none_d1_p0d', 'lulc_o_modis_wsv_frac_b_none_d1_p0d', 'lulc_o_modis_pwt_frac_b_none_d1_p0d',
                 'lulc_o_modis_ebf_frac_b_none_d1_p0d', 'lulc_o_modis_enf_frac_b_none_d1_p0d', 'lulc_o_modis_wat_frac_b_none_d1_p0d', 'lulc_o_modis_svn_frac_b_none_d1_p0d', 'lulc_o_modis_cvm_frac_b_none_d1_p0d', 'lulc_o_modis_csh_frac_b_none_d1_p0d',
                 'lulc_o_modis_grs_frac_b_none_d1_p0d', 'lulc_o_modis_brn_frac_b_none_d1_p0d', 'snw_o_era5_snr_mean_b_none_d1_p0d', 'pp_o_era5_plen_mean_b_none_d1_p0d']


'''
list_forecasted = ['tmp_f_gfs_tmp_mean_b_none_d1_p0d', 'tmp_f_gfs_tmp_mean_b_none_d1_p1d', 'tmp_f_gfs_tmp_mean_b_none_d1_p2d', 'tmp_f_gfs_tmp_mean_b_none_d1_p3d', 'tmp_f_gfs_tmp_mean_b_none_d1_p4d',
                   'tmp_f_gfs_tmp_max_b_none_d1_p0d',  'tmp_f_gfs_tmp_max_b_none_d1_p1d', 'tmp_f_gfs_tmp_max_b_none_d1_p2d', 'tmp_f_gfs_tmp_max_b_none_d1_p3d', 'tmp_f_gfs_tmp_max_b_none_d1_p4d',
                   'tmp_f_gfs_tmp_min_b_none_d1_p0d',  'tmp_f_gfs_tmp_min_b_none_d1_p1d', 'tmp_f_gfs_tmp_min_b_none_d1_p2d', 'tmp_f_gfs_tmp_min_b_none_d1_p3d', 'tmp_f_gfs_tmp_min_b_none_d1_p4d',
                   'atm_f_gfs_uw_mean_b_none_d1_p0d', 'atm_f_gfs_uw_mean_b_none_d1_p1d', 'atm_f_gfs_uw_mean_b_none_d1_p2d', 'atm_f_gfs_uw_mean_b_none_d1_p3d', 'atm_f_gfs_uw_mean_b_none_d1_p4d',
                   'atm_f_gfs_vw_mean_b_none_d1_p0d', 'atm_f_gfs_vw_mean_b_none_d1_p1d', 'atm_f_gfs_vw_mean_b_none_d1_p2d', 'atm_f_gfs_vw_mean_b_none_d1_p3d', 'atm_f_gfs_vw_mean_b_none_d1_p4d',
                   'atm_f_gfs_gh_mean_b_none_d1_p0d', 'atm_f_gfs_gh_mean_b_none_d1_p1d', 'atm_f_gfs_gh_mean_b_none_d1_p2d', 'atm_f_gfs_gh_mean_b_none_d1_p3d', 'atm_f_gfs_gh_mean_b_none_d1_p4d',
                   'awc_f_gfs_rh_mean_b_none_d1_p0d', 'awc_f_gfs_rh_mean_b_none_d1_p1d', 'awc_f_gfs_rh_mean_b_none_d1_p2d', 'awc_f_gfs_rh_mean_b_none_d1_p3d', 'awc_f_gfs_rh_mean_b_none_d1_p4d',
                   'pp_f_gfs_pp_mean_b_none_d1_p0d',  'pp_f_gfs_pp_mean_b_none_d1_p1d', 'pp_f_gfs_pp_mean_b_none_d1_p2d', 'pp_f_gfs_pp_mean_b_none_d1_p3d', 'pp_f_gfs_pp_mean_b_none_d1_p4d',
                   'pp_f_gfs_pp_max_b_none_d1_p0d', 'pp_f_gfs_pp_max_b_none_d1_p1d', 'pp_f_gfs_pp_max_b_none_d1_p2d', 'pp_f_gfs_pp_max_b_none_d1_p3d', 'pp_f_gfs_pp_max_b_none_d1_p4d',
                   'pp_f_gfs_plen_mean_b_none_d1_p0d', 'pp_f_gfs_plen_mean_b_none_d1_p1d', 'pp_f_gfs_plen_mean_b_none_d1_p2d', 'pp_f_gfs_plen_mean_b_none_d1_p3d', 'pp_f_gfs_plen_mean_b_none_d1_p4d']

'''

###### APLICACIÓN DE LA CLASE PARA IMPUTACIÓN #######

variable.get_imputation(varcodes_list = list_observed, path_save_base = UPDATED_PATH, path_read = IMPUTED_PATH)
#variable.get_imputation(varcodes_list = list_forecasted, path_save_base = UPDATED_PATH, path_read = IMPUTED_PATH)
print('DFs  ACTUALIZADO:')

#print(list_forecasted[0]) 
# Registra el tiempo de finalización
final = time.time()

# Calcula y muestra la duración de la ejecución
duracion_en_minutos = (final - inicio)/60
print(f"El código se ejecutó en {duracion_en_minutos:.2f} minutos")
