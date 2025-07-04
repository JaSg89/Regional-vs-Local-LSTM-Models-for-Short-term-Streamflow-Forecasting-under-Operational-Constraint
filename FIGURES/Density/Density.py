
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy import interpolate
import math


# Cargar los archivos csv en una lista de dataframes

dataframes = [pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/forecasted/tmp_f_gfs_tmp_mean_b_none_d1_p0d.csv'),
              pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/forecasted/tmp_f_gfs_tmp_mean_b_none_d1_p1d.csv'),
              pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/forecasted/tmp_f_gfs_tmp_mean_b_none_d1_p2d.csv'),
              pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/forecasted/tmp_f_gfs_tmp_mean_b_none_d1_p3d.csv'),
              pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/forecasted/tmp_f_gfs_tmp_mean_b_none_d1_p4d.csv')]

# Eliminar name_id
for df in dataframes:
    df.drop('name_id', axis=1, inplace=True)

# Completar fechas ausentes
dataframes = [complete_dates(df, '2000-01-01', '2021-12-31') for df in dataframes]

# Asignar los dataframes actualizados a las variables originales
tmp_f_gfs_tmp_mean_b_none_d1_p0d, tmp_f_gfs_tmp_mean_b_none_d1_p1d, \
tmp_f_gfs_tmp_mean_b_none_d1_p2d, tmp_f_gfs_tmp_mean_b_none_d1_p3d, \
tmp_f_gfs_tmp_mean_b_none_d1_p4d = dataframes

# Observed

tmp_era5 = pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/observed/tmp_o_era5_tmp_mean_b_none_d1_p0d.csv')
tmp_era5_rell = pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/observed/tmp_o_era5_tmp_mean_b_none_d1_p0d.csv')

#Forecasted

tmp_p0d = pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/forecasted/tmp_f_gfs_tmp_mean_b_none_d1_p0d.csv')
tmp_p1d = pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/forecasted/tmp_f_gfs_tmp_mean_b_none_d1_p1d.csv')
tmp_p2d = pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/forecasted/tmp_f_gfs_tmp_mean_b_none_d1_p2d.csv')
tmp_p3d = pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/forecasted/tmp_f_gfs_tmp_mean_b_none_d1_p3d.csv')
tmp_p4d = pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/forecasted/tmp_f_gfs_tmp_mean_b_none_d1_p4d.csv')


# Completar fechas ausentes

dataframes = [tmp_era5, tmp_era5_rell, tmp_p0d, tmp_p1d, tmp_p2d, tmp_p3d, tmp_p4d]

for i, df in enumerate(dataframes):
    dataframes[i] = complete_dates(df, '2000-01-01', '2021-12-31')

tmp_era5, tmp_era5_rell, tmp_p0d, tmp_p1d, tmp_p2d, tmp_p3d, tmp_p4d = dataframes

# Filtrar las fechas 2006-2016

def filter_df(df):
    df_filtered = df[(df['date'] >= '2006-10-10') & (df['date'] <= '2021-12-31')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)
    return df_filtered

tmp_era5 = filter_df(tmp_era5)
tmp_p0d = filter_df(tmp_p0d)
tmp_p1d = filter_df(tmp_p1d)
tmp_p2d = filter_df(tmp_p2d)
tmp_p3d = filter_df(tmp_p3d)
tmp_p4d = filter_df(tmp_p4d)

def filter_df(df):
    df_filtered = df[(df['date'] >= '2000-01-01') & (df['date'] <= '2016-12-31')].drop(columns=['name_id']).iloc[:, 1:].reset_index(drop=True)
    return df_filtered

tmp_era5_rell = filter_df(tmp_era5_rell)

#Funcion de transformación

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

#Se convierten a df y rename
def convert_and_rename(data, column_names):
    df = pd.DataFrame(data)
    df.rename(columns=dict(zip(df.columns, column_names)), inplace=True)
    return df

def llenar_nan(df1, df2):

    # Identificar las columnas comunes entre df1 y df2
    columnas_comunes = df1.columns.intersection(df2.columns)

    # Llenar los NaN en las columnas comunes de df1 con los valores correspondientes en df2
    for columna in columnas_comunes:
        df1[columna].fillna(df2[columna], inplace=True)

    return df1
def complete_dates(df, start_date, end_date):
    try:
        # Crea un rango de fechas entre start_date y end_date
        date_range = pd.date_range(start=start_date, end=end_date)
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

def guardar_df(dict_dataframes, ruta_drive):
    for nombre, df in dict_dataframes.items():
        ruta_archivo = f"{ruta_drive}/{nombre}.csv"
        df.to_csv(ruta_archivo, index=False)
        print(f"Archivo guardado en: {ruta_archivo}")   
 # Lectura del archivo tsne.csv
Tsne = pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/tsne.csv', sep=';')

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

def procesar_datos_tmp(prefix):
    # Función Quantil-Quantil 2006-2016
    zonas = ['NG', 'NC', 'ZC', 'ZS', 'ZA']
    tmp_era5_conv = {}
    transform_func = {}

    for zona in zonas:
        _, transform_func[zona] = quantile_func(globals()[f'tmp_era5_{zona}'], globals()[f'tmp_{prefix}_{zona}'], num_quantile=101)

    # Función Quantil-Quantil 2000-2021 y transformación a df (con rename)
    tmp_era5_conv_rell = {}
    for zona in zonas:
        tmp_era5_conv_rell[zona] = transform_func[zona](globals()[f'tmp_era5_rell_{zona}'])

    tmp_era5_conv_rell_df = {zona: convert_and_rename(tmp_era5_conv_rell[zona], globals()[f'tmp_era5_{zona}'].columns) for zona in zonas}

    # Unir df
    tmp = [tmp_era5_conv_rell_df[zona] for zona in zonas]
    tmp_era5_conv_rellU = pd.concat(tmp, axis=1)
    # LLenar df
    tmp_f_gfs_tmp_mean_b_none_d1 = llenar_nan(df1=globals()[f'tmp_f_gfs_tmp_mean_b_none_d1_{prefix}'], df2=tmp_era5_conv_rellU)

    return tmp_f_gfs_tmp_mean_b_none_d1


# Para posterior comparación

tmp0 = pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/forecasted/tmp_f_gfs_tmp_mean_b_none_d1_p0d.csv')
tmp1 = pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/forecasted/tmp_f_gfs_tmp_mean_b_none_d1_p1d.csv')
tmp2 = pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/forecasted/tmp_f_gfs_tmp_mean_b_none_d1_p2d.csv')
tmp3 = pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/forecasted/tmp_f_gfs_tmp_mean_b_none_d1_p3d.csv')
tmp4 = pd.read_csv('/content/drive/MyDrive/FONDEF/DATASET/Data/forecasted/tmp_f_gfs_tmp_mean_b_none_d1_p4d.csv')

# Completar fechas ausentes

dataframes = [tmp0, tmp1, tmp2, tmp3, tmp4]

for i, df in enumerate(dataframes):
    dataframes[i] = complete_dates(df, '2000-01-01', '2021-12-31')

tmp0, tmp1, tmp2, tmp3, tmp4 = dataframes

# División en zonas geográficas

dataframes = [tmp_era5, tmp_era5_rell, tmp_p0d, tmp_p1d, tmp_p2d, tmp_p3d, tmp_p4d]
df_names = ['tmp_era5', 'tmp_era5_rell','tmp_p0d', 'tmp_p1d', 'tmp_p2d', 'tmp_p3d', 'tmp_p4d']
seleccion_por_zonas(dataframes, df_names, Tsne)


tmp_f_gfs_tmp_mean_b_none_d1_p0d = procesar_datos_tmp('p0d')
tmp_f_gfs_tmp_mean_b_none_d1_p1d = procesar_datos_tmp('p1d')
tmp_f_gfs_tmp_mean_b_none_d1_p2d = procesar_datos_tmp('p2d')
tmp_f_gfs_tmp_mean_b_none_d1_p3d = procesar_datos_tmp('p3d')
tmp_f_gfs_tmp_mean_b_none_d1_p4d = procesar_datos_tmp('p4d')


def compare_densities(Forecasted_data, Observed_data_transformed, Variable, feature_col, plot_together=True):
    if plot_together:
        plt.figure(figsize=(18, 8))
        # Crea un gráfico de densidad para ambos conjuntos de datos
        sns.kdeplot(Forecasted_data[feature_col], fill=True, label="Forecasted data", color='blue')
        sns.kdeplot(Observed_data_transformed[feature_col], fill=True, label="Observed data\n transformed", color='orange')

        # Agrega un título y etiquetas de eje al gráfico
       # plt.title('Comparison of Densities: Forecasted vs Observed Transformed Data', fontsize=30)
        plt.xlabel(Variable, fontsize=33)
        plt.ylabel('Density', fontsize=33)

        # Agrega una leyenda que indique a qué conjunto de datos pertenece cada densidad
        plt.legend(fontsize=20, title="Data Type", title_fontsize=25)

        # Ajustar el tamaño de los ticks en los ejes
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.savefig(f'/content/Rio_Larqui_En_Santa_Cruz_De_Cuca_join.png', dpi=600, bbox_inches='tight', facecolor='white')
        plt.grid(alpha=0.3)
        plt.show()

    else:
        # Crea dos gráficos de densidad, uno para cada conjunto de datos
        fig, axes = plt.subplots(ncols=2, figsize=(19, 6))

        # Encuentra el rango mínimo y máximo en el eje x para ambos conjuntos de datos
        min_x = min(Forecasted_data[feature_col].min(), Observed_data_transformed[feature_col].min())
        max_x = max(Forecasted_data[feature_col].max(), Observed_data_transformed[feature_col].max())

        sns.kdeplot(Forecasted_data[feature_col], fill=True, label="Forecasted data", ax=axes[0], color='blue')
        sns.kdeplot(Observed_data_transformed[feature_col], fill=True, label="Observed data\ntransformed", ax=axes[1], color='orange')

        # Establece el mismo rango en el eje x para ambos gráficos
        axes[0].set_xlim(min_x, max_x)
        axes[1].set_xlim(min_x, max_x)

        # Agrega un título y etiquetas de eje a cada gráfico
        axes[0].set_title('a)', fontsize=32)
        axes[0].set_xlabel(Variable, fontsize=32)
        axes[0].set_ylabel('Density', fontsize=32)

        axes[1].set_title('b)', fontsize=32)
        #axes[1].set_xlabel(Variable, fontsize=32)
        axes[1].set_ylabel(' ', fontsize=32)

        # Ajustar el tamaño de los ticks en los ejes
        axes[0].tick_params(axis='both', which='major', labelsize=30)
        axes[1].tick_params(axis='both', which='major', labelsize=30)

        # Ajustar las leyendas
        axes[0].legend(fontsize=22, loc='upper right')
        axes[1].legend(fontsize=22, loc='upper right')
        plt.grid()

        # Guardar la figura
        plt.tight_layout()
        plt.savefig(f'/content/Rio_Larqui_En_Santa_Cruz_De_Cuca.png', dpi=1000, bbox_inches='tight', facecolor='white')

        # Muestra los gráficos
        plt.show()

compare_densities(Forecasted_data =  tmp0.iloc[:,2:]/10,
                   Observed_data_transformed = tmp_f_gfs_tmp_mean_b_none_d1_p0d.iloc[:,1:]/10,
                     Variable  = 'Average temperature', feature_col = '8134003', plot_together=True)