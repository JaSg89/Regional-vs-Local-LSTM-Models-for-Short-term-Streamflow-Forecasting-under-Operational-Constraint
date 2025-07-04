# --------------------------------------------------------------------------- #
# Necessary Libraries
# --------------------------------------------------------------------------- #
#pip install geopandas
#pip install rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import rasterio
import rasterio.plot
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
KGE_THRESHOLD = 0.595
MASK_VALUE = 2
DRIVE_PATH = '/content/drive/MyDrive/FONDEF/Variables'

# --------------------------------------------------------------------------- #
# PARTE 1: CÁLCULO DE DATOS (TU SCRIPT ORIGINAL)
# --------------------------------------------------------------------------- #

#### #### #### #### #### #### #### ####
#### CAUDAL MEDIO LOOOOOCAAAALLLL  ####
#### #### #### #### #### #### #### ####
print("Iniciando carga de datos locales...")
dir_path = '/content/drive/MyDrive/FONDEF/Variables/ENTRENAMIENTO_LOCAL/Resultados_train_full'
ruta0 = f'{dir_path}/Model_Local_p0d.csv'
df0_loc =  pd.read_csv(ruta0,float_precision='legacy')
cols_to_multiply = ['Qsim', 'Qobs', 'mean', 'std_dev', 'median', 'per_5', 'per_25', 'per_75', 'per_95']
df0_loc[cols_to_multiply] = df0_loc[cols_to_multiply] * 10
print("Datos locales cargados y multiplicados.")

#### #### #### #### #### #### #### ####
     ########## GLOBAL ###########
#### #### #### #### #### #### #### ####
print("Iniciando carga de datos globales...")
df0_glob = pd.read_csv('/content/drive/MyDrive/FONDEF/Variables/LSTM3/Ult_Model_LSTM/Qmean/Qmean3/Qmean_LSTM_p0d.csv')
df0_glob[cols_to_multiply] = df0_glob[cols_to_multiply] * 10
print("Datos globales cargados y multiplicados.")

############# PASO A M3/S ###############
print("Convirtiendo a m3/s...")
data_area = pd.read_csv('/content/drive/MyDrive/FONDEF/Variables/DF_LSTM/LSTM11.csv', usecols=['gauge_id', 'date',  'top_s_cam_area_tot_b_none_c_c'])
data_area.rename(columns={'gauge_id': 'ID'}, inplace=True)

df0_glob = pd.merge(df0_glob, data_area, on=['ID', 'date'], how='inner')
df0_loc = pd.merge(df0_loc, data_area, on=['ID', 'date'], how='inner')

df0_glob['Qsim'] = (df0_glob['top_s_cam_area_tot_b_none_c_c'] * df0_glob['Qsim']) / 864
df0_loc['Qsim'] = (df0_loc['top_s_cam_area_tot_b_none_c_c'] * df0_loc['Qsim']) / 864
df0_glob['Qobs'] = (df0_glob['top_s_cam_area_tot_b_none_c_c'] * df0_glob['Qobs']) / 864
df0_loc['Qobs'] = (df0_loc['top_s_cam_area_tot_b_none_c_c'] * df0_loc['Qobs']) / 864
print("Conversión a m3/s completada.")

####################################
########## CALCULO DE KGE ##########
####################################
print("Calculando KGE...")
def KGE_metric(sim, obs): 
   
    if obs.std() == 0 or np.isnan(obs.std()) or obs.mean() == 0 or np.isnan(obs.mean()):
        return np.nan # O algún otro valor que indique invalidez
    alpha = sim.std()/obs.std()
    beta = sim.mean()/obs.mean()
    # Asegurarse de que sim y obs tengan la misma longitud y no contengan NaNs que afecten np.corrcoef
    valid_indices = ~np.isnan(sim) & ~np.isnan(obs)
    if np.sum(valid_indices) < 2: 
        return np.nan
    sim_valid = sim[valid_indices]
    obs_valid = obs[valid_indices]
    if len(sim_valid) < 2 or len(obs_valid) < 2: 
        return np.nan
        
    corr_coeff_matrix = np.corrcoef(sim_valid, obs_valid)
    if corr_coeff_matrix.shape != (2,2) or np.isnan(corr_coeff_matrix[0,1]):
      return np.nan # No se pudo calcular la correlación
    corr_coeff = corr_coeff_matrix[0,1]
    
    kge_val = 1 - np.sqrt((corr_coeff-1)**2 + (alpha-1)**2 + (beta-1)**2 )
    return kge_val

def compute_KGE_for_each_ID(df):
    kge_results = df.groupby('ID').apply(lambda g: KGE_metric(g['Qsim'], g['Qobs']))
    kge_df = pd.DataFrame(kge_results, columns=['KGE']).reset_index()
    return kge_df

# LOCAL
kge_df0_loc = compute_KGE_for_each_ID(df0_loc[df0_loc['mask']==MASK_VALUE])
lista_df0 = kge_df0_loc[kge_df0_loc['KGE']>=KGE_THRESHOLD].ID.tolist()
df0_loc_06 = df0_loc[df0_loc['ID'].isin(lista_df0)]
print(f'df0_loc KGE >= {KGE_THRESHOLD}: {df0_loc_06.ID.unique().shape[0]} estaciones')

# GLOBAL
kge_df0_glob = compute_KGE_for_each_ID(df0_glob[df0_glob['mask']==MASK_VALUE])
lista_df0_glob = kge_df0_glob[kge_df0_glob['KGE']>=KGE_THRESHOLD].ID.tolist()
df0_glob_06 = df0_glob[df0_glob['ID'].isin(lista_df0_glob)]
print(f'df0_glob KGE >= {KGE_THRESHOLD}: {df0_glob_06.ID.unique().shape[0]} estaciones')
print("Cálculo de KGE completado.")

#################################################################################
################ FUNCIÓN QUE FILTRA LOS CAUDALES ALTOS EN QOBS ##################
#################################################################################

print("Filtrando caudales altos...")
def filtrar_caudales_altos(df, column_qobs, column_id, percentile):
    df_copy = df.copy()
    threshold = df_copy.groupby(column_id)[column_qobs].quantile(percentile / 100).reset_index()
    threshold.columns = [column_id, 'umbral']
    dfs = pd.merge(df_copy, threshold, on=column_id, how = 'left')
    df_high_flow = dfs[dfs[column_qobs] >= dfs['umbral']]
    df_high_flow = df_high_flow.drop(columns=['umbral'])
    return df_high_flow

# PERCENTIL 99
df0_high_glob_p99 = filtrar_caudales_altos(df = df0_glob, column_qobs = 'Qobs', column_id = 'ID', percentile =  99.0)
df0_high_loc_p99 = filtrar_caudales_altos(df = df0_loc, column_qobs = 'Qobs', column_id = 'ID', percentile =  99.0)
df0_high_glob_p99 = df0_high_glob_p99[df0_high_glob_p99['mask']==MASK_VALUE]
df0_high_loc_p99 = df0_high_loc_p99[df0_high_loc_p99['mask']==MASK_VALUE]

df0_high_glob_p99_06 = filtrar_caudales_altos(df = df0_glob_06, column_qobs = 'Qobs', column_id = 'ID', percentile =  99.0)
df0_high_loc_p99_06 = filtrar_caudales_altos(df = df0_loc_06, column_qobs = 'Qobs', column_id = 'ID', percentile =  99.0)
df0_high_glob_p99_06 = df0_high_glob_p99_06[df0_high_glob_p99_06['mask']==MASK_VALUE]
df0_high_loc_p99_06 = df0_high_loc_p99_06[df0_high_loc_p99_06['mask']==MASK_VALUE]

# PERCENTIL 90
df0_high_glob_p90 = filtrar_caudales_altos(df = df0_glob, column_qobs = 'Qobs', column_id = 'ID', percentile =  90.0)
df0_high_loc_p90 = filtrar_caudales_altos(df = df0_loc, column_qobs = 'Qobs', column_id = 'ID', percentile =  90.0)
df0_high_glob_p90 = df0_high_glob_p90[df0_high_glob_p90['mask']==MASK_VALUE]
df0_high_loc_p90 = df0_high_loc_p90[df0_high_loc_p90['mask']==MASK_VALUE]

df0_high_glob_p90_06 = filtrar_caudales_altos(df = df0_glob_06, column_qobs = 'Qobs', column_id = 'ID', percentile =  90.0)
df0_high_loc_p90_06 = filtrar_caudales_altos(df = df0_loc_06, column_qobs = 'Qobs', column_id = 'ID', percentile =  90.0)
df0_high_glob_p90_06 = df0_high_glob_p90_06[df0_high_glob_p90_06['mask']==MASK_VALUE]
df0_high_loc_p90_06 = df0_high_loc_p90_06[df0_high_loc_p90_06['mask']==MASK_VALUE]
print("Filtrado de caudales altos completado.")

#####################################################
############# CALCULA LA METRICA FHV ################
#####################################################

print("Calculando métrica FHV...")
def calcular_bias_fhv(df, column_qsim, column_qobs, column_id):
    df_copy = df.copy() 
    # Filtrar IDs donde la suma de Qobs es cero para evitar división por cero
    sum_qobs_per_id = df_copy.groupby(column_id)[column_qobs].sum()
    valid_ids = sum_qobs_per_id[sum_qobs_per_id != 0].index
    df_filtered = df_copy[df_copy[column_id].isin(valid_ids)]

    if df_filtered.empty:
        # Devolver un DataFrame vacío con las columnas esperadas si no hay datos válidos
        return pd.DataFrame(columns=[column_id, 'Percent_BiasFHV'])

    results = df_filtered.groupby(column_id).apply(
        lambda x: (((x[column_qsim] - x[column_qobs]).sum()) / x[column_qobs].sum()) * 100
        if x[column_qobs].sum() != 0 else np.nan # Doble chequeo por si acaso
    ).reset_index()
    results.columns = [column_id, 'Percent_BiasFHV']
    results.dropna(subset=['Percent_BiasFHV'], inplace=True) 
    return results

# PERCENTIL 99
bias_fhv_glob_p99 = calcular_bias_fhv(df = df0_high_glob_p99, column_qsim = 'Qsim', column_qobs = 'Qobs', column_id = 'ID')
bias_fhv_loc_p99 = calcular_bias_fhv(df = df0_high_loc_p99, column_qsim = 'Qsim', column_qobs = 'Qobs', column_id = 'ID')
bias_fhv_glob_p99_06 = calcular_bias_fhv(df = df0_high_glob_p99_06, column_qsim = 'Qsim', column_qobs = 'Qobs', column_id = 'ID')
bias_fhv_loc_p99_06 = calcular_bias_fhv(df = df0_high_loc_p99_06, column_qsim = 'Qsim', column_qobs = 'Qobs', column_id = 'ID')

# PERCENTIL 90
bias_fhv_glob_p90 = calcular_bias_fhv(df = df0_high_glob_p90, column_qsim = 'Qsim', column_qobs = 'Qobs', column_id = 'ID')
bias_fhv_loc_p90 = calcular_bias_fhv(df = df0_high_loc_p90, column_qsim = 'Qsim', column_qobs = 'Qobs', column_id = 'ID')
bias_fhv_glob_p90_06 = calcular_bias_fhv(df = df0_high_glob_p90_06, column_qsim = 'Qsim', column_qobs = 'Qobs', column_id = 'ID')
bias_fhv_loc_p90_06 = calcular_bias_fhv(df = df0_high_loc_p90_06, column_qsim = 'Qsim', column_qobs = 'Qobs', column_id = 'ID')
print("Cálculo de métrica FHV completado.")
print("Ejemplo de descripción de bias_fhv_glob_p90:")
print(bias_fhv_glob_p90.describe().round(2))

# --------------------------------------------------------------------------- #
# PARTE 2: PLOTEO DE MAPAS
# --------------------------------------------------------------------------- #
print("\nIniciando preparación para el ploteo de mapas...")

# --------------------------------------------------------------------------- #
# Carga del DataFrame de Coordenadas
# --------------------------------------------------------------------------- #
coords_file_path = f"{DRIVE_PATH}/Graficas_para_Mapas/gauge_and_outlet_coordinates.csv"
coords_master_df = pd.DataFrame() # Inicializar como DataFrame vacío
try:
    coords_master_df = pd.read_csv(coords_file_path)


    required_coord_cols = ['gauge_id', 'outlet_camels_lon', 'outlet_camels_lat']
    if not all(col in coords_master_df.columns for col in required_coord_cols):
        print(f"ADVERTENCIA: El archivo de coordenadas {coords_file_path} no tiene todas las columnas requeridas: {required_coord_cols}.")
        print(f"Columnas encontradas: {coords_master_df.columns.tolist()}")
        print("El ploteo de mapas podría no funcionar correctamente.")
        # Continuar con un DataFrame vacío podría ser problemático, pero lo dejamos para que el error sea evidente más adelante si se usa.
    else:
        print(f"Archivo de coordenadas '{coords_file_path}' cargado exitosamente con {len(coords_master_df)} registros.")

except FileNotFoundError:
    print(f"ERROR CRÍTICO: Archivo de coordenadas no encontrado en {coords_file_path}")
    print("El ploteo de mapas no podrá realizarse sin este archivo.")
   
except Exception as e:
    print(f"ERROR CRÍTICO al cargar el archivo de coordenadas: {e}")
    # Igual que arriba.

datasets_to_plot = {
    'bias_fhv_glob_p99': bias_fhv_glob_p99,
    'bias_fhv_loc_p99': bias_fhv_loc_p99,
    'bias_fhv_glob_p99_06': bias_fhv_glob_p99_06,
    'bias_fhv_loc_p99_06': bias_fhv_loc_p99_06,
    'bias_fhv_glob_p90': bias_fhv_glob_p90,
    'bias_fhv_loc_p90': bias_fhv_loc_p90,
    'bias_fhv_glob_p90_06': bias_fhv_glob_p90_06,
    'bias_fhv_loc_p90_06': bias_fhv_loc_p90_06
}

processed_map_data = {}
for key, bias_df_actual in datasets_to_plot.items():
    print(f"Procesando datos del mapa para: {key}")
    if coords_master_df.empty:
        print(f"  ADVERTENCIA: El DataFrame de coordenadas (coords_master_df) está vacío. No se pueden agregar coordenadas para {key}.")
        processed_map_data[key] = pd.DataFrame(columns=['ID', 'Percent_BiasFHV', 'outlet_camels_lon', 'outlet_camels_lat'])
        continue
    
    if bias_df_actual.empty:
        print(f"  ADVERTENCIA: El DataFrame de bias FHV ({key}) está vacío. No hay datos para fusionar.")
        processed_map_data[key] = pd.DataFrame(columns=['ID', 'Percent_BiasFHV', 'outlet_camels_lon', 'outlet_camels_lat'])
        continue

    # Fusionar el bias_df_actual (que tiene 'ID') con coords_master_df (que tiene 'gauge_id')
    map_data = pd.merge(bias_df_actual, coords_master_df, left_on='ID', right_on='gauge_id', how='inner')
    
    # Asegurarse de que las columnas necesarias para el ploteo existen después del merge
    map_data.dropna(subset=['Percent_BiasFHV', 'outlet_camels_lon', 'outlet_camels_lat'], inplace=True)
    processed_map_data[key] = map_data
    print(f"  Se encontraron {len(map_data)} estaciones con datos FHV válidos y coordenadas para {key}")

# --------------------------------------------------------------------------- #
# Map Plotting
# --------------------------------------------------------------------------- #
print("\nIniciando ploteo de mapas...")
# Load raster and shapefile once
hillshade_raster_path = f"{DRIVE_PATH}/Graficas_para_Mapas/hill.tif"
hillshade_raster = None
try:
    hillshade_raster = rasterio.open(hillshade_raster_path)
    print("Hillshade raster cargado.")
except rasterio.errors.RasterioIOError:
    print(f"ADVERTENCIA: No se pudo cargar el archivo hillshade: {hillshade_raster_path}")
except Exception as e:
    print(f"ADVERTENCIA: Error inesperado al cargar hillshade: {e}")


chile_shapefile_path = f"{DRIVE_PATH}/Graficas_para_Mapas/SHP/Chile_arg_per_bol.shp"
df_chile_shp = None
try:
    df_chile_shp = gpd.read_file(chile_shapefile_path)
    print("Shapefile de Chile cargado.")
except Exception as e:
    print(f"ADVERTENCIA: No se pudo cargar el archivo shapefile: {chile_shapefile_path}. Error: {e}")

# Define colormap and normalization
bias_colors_list = ["#3B5F91", "#7AB8E6", "#3a9308", "#3a9308", "#FFEA00", "#D4A020"]
bias_boundaries = [-100, -50, -25, 0, 25, 50, 100]
bias_cmap = mcolors.ListedColormap(bias_colors_list)
bias_norm = mcolors.BoundaryNorm(bias_boundaries, bias_cmap.N)
colorbar_ticks = [-50, -25, 0, 25, 50]

# Create the 2x4 subplot figure
fig, axes = plt.subplots(2, 4, figsize=(18, 17), dpi=600, sharex=True, sharey=True)
fig.subplots_adjust(left=0.06, right=0.88, bottom=0.05, top=0.94, wspace=0.0001, hspace=0.06)

plot_order = [
    'bias_fhv_glob_p99', 'bias_fhv_loc_p99', 'bias_fhv_glob_p99_06', 'bias_fhv_loc_p99_06',
    'bias_fhv_glob_p90', 'bias_fhv_loc_p90', 'bias_fhv_glob_p90_06', 'bias_fhv_loc_p90_06'
]
subplot_titles_short = [
    f"a) RM P99", f"b) LM P99", f"c) RM P99 (KGE≥{KGE_THRESHOLD})", f"d) LM P99 (KGE≥{KGE_THRESHOLD})",
    f"e) RM P90", f"f) LM P90", f"g) RM P90 (KGE≥{KGE_THRESHOLD})", f"h) LM P90 (KGE≥{KGE_THRESHOLD})"
]

scatter_plot_reference = None # Para la referencia de la barra de color

any_data_plotted = False # Flag para ver si algún subplot tiene datos

for i, key in enumerate(plot_order):
    row = i // 4
    col = i % 4
    ax = axes[row, col]
    data_to_plot = processed_map_data.get(key)

    if hillshade_raster:
        try:
            rasterio.plot.show(hillshade_raster, cmap=mcolors.LinearSegmentedColormap.from_list("", ["#CECECE", "#FFFFFF"]),
                                ax=ax, interpolation='none', vmin=9000, vmax=22000)
        except Exception as e:
            print(f"Error al plotear hillshade en subplot {key}: {e}")
    if df_chile_shp is not None:
        try:
            df_chile_shp.boundary.plot(color='grey', lw=0.4, ax=ax)
        except Exception as e:
            print(f"Error al plotear shapefile en subplot {key}: {e}")


    if data_to_plot is not None and not data_to_plot.empty:
        if not {'outlet_camels_lon', 'outlet_camels_lat', 'Percent_BiasFHV'}.issubset(data_to_plot.columns):
            ax.text(0.5, 0.5, "Faltan columnas\npara plotear", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=8, color='red')
            print(f"  ADVERTENCIA: Para {key}, faltan columnas de coordenadas o FHV en data_to_plot. Columnas presentes: {data_to_plot.columns.tolist()}")
        else:
            current_scatter = ax.scatter(data_to_plot['outlet_camels_lon'], data_to_plot['outlet_camels_lat'],
                                         c=data_to_plot['Percent_BiasFHV'], cmap=bias_cmap, norm=bias_norm,
                                         edgecolor='k', linewidth=0.5, s=15, zorder=10)
            if scatter_plot_reference is None:
                scatter_plot_reference = current_scatter
            any_data_plotted = True 
    else:
        ax.text(0.5, 0.5, "No data available\nor no coordinates", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=8)

    ax.set_title(subplot_titles_short[i], fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim([-76.5, -65.5])
    ax.set_ylim([-58, -16])

    if col == 0:
        ax.set_ylabel('Latitude', fontsize=16)
    if row == 1:
        ax.set_xlabel('Longitude', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)

    if i == 0: # Flecha Norte solo en el primer plot
        x_arrow, y_arrow, arrow_length = 0.12, 0.95, 0.05
        ax.annotate('N', xy=(x_arrow, y_arrow), xytext=(x_arrow, y_arrow - arrow_length),
                    arrowprops=dict(facecolor='black', width=3, headwidth=8),
                    ha='center', va='center', fontsize=12, xycoords=ax.transAxes)

if any_data_plotted and scatter_plot_reference:
    cbar_ax = fig.add_axes([0.90, 0.25, 0.025, 0.5])
    cbar = fig.colorbar(scatter_plot_reference, cax=cbar_ax, orientation='vertical', ticks=colorbar_ticks, extend='both', spacing='proportional')
    cbar.set_label('Percent BiasFHV [%]', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_yticklabels([str(t) for t in colorbar_ticks])
else:
    print("ADVERTENCIA: No se plotearon datos con coordenadas, por lo tanto no se generará la barra de color.")

# Ajustar layout para evitar superposiciones
# fig.tight_layout(rect=[0.02, 0.02, 0.88, 0.95]) # tight_layout a veces no juega bien con add_axes para colorbar

output_filename = f"{DRIVE_PATH}/Combined_FHV_Maps_P99_P90_COMPLETO.png"
try:
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    print(f"Mapa combinado guardado en {output_filename}")
except Exception as e:
    print(f"Error al guardar la figura: {e}")

plt.show()
print("Proceso de ploteo completado.")