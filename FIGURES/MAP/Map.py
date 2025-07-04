# Installation of optional geospatial libraries (kept as requested)
# pip install geopandas
# pip install rasterio

# --------------------------------------------------------------------------- #
# Necessary Libraries (based on usage in this script)
# --------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
import rasterio.plot
import matplotlib.colors as cl # For colormaps and normalization
import matplotlib.cm as cm     # For base colormaps like viridis
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

###### ###### ###### ######
##### MEAN FLOW      ######## (Loading Mean Flow Model Results)
###### ###### ###### ######

# Directory path for model results
dir_path = '/content/drive/MyDrive/FONDEF/Variables/LSTM3/Ult_Model_LSTM/Qmean/Qmean3'
# dir_path = '/content/drive/MyDrive/FONDEF/Variables/PRED_NEW' # Alternative path, commented out

# --- Load results for prediction day 0 (p0d) ---
ruta0 = f'{dir_path}/Qmean_LSTM_p0d.csv'
df0 = pd.read_csv(ruta0, float_precision='legacy')
# Scale flow-related columns (unit conversion or denormalization)
cols_to_scale = ['Qsim', 'Qobs', 'mean', 'std_dev', 'median', 'per_5', 'per_25', 'per_75', 'per_95']
df0[cols_to_scale] = df0[cols_to_scale] * 10

# --- Load results for prediction day 1 (p1d) ---
# Note: df1 is loaded but not used in the subsequent KGE calculations or plots in this script.
ruta1 = f'{dir_path}/Qmean_LSTM_p1d.csv'
df1 = pd.read_csv(ruta1, float_precision='legacy')
df1[cols_to_scale] = df1[cols_to_scale] * 10

# --- Load results for prediction day 2 (p2d) ---
# Note: df2 (this one, not the one created later from df_kge and coords) is loaded but not used.
ruta2 = f'{dir_path}/Qmean_LSTM_p2d.csv'
df2_loaded = pd.read_csv(ruta2, float_precision='legacy') # Renamed to avoid conflict
df2_loaded[cols_to_scale] = df2_loaded[cols_to_scale] * 10

# --- Load results for prediction day 3 (p3d) ---
# Note: df3 is loaded but not used.
ruta3 = f'{dir_path}/Qmean_LSTM_p3d.csv'
df3 = pd.read_csv(ruta3, float_precision='legacy')
df3[cols_to_scale] = df3[cols_to_scale] * 10

# --- Load results for prediction day 4 (p4d) ---
# Note: df4 is loaded but not used.
ruta4 = f'{dir_path}/Qmean_LSTM_p4d.csv'
df4 = pd.read_csv(ruta4, float_precision='legacy')
df4[cols_to_scale] = df4[cols_to_scale] * 10

# --------------------------------------------------------------------------- #
# KGE (Kling-Gupta Efficiency) Calculation Functions
# --------------------------------------------------------------------------- #
def KGE(sim, obs):
    """
    Calculates the Kling-Gupta Efficiency (KGE) and its components.
    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    where:
        r: Pearson correlation coefficient between sim and obs.
        alpha: Ratio of standard deviations (sim.std() / obs.std()). Represents variability.
        beta: Ratio of means (sim.mean() / obs.mean()). Represents bias.
    Returns:
        KGE_val (float): The KGE value.
        corr_coeff (float): Pearson correlation coefficient.
        alpha (float): Variability ratio.
        beta (float): Bias ratio.
    """
    # Ensure inputs are numpy arrays for corrcoef if they are pandas Series
    sim_val = sim.values if isinstance(sim, pd.Series) else sim
    obs_val = obs.values if isinstance(obs, pd.Series) else obs

    # Handle potential NaN or zero standard deviation issues for alpha
    if np.std(obs_val) == 0 or pd.isna(np.std(obs_val)) or pd.isna(np.std(sim_val)):
        alpha = np.nan
    else:
        alpha = np.std(sim_val) / np.std(obs_val)

    # Handle potential NaN or zero mean issues for beta
    if np.mean(obs_val) == 0 or pd.isna(np.mean(obs_val)) or pd.isna(np.mean(sim_val)):
        beta = np.nan
    else:
        beta = np.mean(sim_val) / np.mean(obs_val)
    
    # Calculate Pearson correlation coefficient, handle NaNs if inputs are all NaN
    if np.all(np.isnan(sim_val)) or np.all(np.isnan(obs_val)) or len(sim_val) < 2 or len(obs_val) < 2:
        corr_coeff = np.nan
    else:
        # Replace NaNs with mean for correlation calculation to avoid error with all-NaN slices
        sim_val_no_nan = np.nan_to_num(sim_val, nan=np.nanmean(sim_val))
        obs_val_no_nan = np.nan_to_num(obs_val, nan=np.nanmean(obs_val))
        if np.all(sim_val_no_nan == sim_val_no_nan[0]) or np.all(obs_val_no_nan == obs_val_no_nan[0]): # Handle constant series
             corr_coeff = np.nan
        else:
            corr_coeff_matrix = np.corrcoef(sim_val_no_nan, obs_val_no_nan)
            corr_coeff = corr_coeff_matrix[0, 1]


    # Calculate KGE. If any component is NaN, KGE is NaN.
    if pd.isna(corr_coeff) or pd.isna(alpha) or pd.isna(beta):
        KGE_val = np.nan
    else:
        KGE_val = 1 - np.sqrt((corr_coeff - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return KGE_val, corr_coeff, alpha, beta


def compute_KGE_for_each_ID(df):
    """Calculates KGE and its components for each unique ID in the DataFrame."""
    # Group by 'ID' and apply the KGE function to each group
    # Using .dropna() on Qsim/Qobs for KGE calculation to avoid issues with missing data within a series
    kge_results = df.groupby('ID').apply(lambda x: KGE(x['Qsim'].dropna(), x['Qobs'].dropna())[0])
    corr_results = df.groupby('ID').apply(lambda x: KGE(x['Qsim'].dropna(), x['Qobs'].dropna())[1])
    var_results = df.groupby('ID').apply(lambda x: KGE(x['Qsim'].dropna(), x['Qobs'].dropna())[2])
    sesgo_results = df.groupby('ID').apply(lambda x: KGE(x['Qsim'].dropna(), x['Qobs'].dropna())[3])

    # Create DataFrames with the results
    kge_df = kge_results.reset_index(name='KGE')
    corr_df = corr_results.reset_index(name='Corr')    # Correlation (r)
    var_df = var_results.reset_index(name='Var')      # Variability (alpha)
    sesgo_df = sesgo_results.reset_index(name='Sesgo')  # Bias (beta)

    return kge_df, corr_df, var_df, sesgo_df

# --------------------------------------------------------------------------- #
# KGE Calculation and Statistics for p0d (Test Set)
# --------------------------------------------------------------------------- #
# Filter p0d data for the test set (mask == 2)
df00 = df0[df0['mask'] == 2]

# Calculate KGE and its components for the filtered p0d data
# Note: This is the first calculation of KGE for df00.
kge_df0, corr_df0, var_df0, sesgo_df0 = compute_KGE_for_each_ID(df00)

# Print statistics based on KGE values for p0d
print('KGE Statistics for p0d (Test Set):')
print(f"Stations with KGE >= 0.6: {kge_df0[kge_df0['KGE'] >= 0.595].shape[0]}")
print(f"Stations with KGE >= 0.5: {kge_df0[kge_df0['KGE'] >= 0.495].shape[0]}") # Note: Original had 0.5, script has 0.495
print(f"Stations with KGE >= 0: {kge_df0[kge_df0['KGE'] >= 0].shape[0]}")
print(f"Stations with KGE < 0: {kge_df0[kge_df0['KGE'] < 0].shape[0]}")

#####################
#### COORDINATES ####
#####################

# Filter data for mask == 2 for all loaded forecast horizons

df00 = df0[df0['mask'] == 2] 
df10 = df1[df1['mask'] == 2]
df20_filtered = df2_loaded[df2_loaded['mask'] == 2] 
df30 = df3[df3['mask'] == 2]
df40 = df4[df4['mask'] == 2]

# Calculate KGE and its components again for df00 (this is redundant)
df_kge, corr_df, var_df, sesgo_df = compute_KGE_for_each_ID(df00)
# Merge KGE components into a single DataFrame
df_kge = pd.merge(df_kge, corr_df, on='ID')
df_kge = pd.merge(df_kge, var_df, on='ID')
df_kge = pd.merge(df_kge, sesgo_df, on='ID')
print(f"Shape of merged KGE data before coordinate merge: {df_kge.shape}")
print(f"Number of stations with KGE >= 0.6: {df_kge[df_kge['KGE'] >= 0.6].shape[0]}")

# Load station coordinate data
coords = pd.read_csv('/content/drive/MyDrive/FONDEF/Variables/Graficas_para_Mapas/gauge_and_outlet_coordinates.csv')

# Filter coordinates to include only stations present in the KGE results
coords = coords[coords['gauge_id'].isin(df_kge['ID'])]

# Merge KGE data with coordinates. This DataFrame will be used for plotting.
df_plot_data = pd.merge(df_kge, coords, left_on='ID', right_on='gauge_id')
df_plot_data.sort_values(by='KGE', ascending=True, inplace=True) 

###############################################################################################
################# MAP FOR THE PAPER ###########################################################
# This section creates a 4-panel map showing KGE, Correlation, Variance Bias, and Mean Bias.
# All data plotted is from df_plot_data, which pertains to p0d results.
###############################################################################################

# Load raster file for hillshade background
try:
    r = rasterio.open('/content/drive/MyDrive/FONDEF/Variables/Graficas_para_Mapas/hill.tif')
except rasterio.errors.RasterioIOError:
    print("Error: Could not open hillshade raster file. Plotting will proceed without it.")
    r = None # Set r to None if file not found

# Set hillshade colormap
cmapg = cl.LinearSegmentedColormap.from_list("", ["#CECECE", "#FFFFFF"]) # Grayscale for hillshade
# Load shapefile for country boundaries
df_chile = gpd.read_file('/content/drive/MyDrive/FONDEF/Variables/Graficas_para_Mapas/SHP/Chile_arg_per_bol.shp')

# --- Figure Setup ---
num_subplots = 4
rows = 1
cols = 4
fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5.26 * cols, 10), dpi=600) 
plt.rcParams['font.size'] = '14' 

# Common settings for North arrow
x_arrow, y_arrow, arrow_length = 0.14, 0.98, 0.05 # Relative coordinates for North arrow


# --- Panel 1: KGE Map ---
ax1 = axs[0]
if r: # Plot hillshade if raster was loaded successfully
    rasterio.plot.show(r, cmap=cmapg, interpolation='none', vmin=9000, vmax=22000, ax=ax1)
df_chile.boundary.plot(color='grey', lw=0.4, ax=ax1) # Plot country boundaries

# Define colormap and normalization for KGE
kge_colors = ["#fdc729", "#cac729","#45b705","#41a608","#3a9308","#2f850a","#226e02"]  # Yellow to Green
kge_bins = [0.4, 0.5, 0.595, 0.7, 0.8, 0.9, 1.0] # Bins for KGE values
kge_cmap = cl.LinearSegmentedColormap.from_list("kge_map", kge_colors, N=len(kge_colors))
kge_norm = cl.BoundaryNorm(kge_bins, kge_cmap.N)

# Scatter plot for KGE values
im1 = ax1.scatter(df_plot_data['outlet_camels_lon'], df_plot_data['outlet_camels_lat'],
                  c=df_plot_data['KGE'], cmap=kge_cmap, marker='.', norm=kge_norm)
ax1.grid(True, linestyle='--', alpha=0.5)
# Colorbar for KGE
cbar1 = fig.colorbar(im1, ax=ax1, spacing='proportional', boundaries=kge_bins, extend='min')
kge_ticks_labels = ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'] # Custom tick labels
cbar1.set_ticks(kge_bins)
cbar1.set_ticklabels(kge_ticks_labels)

ax1.text(-76, -57.8, 'a)', fontsize=30)
ax1.set_xlabel('Longitude', fontsize=32)
ax1.set_ylabel('Latitude', fontsize=32)
cbar1.ax.tick_params(labelsize=22)
ax1.tick_params(axis='both', which='major', labelsize=22)
ax1.set_xlim([-76.5, -65.5])
ax1.set_ylim([-58, -16])
ax1.annotate('N', xy=(x_arrow, y_arrow), xytext=(x_arrow, y_arrow - arrow_length),
             arrowprops=dict(facecolor='black', width=4, headwidth=10),
             ha='center', va='center', fontsize=15, xycoords=ax1.transAxes)


# --- Panel 2: Correlation (r) Map ---
ax2 = axs[1]
if r:
    rasterio.plot.show(r, cmap=cmapg, interpolation='none', vmin=9000, vmax=22000, ax=ax2)
df_chile.boundary.plot(color='grey', lw=0.4, ax=ax2)

# Colormap and normalization for Correlation (using similar scheme as KGE for consistency if desired)
corr_colors = ["#fdc729", "#cac729", "#45b705", "#41a608", "#3a9308", "#2f850a", "#226e02"] # Yellow to Green
corr_bins = [0, 0.5, 0.595, 0.7, 0.8, 0.9, 1.0] # Bins for Correlation values
corr_cmap = cl.LinearSegmentedColormap.from_list("corr_map", corr_colors, N=len(corr_colors))
corr_norm = cl.BoundaryNorm(corr_bins, corr_cmap.N)

# Scatter plot for Correlation values
im2 = ax2.scatter(df_plot_data['outlet_camels_lon'], df_plot_data['outlet_camels_lat'],
                  c=df_plot_data['Corr'], cmap=corr_cmap, marker='.', norm=corr_norm)
ax2.grid(True, linestyle='--', alpha=0.5)
# Colorbar for Correlation
cbar2 = fig.colorbar(im2, ax=ax2, spacing='proportional', boundaries=corr_bins, extend='min')
corr_ticks_labels = ['0', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
cbar2.set_ticks(corr_bins)
cbar2.set_ticklabels(corr_ticks_labels)
# cbar2.set_label('Correlation (r)', fontsize=12)

ax2.text(-76, -57.8, 'b)', fontsize=30)
cbar2.ax.tick_params(labelsize=22)
ax2.tick_params(axis='both', which='major', labelsize=22)
ax2.set_xlim([-76.5, -65.5])
ax2.set_ylim([-58, -16])
ax2.annotate('N', xy=(x_arrow, y_arrow), xytext=(x_arrow, y_arrow - arrow_length),
             arrowprops=dict(facecolor='black', width=4, headwidth=10),
             ha='center', va='center', fontsize=15, xycoords=ax2.transAxes)


# --- Panel 3: Bias in Variance (Alpha) Map ---
ax3 = axs[2]
if r:
    rasterio.plot.show(r, cmap=cmapg, interpolation='none', vmin=9000, vmax=22000, ax=ax3)
df_chile.boundary.plot(color='grey', lw=0.4, ax=ax3)

# Colormap and normalization for Variance Bias (Alpha component of KGE)
var_colors = ["#3B5F91", "#7AB8E6", "#2f850a", "#2f850a",  "#FFEA00", "#cac729", "#D4A020"] # Blue -> Green (ideal) -> Yellow/Orange
var_bins = [0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0] # Bins centered around 1 (ideal)
var_cmap = cl.LinearSegmentedColormap.from_list("var_map", var_colors, N=len(var_colors))
var_norm = cl.BoundaryNorm(var_bins, var_cmap.N)

# Scatter plot for Variance Bias
im3 = ax3.scatter(df_plot_data['outlet_camels_lon'], df_plot_data['outlet_camels_lat'],
                  c=df_plot_data['Var'], cmap=var_cmap, marker='.', norm=var_norm)
ax3.grid(True, linestyle='--', alpha=0.5)
# Colorbar for Variance Bias
cbar3 = fig.colorbar(im3, ax=ax3, spacing='proportional', boundaries=var_bins, extend='both') # Extend both for values outside range
var_ticks_labels = ['0.5', '0.6', '0.8', '1', '1.25', '1.5', '2']
cbar3.set_ticks(var_bins)
cbar3.set_ticklabels(var_ticks_labels)


ax3.text(-76, -57.8, 'c)', fontsize=30)
cbar3.ax.tick_params(labelsize=22)
ax3.tick_params(axis='both', which='major', labelsize=22)
ax3.set_xlim([-76.5, -65.5])
ax3.set_ylim([-58, -16])
ax3.annotate('N', xy=(x_arrow, y_arrow), xytext=(x_arrow, y_arrow - arrow_length),
             arrowprops=dict(facecolor='black', width=4, headwidth=10),
             ha='center', va='center', fontsize=15, xycoords=ax3.transAxes)


# --- Panel 4: Bias in Mean (Beta) Map ---
ax4 = axs[3]
if r:
    rasterio.plot.show(r, cmap=cmapg, interpolation='none', vmin=9000, vmax=22000, ax=ax4)
df_chile.boundary.plot(color='grey', lw=0.4, ax=ax4)


# Scatter plot for Mean Bias
im4 = ax4.scatter(df_plot_data['outlet_camels_lon'], df_plot_data['outlet_camels_lat'],
                  c=df_plot_data['Sesgo'], cmap=var_cmap, marker='.', norm=var_norm) # Reusing var_cmap and var_norm
ax4.grid(True, linestyle='--', alpha=0.5)
# Colorbar for Mean Bias
cbar4 = fig.colorbar(im4, ax=ax4, spacing='proportional', boundaries=var_bins, extend='both')
# var_ticks_labels already defined
cbar4.set_ticks(var_bins)
cbar4.set_ticklabels(var_ticks_labels)

ax4.text(-76, -57.8, 'd)', fontsize=30)
cbar4.ax.tick_params(labelsize=22)
ax4.tick_params(axis='both', which='major', labelsize=22)
ax4.set_xlim([-76.5, -65.5])
ax4.set_ylim([-58, -16])
ax4.annotate('N', xy=(x_arrow, y_arrow), xytext=(x_arrow, y_arrow - arrow_length),
             arrowprops=dict(facecolor='black', width=4, headwidth=10),
             ha='center', va='center', fontsize=15, xycoords=ax4.transAxes)

plt.tight_layout() # Adjust layout to prevent overlapping elements
#plt.savefig('/content/MAPA_Qmean_KGE_p0d_GLOBAL.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
