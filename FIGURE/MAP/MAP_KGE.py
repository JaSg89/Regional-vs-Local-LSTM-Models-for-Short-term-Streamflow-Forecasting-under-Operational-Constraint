"""
Hydrological Model Performance Analysis & Mapping (GitHub Backup)
Language: Python 3
Dependencies: geopandas, rasterio, matplotlib, pandas, numpy

Install:
    pip install geopandas rasterio
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import rasterio
import rasterio.plot
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION (PATHS)
# ==========================================
PATHS = {
    'attributes': '/content/drive/MyDrive/FONDEF/Variables/DF_LSTM/LSTM11.csv',
    'regional_model': '/content/drive/MyDrive/FONDEF/Variables/LSTM3/Ult_Model_LSTM/Qmean/Qmean3/Qmean_LSTM_p0d.csv',
    'local_model': '/content/drive/MyDrive/FONDEF/Variables/ENTRENAMIENTO_LOCAL/Trained_Models/Qmean_LSTM_p0d_Local.csv',
    'coords': '/content/drive/MyDrive/FONDEF/Variables/Graficas_para_Mapas/gauge_and_outlet_coordinates.csv',
    'hillshade': '/content/drive/MyDrive/FONDEF/Variables/Graficas_para_Mapas/hill.tif',
    'shapefile': '/content/drive/MyDrive/FONDEF/Variables/Graficas_para_Mapas/SHP/Chile_arg_per_bol.shp',
    'output_fig': '/content/MAPA_Regional_subtle_zones.png'
}

# ==========================================
# 2. DATA PROCESSING FUNCTIONS
# ==========================================

def load_catchment_area(path):
    """
    Loads only the catchment area and date/ID for unit conversion.
    Precipitation columns have been removed.
    """
    # Only loading ID, date, and Area (needed for the 864 conversion)
    cols = ['gauge_id', 'date', 'top_s_cam_area_tot_b_none_c_c']

    data = pd.read_csv(path, usecols=cols)
    data.rename(columns={'gauge_id': 'ID'}, inplace=True)

    return data

def process_model_data(path, area_data):
    """
    Loads model results, merges with area data, and applies unit conversion.
    Strict adherence to original preprocessing logic.
    """
    df = pd.read_csv(path, float_precision='legacy')
    df = df[df['ID'] != 8380001]

    # 1. Scale model columns by 10 (as per original logic)
    cols_model = ['Qsim', 'Qobs', 'mean', 'std_dev', 'median', 'per_5',
                  'per_25', 'per_75', 'per_95', 'crps']
    df[cols_model] = df[cols_model] * 10

    # 2. Merge with Area data
    df = pd.merge(df, area_data, left_on=['ID', 'date'], right_on=['ID', 'date'], how='inner')

    # 3. Unit conversion using Area / 864
    for column in cols_model:
        df[column] = (df[column] * df['top_s_cam_area_tot_b_none_c_c']) / 864

    return df

def calculate_kge_stats(sim, obs):
    """Calculates KGE components. Returns tuple."""
    if len(sim) == 0: return np.nan, np.nan, np.nan, np.nan

    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    corr_coeff = np.corrcoef(sim, obs)[0, 1]

    kge_val = 1 - np.sqrt((corr_coeff - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge_val, corr_coeff, alpha, beta

def compute_metrics_dataframe(df):
    """
    Computes KGE, Correlation, Variance (Alpha), and Bias (Beta) for each ID.
    """
    results = []
    # GroupBy approach to calculate metrics per basin
    for gauge_id, group in df.groupby('ID'):
        kge, r, alpha, beta = calculate_kge_stats(group['Qsim'], group['Qobs'])
        results.append([gauge_id, kge, r, alpha, beta])

    metrics_df = pd.DataFrame(results, columns=['ID', 'KGE', 'Corr', 'Var', 'Sesgo'])
    return metrics_df

# ==========================================
# 3. PLOTTING FUNCTION
# ==========================================

def plot_performance_maps(
    results_df, coords_path, hillshade_path, shapefile_path,
    output_filename=None, title=None
):
    # Load geospatial data
    coords = pd.read_csv(coords_path)
    df_chile = gpd.read_file(shapefile_path)

    try:
        r = rasterio.open(hillshade_path)
        cmapg = cl.LinearSegmentedColormap.from_list("", ["#CECECE", "#FFFFFF"])
    except:
        print("Warning: Hillshade not found. Plotting without it.")
        r = None

    # Filter and merge coordinates
    coords_filtered = coords[coords['gauge_id'].isin(results_df['ID'])]
    df_plot_data = pd.merge(results_df, coords_filtered, left_on='ID', right_on='gauge_id')
    df_plot_data.sort_values(by='KGE', ascending=True, inplace=True)

    # Plot Configurations (Strictly original parameters)
    kge_conf = {
        'metric': 'KGE', 'label': 'a)',
        'colors': ["#D4A020", "#FFEA00","#45b705","#41a608","#226e02","#226e02","#226e02"],
        'bins': [0.4, 0.5, 0.595, 0.7, 0.8, 0.9, 1.0],
        'ticks': ['', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
        'extend': 'min'
    }
    corr_conf = {
        'metric': 'Corr', 'label': 'b)',
        'colors': ["#D4A020", "#FFEA00", "#41a608", "#226e02"],
        'bins': [0, 0.5, 0.595, 0.8, 1.0],
        'ticks': ['', '0.5', '0.6', '0.8', '1.0'],
        'extend': 'min'
    }
    bias_conf = {
        'colors': ["#3B5F91", "#7AB8E6", "#2f850a", "#2f850a",  "#FFEA00", "#cac729", "#D4A020"],
        'bins': [0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0],
        'ticks': ['', '0.6', '0.8', '1', '1.25', '1.5', '2'],
        'extend': 'both'
    }

    plot_configs = [
        kge_conf,
        corr_conf,
        {**bias_conf, 'metric': 'Var', 'label': 'c)'},
        {**bias_conf, 'metric': 'Sesgo', 'label': 'd)'}
    ]

    # Figure setup
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(5.26 * 4, 10), dpi=600)
    fig.subplots_adjust(wspace=-0.55)
    plt.rcParams['font.size'] = '14'

    if title:
        fig.suptitle(title, fontsize=36, y=1.02)

    # Plotting loop
    for i, config in enumerate(plot_configs):
        ax = axs[i]
        metric_name = config['metric']

        # Background
        if r:
            rasterio.plot.show(r, cmap=cmapg, interpolation='none', vmin=9000, vmax=22000, ax=ax)
        df_chile.boundary.plot(color='grey', lw=0.4, ax=ax)

        # Scatter Map
        cmap = cl.LinearSegmentedColormap.from_list(f"{metric_name}_map", config['colors'], N=len(config['colors']))
        norm = cl.BoundaryNorm(config['bins'], cmap.N)

        im = ax.scatter(
            df_plot_data['outlet_camels_lon'], df_plot_data['outlet_camels_lat'],
            c=df_plot_data[metric_name], cmap=cmap, marker='.', norm=norm
        )

        # Zones & Labels
        division_latitudes = [-26.0, -32.0, -38.0, -44.0]
        for lat in division_latitudes:
            ax.axhline(y=lat, color='black', linestyle='--', linewidth=1.2, alpha=0.5)

        if i == 0:
            zone_names = ['Far\nNorth', 'Near\nNorth', 'Central\nChile', 'Southern\nChile', 'Far\nSouth']
            zone_centers = [(-16 + -26)/2, (-26 + -32)/2, (-32 + -38)/2, (-38 + -44)/2, (-44 + -58)/2]

            x_pos = -66.8
            for zone_name, lat_center in zip(zone_names, zone_centers):
                ax.text(x_pos, lat_center, zone_name, fontsize=14, ha='center', va='center', rotation=90)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, spacing='proportional', boundaries=config['bins'], extend=config['extend'])
        cbar.set_ticks(config['bins'])
        cbar.set_ticklabels(config['ticks'])
        cbar.ax.tick_params(labelsize=22)

        # Axes Formatting
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim([-76.5, -65.5])
        ax.set_ylim([-58, -16])
        ax.text(-76, -57.8, config['label'], fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=22)

        if i == 0:
            ax.set_xlabel('Longitude', fontsize=32)
            ax.set_ylabel('Latitude', fontsize=32)
        else:
            ax.tick_params(axis='y')
            ax.set_xlabel('Longitude', fontsize=32)

        # North Arrow
        x_arrow, y_arrow, arrow_length = 0.14, 0.98, 0.05
        ax.annotate('N', xy=(x_arrow, y_arrow), xytext=(x_arrow, y_arrow - arrow_length),
                     arrowprops=dict(facecolor='black', width=4, headwidth=10),
                     ha='center', va='center', fontsize=15, xycoords=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if output_filename:
        plt.savefig(output_filename, dpi=500, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {output_filename}")

    plt.show()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("--- 1. Loading Data (Attributes & Area) ---")

    # Load only Area info (Precipitation removed)
    area_data = load_catchment_area(PATHS['attributes'])

    # Load models
    print("--- Processing Regional Model ---")
    df0 = process_model_data(PATHS['regional_model'], area_data)

    print("--- Processing Local Model ---")
    df1 = process_model_data(PATHS['local_model'], area_data)

    # Filter for Validation Mask (mask == 2)
    df00 = df0[df0['mask'] == 2]
    df11 = df1[df1['mask'] == 2]

    print("--- 2. Computing Metrics ---")

    # Regional (df00)
    metrics_regional = compute_metrics_dataframe(df00)

    # Local (df11)
    metrics_local = compute_metrics_dataframe(df11)

    # Print Statistics (Quality Check)
    print('\nRegional (p0d) Statistics:')
    print('KGE >= 0.595:', metrics_regional[metrics_regional['KGE'] >= 0.595].shape[0])
    print('KGE >= 0.5:', metrics_regional[metrics_regional['KGE'] >= 0.5].shape[0])
    print('KGE <= 0:', metrics_regional[metrics_regional['KGE'] <= 0].shape[0])

    print('\nLocal (p0d) Statistics:')
    print('KGE >= 0.595:', metrics_local[metrics_local['KGE'] >= 0.595].shape[0])
    print('KGE >= 0.5:', metrics_local[metrics_local['KGE'] >= 0.5].shape[0])
    print('KGE <= 0:', metrics_local[metrics_local['KGE'] <= 0].shape[0])

    print("\n--- 3. Generating Maps (Regional Model) ---")

    # Plotting Regional Results as requested in original script
    plot_performance_maps(
        results_df=metrics_regional,
        coords_path=PATHS['coords'],
        hillshade_path=PATHS['hillshade'],
        shapefile_path=PATHS['shapefile'],
        output_filename=PATHS['output_fig']
    )