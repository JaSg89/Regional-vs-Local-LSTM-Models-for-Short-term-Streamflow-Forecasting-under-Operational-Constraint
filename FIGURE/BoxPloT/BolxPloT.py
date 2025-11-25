"""
Comparative Analysis: Local vs Regional Hydrological Models (Boxplots & Map)
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
import matplotlib.cm as cm
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio
import rasterio.plot
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION (PATHS & CONSTANTS)
# ==========================================
PATHS = {
    # Model Results
    'local_model': '/content/drive/MyDrive/FONDEF/Variables/ENTRENAMIENTO_LOCAL/Trained_Models/Qmean_LSTM_p0d_Local.csv',
    'regional_model': '/content/drive/MyDrive/FONDEF/Variables/LSTM3/Ult_Model_LSTM/Qmean/Qmean3/Qmean_LSTM_p0d.csv',
    
    # Metadata & Zones
    'zones_tsne': '/content/drive/MyDrive/FONDEF/Variables/tsne.csv',
    
    # Map Data
    'coords': '/content/drive/MyDrive/FONDEF/CODIGOS/MAPA_COLORES/gauge_and_outlet_coordinates.csv',
    'natural_zones': '/content/drive/MyDrive/FONDEF/CODIGOS/MAPA_COLORES/SHP/zonas_naturales.shp',
    'hillshade': '/content/drive/MyDrive/FONDEF/CODIGOS/MAPA_COLORES/hill.tif',
    'shapefile_chile': '/content/drive/MyDrive/FONDEF/CODIGOS/MAPA_COLORES/SHP/Chile_arg_per_bol.shp',
    
    # Output
    'output_fig': '/content/BOXPLOT_MSEPOND_273_LOSS4_SIN14.png'
}

# Visual Configuration (Preserving original aesthetics)
ZONE_COLORS = {
    'Norte Grande': 'skyblue',
    'Norte Chico': 'blue',
    'Zona Central': 'green',
    'Zona Sur': 'red',
    'Zona Austral': 'orange'
}

ZONE_ORDER = ['Norte Grande', 'Norte Chico', 'Zona Central', 'Zona Sur', 'Zona Austral']

ZONE_LABELS_EN = {
    'Norte Grande': ' Far North',
    'Norte Chico': ' Near North',
    'Zona Central': ' Central\n Chile',
    'Zona Sur': ' Southern\n Chile',
    'Zona Austral': ' Far South'
}

ALPHA_VAL = 0.5

# ==========================================
# 2. DATA PROCESSING FUNCTIONS
# ==========================================

def load_and_preprocess_model(filepath, is_local=False):
    """Loads model CSV, scales data, and filters invalid IDs."""
    df = pd.read_csv(filepath, float_precision='legacy')
    
    # Scale specific columns by 10
    target_cols = ['Qsim', 'Qobs', 'mean', 'std_dev', 'median', 
                   'per_5', 'per_25', 'per_75', 'per_95']
    df[target_cols] = df[target_cols] * 10
    
    # Filter ID
    df = df[df['ID'] != 8380001]
    
    return df

def calculate_kge_components(sim, obs):
    """Calculates KGE and its components."""
    if len(sim) == 0: return np.nan, np.nan, np.nan, np.nan
    
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    corr_coeff = np.corrcoef(sim, obs)[0, 1]
    
    kge = 1 - np.sqrt((corr_coeff - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge, corr_coeff, alpha, beta

def compute_metrics(df):
    """Computes performance metrics for each unique ID."""
    results = []
    for gauge_id, group in df.groupby('ID'):
        kge, r, alpha, beta = calculate_kge_components(group['Qsim'].values, group['Qobs'].values)
        results.append([gauge_id, kge, r, alpha, beta])
    
    return pd.DataFrame(results, columns=['ID', 'KGE', 'Corr', 'Var', 'Sesgo'])

def load_zones(path):
    """Loads the zone classification file."""
    df = pd.read_csv(path, sep=';')
    df = df[['gauge_id', 'Zona']]
    df.rename(columns={'gauge_id': 'ID'}, inplace=True)
    return df

# ==========================================
# 3. PLOTTING FUNCTION
# ==========================================

def create_comparison_figure(
    loc_metrics, reg_metrics, map_data, paths
):
    # Prepare Data Structure for Plotting
    # List of tuples: (Label, Local_DF, Regional_DF, Column_Name)
    metrics_config = [
        ('KGE', loc_metrics, reg_metrics, 'KGE'),
        ('Correlation', loc_metrics, reg_metrics, 'Corr'),
        ('Variance Bias', loc_metrics, reg_metrics, 'Var'),
        ('Mean Bias', loc_metrics, reg_metrics, 'Sesgo')
    ]

    # Initialize Figure
    fig = plt.figure(figsize=(32, 18))
    gs = fig.add_gridspec(4, 2, width_ratios=[6, 1], hspace=0.05, wspace=0.15)
    axes = []

    # --- A. LEFT COLUMN: BOXPLOTS ---
    for row_idx, (metric_name, df_loc, df_reg, col_name) in enumerate(metrics_config):
        ax = fig.add_subplot(gs[row_idx, 0])
        axes.append(ax)

        combined_data = []
        x_labels = []
        box_colors = []

        # 1. Per-Zone Data (Local vs Regional pair)
        for zone in ZONE_ORDER:
            # Local
            val_l = df_loc[df_loc['Zona'] == zone][col_name].dropna().values
            combined_data.append(val_l)
            x_labels.append('L')
            box_colors.append(ZONE_COLORS[zone])

            # Regional
            val_r = df_reg[df_reg['Zona'] == zone][col_name].dropna().values
            combined_data.append(val_r)
            x_labels.append('R')
            box_colors.append(ZONE_COLORS[zone])

        # 2. General Data (All Chile)
        val_l_all = df_loc[col_name].dropna().values
        combined_data.append(val_l_all)
        x_labels.append('L')
        box_colors.append('#999999')

        val_r_all = df_reg[col_name].dropna().values
        combined_data.append(val_r_all)
        x_labels.append('R')
        box_colors.append('#999999')

        # 3. Draw Boxplot
        flier_props = dict(marker='o', markerfacecolor='gray', markersize=5, linestyle='none', alpha=0.5)
        bp = ax.boxplot(combined_data, patch_artist=True, showfliers=False, widths=0.6, flierprops=flier_props)

        # Style boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(ALPHA_VAL)

        for median in bp['medians']:
            median.set(color='black', linewidth=2.5)

        # 4. Axes & Labels
        ax.set_ylabel(metric_name, fontsize=32)
        ax.tick_params(axis='y', labelsize=28)

        # Vertical separators
        for i in range(1, len(ZONE_ORDER) + 1):
            ax.axvline(x=i * 2 + 0.5, color='black', linestyle='-', linewidth=1.5)

        # 5. Special Legend (Only on first plot)
        if metric_name == 'KGE':
            all_chile_patch = Patch(facecolor='#999999', edgecolor='black', alpha=ALPHA_VAL, label='All Chile')
            ax.legend(handles=[all_chile_patch], loc='lower center', 
                      bbox_to_anchor=(0.91, 0.03), fontsize=22, 
                      frameon=True, fancybox=True, shadow=True)

        # 6. Reference Lines & Limits
        if metric_name == 'KGE':
            ax.axhline(y=0.6, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.set_ylim(-1.2, 1.02)
        elif metric_name == 'Correlation':
            ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.set_ylim(-0.6, 1.01)
        elif metric_name in ['Variance Bias', 'Mean Bias']:
            ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
            limit_top = 2.02 if metric_name == 'Variance Bias' else 2.1
            ax.set_ylim(-0.1, limit_top)

        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Hide x labels for upper plots
        if row_idx < len(metrics_config) - 1:
            ax.tick_params(axis='x', labelbottom=False, length=0)
        else:
            ax.set_xticks(range(1, len(x_labels) + 1))
            ax.set_xticklabels(x_labels, fontsize=28)

    # --- B. RIGHT COLUMN: MAP ---
    ax_map = fig.add_subplot(gs[:, 1])

    # Unpack map data
    coords, zn, hillshade, df_chile = map_data
    
    # Plotting Map Layers
    cmap_hill = cl.LinearSegmentedColormap.from_list("", ["#CECECE", "#FFFFFF"])
    rasterio.plot.show(hillshade, cmap=cmap_hill, interpolation='none', vmin=9000, vmax=22000, ax=ax_map)
    df_chile.plot(color='lightgray', ax=ax_map)
    zn.plot(color=zn['col'], ax=ax_map, alpha=ALPHA_VAL)

    ax_map.set_aspect('auto')

    # Map Configuration
    ax_map.set_xlim([-76.5, -65.5])
    ax_map.set_ylim([-58, -16])
    ax_map.set_xlabel('Longitude', fontsize=28)
    ax_map.set_ylabel('Latitude', fontsize=28)
    ax_map.tick_params(axis='both', labelsize=28)

    # North Arrow
    ax_map.annotate('N', xy=(0.14, 0.98), xytext=(0.14, 0.93),
                    arrowprops=dict(facecolor='black', width=4, headwidth=10),
                    ha='center', va='center', fontsize=28, xycoords='axes fraction')

    # Colorbar
    divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("right", size="17%", pad=0.5)

    cmap_opaque = cl.ListedColormap(list(ZONE_COLORS.values()))
    norm = cl.BoundaryNorm(boundaries=np.arange(-0.5, len(ZONE_COLORS)), ncolors=len(ZONE_COLORS))

    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_opaque), cax=cax)
    cbar.solids.set_alpha(ALPHA_VAL)
    cbar.solids.set_edgecolor("face")
    cbar.set_ticks(np.arange(len(ZONE_COLORS)))
    cbar.set_ticklabels([ZONE_LABELS_EN[z] for z in ZONE_ORDER])
    cbar.ax.tick_params(labelsize=28)
    cbar.ax.invert_yaxis()

    # Save
    if paths['output_fig']:
        plt.savefig(paths['output_fig'], dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {paths['output_fig']}")

    plt.show()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("--- Loading Models ---")
    df0_loc = load_and_preprocess_model(PATHS['local_model'], is_local=True)
    df0_reg = load_and_preprocess_model(PATHS['regional_model'], is_local=False)

    # Filter for Validation set
    df0_loc = df0_loc[df0_loc['mask'] == 2]
    df0_reg = df0_reg[df0_reg['mask'] == 2]

    print("--- Computing Metrics ---")
    metrics_loc = compute_metrics(df0_loc)
    metrics_reg = compute_metrics(df0_reg)

    # Print Stats (Quality Check)
    print(f"Local KGE >= 0.595: {metrics_loc[metrics_loc['KGE'] >= 0.595].shape[0]}")
    print(f"Regional KGE >= 0.595: {metrics_reg[metrics_reg['KGE'] >= 0.595].shape[0]}")

    print("--- Integrating Zones ---")
    df_zones = load_zones(PATHS['zones_tsne'])

    # Merge metrics with zones
    # Helper to merge all metrics with zone info
    def merge_zones(metrics_df, zones_df):
        return pd.merge(metrics_df, zones_df, on='ID')

    loc_merged = merge_zones(metrics_loc, df_zones)
    reg_merged = merge_zones(metrics_reg, df_zones)

    print("--- Preparing Map Data ---")
    coords = pd.read_csv(PATHS['coords'])
    zn = gpd.read_file(PATHS['natural_zones'])
    hill = rasterio.open(PATHS['hillshade'])
    df_chile = gpd.read_file(PATHS['shapefile_chile'])

    # Pre-process map shapes
    zn['col'] = [ZONE_COLORS[z] for z in zn['Etiqueta']]
    zn = zn.to_crs('EPSG:4326')

    print("--- Generating Figure ---")
    create_comparison_figure(
        loc_metrics=loc_merged,
        reg_metrics=reg_merged,
        map_data=(coords, zn, hill, df_chile),
        paths=PATHS
    )
    print("Done.")