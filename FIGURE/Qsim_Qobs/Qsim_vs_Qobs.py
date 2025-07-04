
# --------------------------------------------------------------------------- #
# Necessary Libraries
# --------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors # For LogNorm, BoundaryNorm
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------- #
# KGE (Kling-Gupta Efficiency) and NSE (Nash-Sutcliffe Efficiency) Functions
# --------------------------------------------------------------------------- #
def KGE(sim, obs):
    """
    Calculates the Kling-Gupta Efficiency (KGE) and its components.
    Returns: KGE_val, corr_coeff, alpha, beta
    """
    sim_val = sim.values if isinstance(sim, pd.Series) else sim
    obs_val = obs.values if isinstance(obs, pd.Series) else obs

    if np.std(obs_val) == 0 or pd.isna(np.std(obs_val)) or pd.isna(np.std(sim_val)):
        alpha = np.nan
    else:
        alpha = np.std(sim_val) / np.std(obs_val)

    if np.mean(obs_val) == 0 or pd.isna(np.mean(obs_val)) or pd.isna(np.mean(sim_val)):
        beta = np.nan
    else:
        beta = np.mean(sim_val) / np.mean(obs_val)
    
    if np.all(np.isnan(sim_val)) or np.all(np.isnan(obs_val)) or len(sim_val) < 2 or len(obs_val) < 2:
        corr_coeff = np.nan
    else:
        sim_val_no_nan = np.nan_to_num(sim_val, nan=np.nanmean(sim_val))
        obs_val_no_nan = np.nan_to_num(obs_val, nan=np.nanmean(obs_val))
        if np.all(sim_val_no_nan == sim_val_no_nan[0]) or np.all(obs_val_no_nan == obs_val_no_nan[0]):
             corr_coeff = np.nan
        else:
            corr_coeff_matrix = np.corrcoef(sim_val_no_nan, obs_val_no_nan)
            corr_coeff = corr_coeff_matrix[0, 1]

    if pd.isna(corr_coeff) or pd.isna(alpha) or pd.isna(beta):
        KGE_val = np.nan
    else:
        KGE_val = 1 - np.sqrt((corr_coeff - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return KGE_val, corr_coeff, alpha, beta

def compute_KGE_for_each_ID(df):
    """Calculates KGE and its components for each unique ID in the DataFrame."""
    kge_results = df.groupby('ID').apply(lambda x: KGE(x['Qsim'].dropna(), x['Qobs'].dropna())[0])
    corr_results = df.groupby('ID').apply(lambda x: KGE(x['Qsim'].dropna(), x['Qobs'].dropna())[1])
    var_results = df.groupby('ID').apply(lambda x: KGE(x['Qsim'].dropna(), x['Qobs'].dropna())[2])
    sesgo_results = df.groupby('ID').apply(lambda x: KGE(x['Qsim'].dropna(), x['Qobs'].dropna())[3])

    kge_df = kge_results.reset_index(name='KGE')
    corr_df = corr_results.reset_index(name='Corr')
    var_df = var_results.reset_index(name='Var')
    sesgo_df = sesgo_results.reset_index(name='Sesgo')
    return kge_df, corr_df, var_df, sesgo_df

# --------------------------------------------------------------------------- #
# Data Loading and Processing Function
# --------------------------------------------------------------------------- #
def load_and_process_data(csv_path, area_data_df):
    """Loads flow data, scales it, merges with area, and converts units."""
    df = pd.read_csv(csv_path)
    cols_to_scale = ['Qsim', 'Qobs', 'mean', 'std_dev', 'median', 'per_5', 'per_25', 'per_75', 'per_95']
    # Ensure all columns exist before scaling
    for col in cols_to_scale:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in {csv_path}. Skipping scaling for this column.")
    
    existing_cols_to_scale = [col for col in cols_to_scale if col in df.columns]
    df[existing_cols_to_scale] = df[existing_cols_to_scale] * 10

    df = pd.merge(df, area_data_df, on=['ID', 'date'], how='inner')
    # Check if 'top_s_cam_area_tot_b_none_c_c' column exists after merge
    if 'top_s_cam_area_tot_b_none_c_c' in df.columns:
        df['Qsim'] = (df['top_s_cam_area_tot_b_none_c_c'] * df['Qsim']) / 864
        df['Qobs'] = (df['top_s_cam_area_tot_b_none_c_c'] * df['Qobs']) / 864
    else:
        print(f"Warning: Area column 'top_s_cam_area_tot_b_none_c_c' not found after merging for {csv_path}. Unit conversion for Qsim/Qobs might be incorrect.")
    return df

# --------------------------------------------------------------------------- #
# Load Area Data (Common for Qmean and Qmax)
# --------------------------------------------------------------------------- #
area_data_path = '/content/drive/MyDrive/FONDEF/Variables/DF_LSTM/LSTM11.csv'
try:
    area_data = pd.read_csv(area_data_path, usecols=['gauge_id', 'date', 'top_s_cam_area_tot_b_none_c_c'])
    area_data.rename(columns={'gauge_id': 'ID'}, inplace=True)
except FileNotFoundError:
    print(f"Error: Area data file not found at {area_data_path}. Cannot proceed with unit conversions.")
    exit()


# --------------------------------------------------------------------------- #
# Process Qmean Data
# --------------------------------------------------------------------------- #
print("--- Processing Qmean Data ---")
qmean_csv_path = '/content/drive/MyDrive/FONDEF/Variables/LSTM3/Ult_Model_LSTM/Qmean/Qmean3/Qmean_LSTM_p0d.csv'
try:
    df_qmean = load_and_process_data(qmean_csv_path, area_data)

    # Filter for test set (mask == 2)
    df00_qmean = df_qmean[df_qmean['mask'] == 2].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Calculate KGE for all test basins for Qmean
    kge_df_qmean, _, _, _ = compute_KGE_for_each_ID(df00_qmean)
    print('Qmean - p0d Test Set Statistics:')
    print(f"  Stations with KGE >= 0.6: {kge_df_qmean[kge_df_qmean['KGE'] >= 0.595].shape[0]}")
    print(f"  Stations with KGE >= 0.5: {kge_df_qmean[kge_df_qmean['KGE'] >= 0.495].shape[0]}")
    print(f"  Stations with KGE >= 0:     {kge_df_qmean[kge_df_qmean['KGE'] >= 0].shape[0]}")
    print(f"  Stations with KGE < 0:      {kge_df_qmean[kge_df_qmean['KGE'] < 0].shape[0]}")

    # Filter for test basins with KGE >= 0.595 for Qmean
    lista_qmean_high_kge = kge_df_qmean[kge_df_qmean['KGE'] >= 0.595]['ID'].tolist()
    df06_qmean = df00_qmean[df00_qmean['ID'].isin(lista_qmean_high_kge)].copy()
    qmean_data_available = True
except FileNotFoundError:
    print(f"Error: Qmean data file not found at {qmean_csv_path}. Skipping Qmean plots.")
    qmean_data_available = False
    df00_qmean, df06_qmean = pd.DataFrame(), pd.DataFrame() # Empty DFs

# --------------------------------------------------------------------------- #
# Process Qmax Data
# --------------------------------------------------------------------------- #
print("\n--- Processing Qmax Data ---")
qmax_csv_path = '/content/drive/MyDrive/FONDEF/Variables/LSTM3/Ult_Model_LSTM/Qmax/Qmax2/Qmax_LSTM_p0d.csv'
try:
    df_qmax = load_and_process_data(qmax_csv_path, area_data)

    # Filter for test set (mask == 2)
    df00_qmax = df_qmax[df_qmax['mask'] == 2].copy()

    # Calculate KGE for all test basins for Qmax
    kge_df_qmax, _, _, _ = compute_KGE_for_each_ID(df00_qmax)
    print('Qmax - p0d Test Set Statistics:')
    print(f"  Stations with KGE >= 0.6: {kge_df_qmax[kge_df_qmax['KGE'] >= 0.595].shape[0]}")
    print(f"  Stations with KGE >= 0.5: {kge_df_qmax[kge_df_qmax['KGE'] >= 0.495].shape[0]}")
    print(f"  Stations with KGE >= 0:     {kge_df_qmax[kge_df_qmax['KGE'] >= 0].shape[0]}")
    print(f"  Stations with KGE < 0:      {kge_df_qmax[kge_df_qmax['KGE'] < 0].shape[0]}")

    # Filter for test basins with KGE >= 0.595 for Qmax
    lista_qmax_high_kge = kge_df_qmax[kge_df_qmax['KGE'] >= 0.595]['ID'].tolist()
    df06_qmax = df00_qmax[df00_qmax['ID'].isin(lista_qmax_high_kge)].copy()
    # Recalculate KGE for this subset (optional)
    # kge_df06_qmax, _, _, _ = compute_KGE_for_each_ID(df06_qmax)
    # print(f"Qmax - p0d Test Set (KGE >= 0.595) - Number of stations: {df06_qmax['ID'].nunique()}")
    qmax_data_available = True
except FileNotFoundError:
    print(f"Error: Qmax data file not found at {qmax_csv_path}. Skipping Qmax plots.")
    qmax_data_available = False
    df00_qmax, df06_qmax = pd.DataFrame(), pd.DataFrame() # Empty DFs

# --------------------------------------------------------------------------- #
# Plotting Function for Histograms
# --------------------------------------------------------------------------- #

def plot_histograms(cmap_name, df_to_plot, fig_title, ax):
    """Plots a 2D histogram of log_Qobs vs log_Qsim."""
    if df_to_plot.empty:
        print(f"Skipping plot for '{fig_title}' due to empty DataFrame.")
        ax.text(0.5, 0.5, "Data not available", ha='center', va='center', fontsize=15)
        ax.set_title(fig_title, fontsize=25)
        return

    df_plot_copy = df_to_plot.copy()
    # Replace zeros and negatives in Qobs and Qsim to avoid log10 issues
    # Using a small positive floor for log transformation
    epsilon = 10**-10
    df_plot_copy.loc[df_plot_copy['Qobs'] <= 0, 'Qobs'] = epsilon
    df_plot_copy.loc[df_plot_copy['Qsim'] <= 0, 'Qsim'] = epsilon

    # Compute log-transformed columns
    df_plot_copy['log_Qobs'] = np.log10(df_plot_copy['Qobs'])
    df_plot_copy['log_Qsim'] = np.log10(df_plot_copy['Qsim'])

    # Data for histogram
    x = df_plot_copy['log_Qobs']
    y = df_plot_copy['log_Qsim']
    # Define bins appropriate for the log-transformed data range
    bins = [np.linspace(-2.5, 5, 200), np.linspace(-5, 5, 200)] 

    try:
        # Compute 2D histogram
        hist, xedges, yedges = np.histogram2d(x.dropna(), y.dropna(), bins=bins) # dropna before histogram

        # Plot heatmap
        im = ax.pcolormesh(xedges, yedges, hist.T, cmap=cmap_name,
                           norm=matplotlib.colors.LogNorm(vmin=1, vmax=10**3))

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax, pad=0.02, aspect=15)
        cbar.ax.tick_params(labelsize=25)
        cbar.set_label('Frequency', fontsize=27)

        # Add 1:1 line
        ax.plot([-4, 5], [-4, 5], color='blue', linestyle='--', label='1:1 Line') # Adjusted 1:1 line to match bins

        # X-axis ticks and labels
        ticks_x = np.arange(-3, 6, 2)  # Adjusted ticks
        ax.set_xticks(ticks_x)
        ax.set_xticklabels([r'$10^{%d}$' % tick for tick in ticks_x], fontsize=25)

        # Y-axis ticks and labels
        ticks_y = np.arange(-5, 6, 2)  # Adjusted ticks
        ax.set_yticks(ticks_y)
        ax.set_yticklabels([r'$10^{%d}$' % tick for tick in ticks_y], fontsize=25)

        # Set axis limits
        ax.set_xlim([-3, 5]) # Adjusted limits
        ax.set_ylim([-5, 5]) # Adjusted limits

        # Configure axis labels and title
        ax.set_xlabel(r'Qobs [m³/s]', fontsize=25)
        ax.set_ylabel(r'Qsim [m³/s]', fontsize=25)
        ax.set_title(fig_title, fontsize=25)
        ax.tick_params(axis='both', labelsize=20)

        # Add legend
        ax.legend(loc='lower right', fontsize=20)
        ax.grid()

    except Exception as e:
        print(f"Skipping cmap '{cmap_name}' for '{fig_title}' due to error: {e}")
        ax.text(0.5, 0.5, "Error during plotting", ha='center', va='center', fontsize=15)
        ax.set_title(fig_title, fontsize=25)

# --------------------------------------------------------------------------- #
# Generate and Save Histograms
# --------------------------------------------------------------------------- #

# Plot histograms for Qmean
if qmean_data_available:
    print("\n--- Generating Qmean Histograms ---")
    fig_qmean, axes_qmean = plt.subplots(1, 2, figsize=(25, 7)) # Increased figsize
    plot_histograms('jet', df00_qmean, 'a) Qmean', axes_qmean[0])
    plot_histograms('jet', df06_qmean, 'b) Qmean (KGE≥0.595)', axes_qmean[1])
    plt.tight_layout(pad=3.0) # Add padding
    #plt.savefig('/content/HIST_QMEAN_LOG.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
else:
    print("Skipping Qmean histogram generation as data was not available.")

# Plot histograms for Qmax
if qmax_data_available:
    print("\n--- Generating Qmax Histograms ---")
    fig_qmax, axes_qmax = plt.subplots(1, 2, figsize=(25, 7)) # Increased figsize
    plot_histograms('jet', df00_qmax, 'c) Qmax', axes_qmax[0])
    plot_histograms('jet', df06_qmax, 'd) Qmax (KGE≥0.6)', axes_qmax[1])
    plt.tight_layout(pad=3.0) # Add padding
    #plt.savefig('/content/HIST_QMAX_LOG.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
else:
    print("Skipping Qmax histogram generation as data was not available.")

print("\nScript finished.")