# --------------------------------------------------------------------------- #
# Necessary Libraries
# --------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------- #
# Initial Data Loading and Preprocessing
# --------------------------------------------------------------------------- #
# Load the main dataset from a CSV file
# Specific columns are selected: 'gauge_id', 'date', and precipitation/area variables.
data = pd.read_csv('/Users/saave/Desktop/FONDEF_DATA/Entrenamiento_LOCAL/LSTM_LOCAL2.CSV',
                   usecols=['gauge_id', 'date', 'top_s_cam_area_tot_b_none_c_c'])

# Rename the 'gauge_id' column to 'ID' for clarity
data.rename(columns={'gauge_id': 'ID'}, inplace=True)

# --------------------------------------------------------------------------- #
# Loading Model Results (Regional and Local)
# --------------------------------------------------------------------------- #
# Define paths to directories where model results are stored
dir_path = '/Users/saave/Desktop/FONDEF_DATA/Entrenamiento_Regional'
dir_path_loc = '/Users/saave/Desktop/FONDEF_DATA/Entrenamiento_LOCAL/PRUEBAS/Models'

# Load results from the regional model (Qmean_LSTM_p0d)
ruta0 = f'{dir_path}/Qmean_LSTM_p0d.csv'
df0 = pd.read_csv(ruta0, float_precision='legacy')
# Scale relevant flow and metric columns by 10
columns_to_scale = ['Qsim', 'Qobs', 'mean', 'std_dev', 'median', 'per_5', 'per_25', 'per_75', 'per_95', 'crps']
df0[columns_to_scale] = df0[columns_to_scale] * 10

# Load results from the local model (Model_Local_p0d)
ruta1 = f'{dir_path_loc}/Qmean_LSTM_p0d_Local.csv'
df1 = pd.read_csv(ruta1, float_precision='legacy')
# Scale relevant flow and metric columns by 10
df1[columns_to_scale] = df1[columns_to_scale] * 100

# --------------------------------------------------------------------------- #
# Merging Model Results with Precipitation and Area Data
# --------------------------------------------------------------------------- #
# Merge regional model results (df0) with the 'data' DataFrame using 'ID' and 'date'
df0 = pd.merge(df0, data, left_on=['ID', 'date'], right_on=['ID', 'date'], how='inner')
# Merge local model results (df1) with the 'data' DataFrame using 'ID' and 'date'
df1 = pd.merge(df1, data, left_on=['ID', 'date'], right_on=['ID', 'date'], how='inner')

# --------------------------------------------------------------------------- #
# Flow Normalization (Possibly to Specific Discharge in mm/day)
# --------------------------------------------------------------------------- #
# Columns to modify (flows and related percentiles)
columns_to_modify = ['Qsim', 'Qobs', 'mean', 'std_dev', 'median', 'per_5', 'per_25', 'per_75', 'per_95', 'crps']
dataframes = [df0, df1] # List of DataFrames to process

# Iterate over each DataFrame (regional and local)
for df in dataframes:
    # Iterate over each column to modify
    for column in columns_to_modify:
        # Apply the transformation: (Value * Catchment_Area) / 864
        # This transformation might convert m³/s to mm/day if area is in km² and factors are adjusted.
        # The factor 864 suggests a relation to seconds in a day (86400)
        # and a scaling factor or area unit conversion.
        df[column] = (df[column] * df['top_s_cam_area_tot_b_none_c_c']) / 864

# --------------------------------------------------------------------------- #
# Definition of Kling-Gupta Efficiency (KGE) Metric
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
        KGE, r, alpha, beta
    """
    # Calculate alpha (variability ratio)
    alpha = sim.std() / obs.std()
    # Calculate beta (bias ratio)
    beta = sim.mean() / obs.mean()
    # Calculate Pearson correlation coefficient
    corr_coeff_matrix = np.corrcoef(sim, obs)
    corr_coeff = corr_coeff_matrix[0, 1]

    # Calculate KGE
    KGE_val = 1 - np.sqrt((corr_coeff - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return KGE_val, corr_coeff, alpha, beta

# --------------------------------------------------------------------------- #
# Calculation of KGE per Station (ID)
# --------------------------------------------------------------------------- #
def compute_KGE_for_each_ID(df):
    """
    Calculates KGE and its components (Correlation, Variability, Bias)
    for each unique 'ID' in the provided DataFrame.
    """
    # Group by 'ID' and apply the KGE function to each group to get the KGE value
    kge_results = df.groupby('ID').apply(lambda g: KGE(g['Qsim'], g['Qobs'])[0])
    # Get correlation component (r)
    corr_results = df.groupby('ID').apply(lambda g: KGE(g['Qsim'], g['Qobs'])[1])
    # Get variability component (alpha)
    var_results = df.groupby('ID').apply(lambda g: KGE(g['Qsim'], g['Qobs'])[2])
    # Get bias component (beta)
    sesgo_results = df.groupby('ID').apply(lambda g: KGE(g['Qsim'], g['Qobs'])[3])

    # Create DataFrames with the results
    kge_df = kge_results.reset_index(name='KGE')
    corr_df = corr_results.reset_index(name='Corr')    # Correlation (r)
    var_df = var_results.reset_index(name='Var')      # Variability (alpha)
    sesgo_df = sesgo_results.reset_index(name='Sesgo')  # Bias (beta)

    return kge_df, corr_df, var_df, sesgo_df

# Filter DataFrames by 'mask == 2'
df00 = df0[df0['mask'] == 2] # For the regional model
df10 = df1[df1['mask'] == 2] # For the local model

# Calculate KGE and its components for the regional model
kge_df0, corr_df0, var_df0, sesgo_df0 = compute_KGE_for_each_ID(df00)
# Calculate KGE and its components for the local model
kge_df1, corr_df1, var_df1, sesgo_df1 = compute_KGE_for_each_ID(df10)

# --------------------------------------------------------------------------- #
# Printing KGE Summary Statistics
# --------------------------------------------------------------------------- #
print('Results for Regional Model (based on df0, Qmean_LSTM_p0d):')
print(f"Stations with KGE >= 0.595: {kge_df0[kge_df0['KGE'] >= 0.595].shape[0]}")
print(f"Stations with KGE >= 0.5: {kge_df0[kge_df0['KGE'] >= 0.5].shape[0]}")
print(f"Stations with KGE <= 0: {kge_df0[kge_df0['KGE'] <= 0].shape[0]}")

print('\nResults for Local Model (based on df1, Model_Local_p0d):')
print(f"Stations with KGE >= 0.595: {kge_df1[kge_df1['KGE'] >= 0.595].shape[0]}")
print(f"Stations with KGE >= 0.5: {kge_df1[kge_df1['KGE'] >= 0.5].shape[0]}")
print(f"Stations with KGE <= 0: {kge_df1[kge_df1['KGE'] <= 0].shape[0]}")

# --------------------------------------------------------------------------- #
# Data Preparation for ECDF Plots (Empirical Cumulative Distribution Function)
# --------------------------------------------------------------------------- #
# Label each dataset for the plot legend
corr_df0['Dataset']  = 'Regional'
corr_df1['Dataset']  = 'Local'
var_df0['Dataset']   = 'Regional'
var_df1['Dataset']   = 'Local'
sesgo_df0['Dataset'] = 'Regional'
sesgo_df1['Dataset'] = 'Local'
kge_df0['Dataset']   = 'Regional'
kge_df1['Dataset']   = 'Local'

# Desired order for legend items in the plot
training_type_order = ['Regional', 'Local']

# --------------------------------------------------------------------------- #
# Creation of ECDF Plots for KGE and its Components
# --------------------------------------------------------------------------- #
# Create a figure with 4 subplots (1 row, 4 columns)
fig, axes = plt.subplots(1, 4, figsize=(10, 8), dpi=500)

# Define parameters for each of the 4 plot panels
# (DataFrame_Regional, DataFrame_Local, Metric_Column_Name, X_Axis_Label, Subplot_Title, Use_Log_X_Scale)
datasets_for_plotting = [
    (kge_df0,   kge_df1,   'KGE',   'KGE Value',            'a)', False),
    (corr_df0,  corr_df1,  'Corr',  'Correlation Value (r)', 'b)', False),
    (var_df0,   var_df1,   'Var',   'Variability Ratio (α)', 'c)', True),
    (sesgo_df0, sesgo_df1, 'Sesgo', 'Bias Ratio (β)',     'd)', True),
]

# Iterate over each subplot (ax) and the corresponding panel configuration
for i, (ax, (df_reg, df_loc, column_metric, xlabel_text, title_text, use_log_scale)) in enumerate(zip(axes, datasets_for_plotting)):
    # Concatenate regional and local DataFrames for the current metric
    all_metric_df = pd.concat([df_reg, df_loc])

    # Plot ECDF for each training type ('Regional', 'Local')
    for training_type in training_type_order:
        group = all_metric_df[all_metric_df['Dataset'] == training_type]
        sns.ecdfplot(data=group, x=column_metric, ax=ax, label=training_type, linewidth=2.5)

    # Set axis labels and subplot title
    ax.set_xlabel(xlabel_text, fontsize=30)
    if i == 0: # Only for the first subplot (leftmost)
        ax.set_ylabel('Cumulative Probability', fontsize=32)
    else:
        ax.set_ylabel('') # Do not show Y label for other subplots

    ax.set_title(title_text, fontsize=34)

    # Adjust axis limits
    ax.set_ylim(-0.005, 1.005) # Y-limit for cumulative probability
    if not use_log_scale:
        ax.set_xlim(-0.005, 1.005) # X-limit for KGE and Correlation (r)
    else: # For Variability (α) and Bias (β)
        ax.set_xscale('log') # Use logarithmic scale on X-axis
        ax.set_xlim(1e-1, 10) # X-limit from 0.1 to 10

    # Add "ideal curve" in dashed red
    # A perfect model would have KGE=1, r=1, α=1, β=1.
    # The ECDF for a perfect value would be a step function at 1.
    if use_log_scale: # For Variability (α) and Bias (β)
        xmin_ax, xmax_ax = ax.get_xlim()
        ax.hlines(0, xmin_ax, 1, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Ideal curve')
        ax.vlines(1, 0, 1, color='red', linestyle='--', linewidth=2, alpha=0.8, label='_nolegend_')
        ax.hlines(1, 1, xmax_ax, color='red', linestyle='--', linewidth=2, alpha=0.8, label='_nolegend_')
    else: # For KGE and Correlation (r)
        # Horizontal line from (-0.1, 0) to (1, 0)
        ax.plot([-0.1, 1], [0, 0], color='red', linestyle='--', linewidth=2, alpha=0.8, label='Ideal curve')
        # Vertical line from (1, 0) to (1, 1)
        ax.plot([1, 1], [0, 1], color='red', linestyle='--', linewidth=2, alpha=0.8, label='_nolegend_')

    # Configure legend, grid, and tick mark size
    ax.legend(title='Training Type', loc='best', fontsize=26, title_fontsize=23)
    ax.grid(True, which='both', linestyle='-', linewidth=1, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=28)

# Adjust layout to prevent overlaps
plt.tight_layout()

# Save the figure to a PNG file
#plt.savefig('/content/CDF_Qmean_G-L.png', dpi=500, bbox_inches='tight', facecolor='white')

# Display the figure
plt.show()