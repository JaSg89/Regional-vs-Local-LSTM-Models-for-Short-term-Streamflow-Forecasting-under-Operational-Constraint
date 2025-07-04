





# --------------------------------------------------------------------------- #
# Necessary Libraries (based on usage in this script)
# --------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf # Used for printing the version
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

#### #### #### #### #### #### #### ####
####    FOR PLOTTING P0D ONLY    ####
#### #### #### #### #### #### #### ####

# Load hydrological cluster data for Chile
df_hidrocl = pd.read_csv("/content/drive/MyDrive/FONDEF/Variables/HidroCL_clustered2.csv", sep=';')
# Directory path for local training results
dir_path = '/content/drive/MyDrive/FONDEF/Variables/ENTRENAMIENTO_LOCAL/Resultados_train_full'

#############################
########### LOCAL ###########
#############################

# Path to local model results for p0d (prediction day 0)
ruta0 = f'{dir_path}/Model_Local_p0d.csv'
df0_loc =  pd.read_csv(ruta0,float_precision='legacy')
# Scale flow-related columns by 10 
cols_to_scale_loc = ['Qsim', 'Qobs', 'mean', 'std_dev', 'median', 'per_5', 'per_25', 'per_75', 'per_95']
df0_loc[cols_to_scale_loc] = df0_loc[cols_to_scale_loc]*10

###############################
########## REGIONAL ########### 
###############################

# Load global/regional model results for p0d
df0 = pd.read_csv('/content/drive/MyDrive/FONDEF/Variables/LSTM3/Ult_Model_LSTM/Qmean/Qmean3/Qmean_LSTM_p0d.csv')
# Scale flow-related columns by 10 (unit conversion or denormalization)
cols_to_scale_global = ['Qsim', 'Qobs', 'mean', 'std_dev', 'median', 'per_5', 'per_25', 'per_75', 'per_95']
df0[cols_to_scale_global] = df0[cols_to_scale_global]*10


##############################
#### ERA5 and GFS PP ######## (Precipitation Data)
##############################

# Load precipitation data and catchment area
data = pd.read_csv('/content/drive/MyDrive/FONDEF/Variables/DF_LSTM/LSTM11.csv',
                   usecols=['gauge_id', 'date', 'pp_f_gfs_pp_mean_b_none_d1_p0d',
                            'pp_o_era5_pp_mean_b_none_d1_m7d', 'top_s_cam_area_tot_b_none_c_c'])
# Rename 'gauge_id' to 'ID' for consistency
data.rename(columns={'gauge_id': 'ID'}, inplace=True)
# Shift ERA5 observed precipitation 7 days backward (aligning past observation with current forecast)
data['pp_o_era5_pp_mean_b_none_d1_m7d'] = data.groupby('ID')['pp_o_era5_pp_mean_b_none_d1_m7d'].shift(-7)
# Scale precipitation data by 10
listt = ['pp_f_gfs_pp_mean_b_none_d1_p0d', 'pp_o_era5_pp_mean_b_none_d1_m7d']
data[listt] = data[listt]/10

## ADDING PP TO PREDICTIONS ##
# Merge precipitation data with global/regional model results
df0 = pd.merge(df0, data, left_on=['ID', 'date'], right_on=['ID', 'date'], how='inner')
# Merge precipitation data with local model results
df0_loc = pd.merge(df0_loc, data, left_on=['ID', 'date'], right_on=['ID', 'date'], how='inner')


##############################
#### CONVERSION TO M3/S ###### 

# The operation (Area * Value) / 864 likely converts flow from mm/day (if area is in km^2) or similar specific discharge
# back to a volumetric flow rate, or applies some other normalization factor.
# The factor 864 could relate to 86400 seconds/day if area units and flow units are specific.

############# QSIM ############# (Simulated Discharge)
# Apply conversion to simulated discharge for the global/regional model
df0.loc[:, 'Qsim'] = ((df0.loc[:, 'top_s_cam_area_tot_b_none_c_c']) * df0.loc[:, 'Qsim']) / 864
# Apply conversion to simulated discharge for the local model
df0_loc.loc[:, 'Qsim'] = ((df0_loc.loc[:, 'top_s_cam_area_tot_b_none_c_c']) * df0_loc.loc[:, 'Qsim']) / 864

############# QOBS ############# 
# Apply conversion to observed discharge for the global/regional model
df0.loc[:, 'Qobs'] = ((df0.loc[:, 'top_s_cam_area_tot_b_none_c_c']) * df0.loc[:, 'Qobs']) / 864
# Apply conversion to observed discharge for the local model
df0_loc.loc[:, 'Qobs'] = ((df0_loc.loc[:, 'top_s_cam_area_tot_b_none_c_c']) * df0_loc.loc[:, 'Qobs']) / 864


####################################
########## KGE CALCULATION #########
####################################

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
    alpha = sim.std()/obs.std()
    beta = sim.mean()/obs.mean()
    # Calculate Pearson correlation coefficient
    # Ensure inputs are numpy arrays for corrcoef if they are pandas Series
    sim_val = sim.values if isinstance(sim, pd.Series) else sim
    obs_val = obs.values if isinstance(obs, pd.Series) else obs
    corr_coeff_matrix = np.corrcoef(sim_val, obs_val)
    corr_coeff = corr_coeff_matrix[0,1]

    KGE_val = 1 - np.sqrt((corr_coeff-1)**2 + (alpha-1)**2 + (beta-1)**2 )
    return KGE_val, corr_coeff, alpha, beta

def compute_KGE_for_each_ID(df):
    """Calculates KGE for each unique ID in the DataFrame."""
    # Note: This function is defined but not called in the main script flow.
    # Its results (kge_df, corr_df, etc.) are computed locally but not returned or used externally.
    # This is preserved as per the user's request not to change functions.

    # Group by 'ID' and apply the KGE function to each group
    kge_results = df.groupby('ID').apply(lambda x: KGE(x['Qsim'], x['Qobs'])[0])
    corr_results = df.groupby('ID').apply(lambda x: KGE(x['Qsim'], x['Qobs'])[1])
    var_results = df.groupby('ID').apply(lambda x: KGE(x['Qsim'], x['Qobs'])[2])
    sesgo_results = df.groupby('ID').apply(lambda x: KGE(x['Qsim'], x['Qobs'])[3])

    # Create a DataFrame with the results (these are local to the function and not returned)
    kge_df = kge_results.reset_index(name='KGE')
    corr_df = corr_results.reset_index(name='Corr')
    var_df = var_results.reset_index(name='Var')
    sesgo_df = sesgo_results.reset_index(name='Sesgo')

########################################################################
################# HYDROGRAPH WITH ANOMALY ##############################
########################################################################

# Load station metadata (used for getting gauge names)
ID_metadata = pd.read_csv('/content/drive/MyDrive/FONDEF/Variables/HidroCL_clustered2.csv', sep=';')

def plot_streamflow2(df, df_loc, Codes, set=None, start_date=None, precip_y_lim=None, qmean_y_lim=None):
    # Create copies to avoid modifying original DataFrames
    df = df.copy()
    df_loc = df_loc.copy()

    # Unfiltered copies for KGE calculation
    df_unfiltered = df.copy()
    df_loc_unfiltered = df_loc.copy()

    # Filter data based on the 'set' parameter
    if set is None:
        pass  # No filter applied, use the entire DataFrame
    elif isinstance(set, list) and set == [1, 2]: # if set is [1,2]
        df = df[df['mask'].isin(set)]
        df_loc = df_loc[df_loc['mask'].isin(set)]
    elif isinstance(set, list) and set == [1]: # if set is [1]
        df = df[df['mask'].isin(set)]
        df_loc = df_loc[df_loc['mask'].isin(set)]
    elif isinstance(set, list) and set == [2]: # if set is [2]
        df = df[df['mask'].isin(set)]
        df_loc = df_loc[df_loc['mask'].isin(set)]
    elif isinstance(set, list) and set == [0,1, 2]: # if set is [0,1,2]
        df = df[df['mask'].isin(set)]
        df_loc = df_loc[df_loc['mask'].isin(set)]
    else: 
        df = df[df['mask'] == set]
        df_loc = df_loc[df_loc['mask'] == set]

    # Ensure 'Codes' (station IDs) is a list
    if not isinstance(Codes, list):
        Codes = [Codes]

    # Iterate over each station Code
    for Code in Codes:
        # Filter data for the current station
        df_temp = df[df['ID'] == Code]
        df_loc_temp = df_loc[df_loc['ID'] == Code]

        # Convert 'date' column to datetime objects
        df_temp['date'] = pd.to_datetime(df_temp['date'], format='%Y-%m-%d')
        df_loc_temp['date'] = pd.to_datetime(df_loc_temp['date'], format='%Y-%m-%d')

        # Calculate KGE using data where mask == 2 (test set) from unfiltered copies
        df_temp_test = df_unfiltered[(df_unfiltered['ID'] == Code) & (df_unfiltered['mask'] == 2)]
        df_loc_test = df_loc_unfiltered[(df_loc_unfiltered['ID'] == Code) & (df_loc_unfiltered['mask'] == 2)]

        # Calculate KGE for the global/regional model on the test set
        if not df_temp_test.empty and not df_temp_test['Qobs'].isnull().all() and not df_temp_test['Qsim'].isnull().all() and df_temp_test['Qobs'].std() != 0:
             KGE_global, _, _, _ = KGE(df_temp_test['Qsim'], df_temp_test['Qobs'])
        else:
            KGE_global = np.nan

        # Calculate KGE for the local model on the test set
        if not df_loc_test.empty and not df_loc_test['Qobs'].isnull().all() and not df_loc_test['Qsim'].isnull().all() and df_loc_test['Qobs'].std() != 0:
            KGE_local, _, _, _ = KGE(df_loc_test['Qsim'], df_loc_test['Qobs'])
        else:
            KGE_local = np.nan 



        if df_temp.empty or df_loc_temp.empty:
            print(f"ID {Code} has no data for the selected set. Plot will not be generated.")
            continue

        # Check if precipitation column has valid data
        max_precip_val = df_temp[['pp_f_gfs_pp_mean_b_none_d1_p0d']].max().max() # Renamed to avoid conflict
        if pd.isna(max_precip_val) or not np.isfinite(max_precip_val) or max_precip_val <= 0:
            print(f"Warning: No valid precipitation data for ID {Code}. This plot will be skipped.")
            continue

        # Determine start_date for plotting
        current_start_date = start_date # Use a new variable to avoid modifying loop variable
        if current_start_date is not None:
            current_start_date = pd.to_datetime(current_start_date)
            if current_start_date < df_temp['date'].min(): # Ensure start_date is not before available data
                current_start_date = df_temp['date'].min()
        else: # If no start_date provided, use the maximum of minimum dates from both datasets
            current_start_date = max(df_temp['date'].min(), df_loc_temp['date'].min())

        # Determine end_date for plotting (minimum of maximum dates)
        current_end_date = min(df_temp['date'].max(), df_loc_temp['date'].max())

        # Check if date range is valid
        if current_start_date > current_end_date:
            print(f"Warning: Invalid date range for ID {Code}. This plot will be skipped.")
            continue

        # Filter DataFrames by the determined date range
        df_temp = df_temp[(df_temp['date'] >= current_start_date) & (df_temp['date'] <= current_end_date)]
        df_loc_temp = df_loc_temp[(df_loc_temp['date'] >= current_start_date) & (df_loc_temp['date'] <= current_end_date)]

        # Create a complete date range and merge to fill missing dates (for continuous plotting)
        days = pd.date_range(start=current_start_date, end=current_end_date, freq='D')
        df_temp = pd.DataFrame({'date': days}).merge(df_temp, on='date', how='left')
        df_loc_temp = pd.DataFrame({'date': days}).merge(df_loc_temp, on='date', how='left')

        # Check for valid data after date filtering and merging before plotting
        if df_temp['pp_f_gfs_pp_mean_b_none_d1_p0d'].isna().all() or df_temp['Qobs'].isna().all():
            print(f"Warning: No valid data to plot for ID {Code} in the selected date range. Plot will be skipped.")
            continue

        # Create figure with two subplots (anomaly on top, hydrograph below)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
        plt.subplots_adjust(hspace=0) # No space between subplots

        # --- Top Panel: GFS and ERA5 Precipitation Anomaly ---
        # Anomaly = GFS_forecast_precipitation - ERA5_observed_precipitation
        # anomaly > 0 → GFS overestimates precipitation compared to ERA5.
        # anomaly < 0 → GFS underestimates precipitation.
        anomaly = df_temp['pp_f_gfs_pp_mean_b_none_d1_p0d'] - df_temp['pp_o_era5_pp_mean_b_none_d1_m7d']
        anomaly_positive = anomaly.clip(lower=0) # Positive anomalies (overestimation)
        anomaly_negative = anomaly.clip(upper=0) # Negative anomalies (underestimation)

        ax1.bar(df_temp['date'], anomaly_positive, color='red', label='Overestimation', alpha=0.8, width=1)
        ax1.bar(df_temp['date'], anomaly_negative, color='navy', label='Underestimation', alpha=0.7, width=1)

        ax1.set_ylabel('Difference (mm/day)', fontsize=20)
        legend1 = ax1.legend(loc='upper right', fontsize=12, title='GFS - ERA5 Difference')
        legend1.get_title().set_fontsize('13')
        # Set title with station name and code
        station_name_series = ID_metadata[ID_metadata["gauge_id"] == Code]["gauge_name"]
        station_name = station_name_series.values[0] if not station_name_series.empty else f"Unknown Station {Code}"
        title_i = f'Forecast Point:  {station_name} ({Code})'
        ax1.set_title(title_i, fontsize=20)
        ax1.tick_params(axis='x', which='both', length=0) # Hide x-axis ticks for top panel
        ax1.tick_params(axis='y', which='major', labelsize=17)

        # --- Bottom Panel: Hydrograph ---
        ax3 = ax2.twinx() # Create a secondary y-axis for precipitation
        # Plot GFS precipitation as bars on the secondary y-axis
        ax3.bar(df_temp['date'], df_temp['pp_f_gfs_pp_mean_b_none_d1_p0d'], color='navy', width=1, alpha=0.7, label='GFS Precipitation')

        # Set y-limits for precipitation axis
        if precip_y_lim:
            ax3.set_ylim(precip_y_lim)
        else:
            ax3.set_ylim(0, max_precip_val * 1.5 if pd.notna(max_precip_val) and max_precip_val > 0 else 10) 
        ax3.set_ylabel('Mean Precipitation (mm/day)', fontsize=20)
        ax3.invert_yaxis() # Invert precipitation axis (common practice for hydrographs)

        # Check for valid Qobs data before setting y-limits for discharge
        max_qobs_val = df_temp['Qobs'].max() 
        if pd.isna(max_qobs_val) or not np.isfinite(max_qobs_val) or max_qobs_val <= 0:
            print(f"Warning: No valid Qobs data for ID {Code}. Hydrograph discharge scaling might be affected.")


        # Plot observed discharge (Qobs)
        ax2.plot(df_temp['date'], df_temp['Qobs'], 'o', color='black', label='Qobs', markersize=2)
        # Plot simulated discharge from global/regional model (Joint Training)
        ax2.plot(df_temp['date'], df_temp['Qsim'], '-', color='green', label=f'Joint Training (KGE={KGE_global:.2f})')
        # Plot simulated discharge from local model (Individual Training)
        ax2.plot(df_loc_temp['date'], df_loc_temp['Qsim'], '-', color='red', alpha= 0.8, label=f'Individual Training (KGE={KGE_local:.2f})')

        # Set y-limits for discharge axis
        if qmean_y_lim:
            ax2.set_ylim(qmean_y_lim)
        else:
            if pd.isna(max_qobs_val) or not np.isfinite(max_qobs_val) or max_qobs_val <= 0:
                max_qsim_g = df_temp['Qsim'].max()
                max_qsim_l = df_loc_temp['Qsim'].max()
                max_flow_val = np.nanmax([max_qsim_g, max_qsim_l])
                if pd.isna(max_flow_val) or not np.isfinite(max_flow_val) or max_flow_val <= 0:
                    ax2.set_ylim(0, 10) 
                else:
                    ax2.set_ylim(0, max_flow_val * 1.75)
            else:
                ax2.set_ylim(0, max_qobs_val * 1.75)

        ax2.set_ylabel('Qmean (m³/s)', fontsize=20)
        ax2.set_xlabel('Date (Year/Month)', fontsize=20)

        ax2.tick_params(axis='both', which='major', labelsize=17)
        ax3.tick_params(axis='y', which='major', labelsize=17)

        # Combine legends from both discharge (ax2) and precipitation (ax3) axes
        lines_1, labels_1 = ax2.get_legend_handles_labels()
        lines_2, labels_2 = ax3.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left', fontsize=12)

        # Format x-axis date ticks
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout() # Adjust subplot params for a tight layout
        # Fine-tune subplot adjustments
        fig.subplots_adjust(top=0.95, bottom=0.05, left=0.07, right=0.93, hspace=0, wspace=0)
        # plt.savefig(f'/content/drive/MyDrive/FONDEF/Variables/ENTRENAMIENTO_LOCAL/HIDROGRAMAS/Hidrograma2_{Code}.png', dpi=500, bbox_inches='tight', facecolor='white')

        plt.show()

plot_streamflow2(df = df0, df_loc = df0_loc, Codes = [9414001],  set=None, start_date='2020-01-01', precip_y_lim=None, qmean_y_lim=None)