# from pyfmreader import loadfile

# internal
import matplotlib.pyplot as plt 
from nptdms import TdmsFile #from nptdms import tdms  # pip install nptdms
import pandas as pd
import os 
import shutil
# import seaborn as sns
import shutil
import glob
import numpy as np
import time

import re
from datetime import datetime
from pathlib import Path

def checkMapFileNULL(directory, CSVfile=True):
    """
    Checks the map directory for TDMS files that are empty and moves them to an error folder.
    Additionally, processes valid files to extract metadata and generate a CSV log file.
    Args:
        directory (str): The path to the directory containing the map files to be checked.
        CSVfile (bool, optional): A flag indicating whether to generate a CSV log file 
                                  with metadata from the processed files. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing metadata and analysis results for the processed files.
        
    """
    from pyfmreader import loadfile 
    from pyfmreader.ps_nex.parseTDMS import grab_tdms
    # Check if the directory is a valid path
    pattern = os.path.join(directory, 'psnex_map__*')
    # files = [f for f in glob.glob(pattern) if not f.endswith('.zip')]
    files = [f for f in glob.glob(pattern) if not (f.endswith('.zip') or f.endswith('.csv'))]

    # if actual map path was given, use that
    if files == []:
        # if no files were found, check if the directory is a valid path
        files.append(directory)
    for rootdir in files[:]:
        print(f"processing file : {rootdir}")
        # _,all_files,filenames = find_directories_with_file_type(rootdir,'.tdms')
        _, all_files = grab_tdms(rootdir)

        df_point_log = pd.DataFrame()

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     # loaded_files = executor.map(load_single_file, files_to_load)
        #     futures = [executor.submit(load_single_file, filepath) for filepath in files_to_load]
        #     for future in concurrent.futures.as_completed(futures):
        #         loaded_files.append(future.result())
        #         count+=1
        #         progress_callback.emit(count)

        # tp_file = []
        for fp in all_files:
            # print(f'now analyzing {fp}')
            try :
                #loading and updating params for hertz
                file = loadfile(fp)
                filemetadata = file.filemetadata
                #closed_loop = filemetadata['z_closed_loop']
                file_deflection_sensitivity = filemetadata['defl_sens_nmbyV']  # nm/V
                file_spring_constant = filemetadata['spring_const_Nbym']  # N/m
                height_channel = filemetadata['height_channel_key']
                #force_set_point = filemetadata["force_setpoint"]
                deflection_sensitivity = filemetadata['defl_sens_nmbyV'] / 1e9  # m/V
                spring_constant = file_spring_constant
            
                curve_properties = filemetadata['curve_properties']
                tick_time_s = filemetadata['instrument_tick_time_(s)']# 2* 10**-6

                force_curve = file.getcurve(0)
                # Preprocess curve
                force_curve.preprocess_force_curve(deflection_sensitivity, height_channel)
                tdms_file_ps_nex_file = TdmsFile.open(fp)
                tdms_groups = tdms_file_ps_nex_file.groups() 
                tdms_psnex_fc = tdms_groups[0]
                
                deflection = tdms_psnex_fc[height_channel][:]
                total_len = len(deflection)
                stored_arr =[]
                cal_arr = []

                for i, segment in force_curve.get_segments():
                    temp_seg_dict = curve_properties[str(0)][i]
                    seg_i_pt_cal = temp_seg_dict[f"segment_{i}_nb_points_cal"]
                    seg_i_pt_stored = temp_seg_dict[f"segment_{i}_nb_points_(points)"]
                    stored_arr.append(seg_i_pt_stored)
                    cal_arr.append(seg_i_pt_cal)

                    segment_duration = temp_seg_dict[f"segment_{i}_duration_(ticks)"]*tick_time_s
                    if i ==0:
                        set_pt_z_pos_V = temp_seg_dict[f"segment_{i}_Z_position_setpoint_trigger_(V)"]
                    print(f"segment duration {segment_duration}, \n numb of point cal (ticks, dec, sampling rate {seg_i_pt_cal}, num pts stored per segment {seg_i_pt_stored}")


                temp_dict = {'file_path':fp,
                                'total_len':total_len,
                                'total_len_cal_tick':sum(cal_arr),
                                'diff_cal':total_len-sum(cal_arr),
                                'diff_store':total_len-sum(stored_arr),
                                'total_len_stored_point':sum(stored_arr),
                                "store_nbpts_app" :stored_arr[0],
                                "store_nbpts_ret" :stored_arr[1],
                                "cal_nbpts_app" :cal_arr[0],
                                "cal_nbpts_ret" :cal_arr[1],
                                "set_pt_z_pos_V":set_pt_z_pos_V
                                }

                df_temp = pd.DataFrame(temp_dict,index = [0])
                df_point_log = pd.concat([df_point_log,df_temp],ignore_index=True)
                print ('df_point log made')

            except KeyError as e:
                print(f"KeyError encountered: {e}")
                error_folder = os.path.join(rootdir, 'error_file')
                if not os.path.exists(error_folder):
                    os.makedirs(error_folder)
                indexfile = fp+'_index'
                shutil.move(fp, os.path.join(error_folder, os.path.basename(fp))) #move tdms file
                shutil.move(indexfile, os.path.join(error_folder, os.path.basename(indexfile))) #move index file
                print(os.path.basename(fp))            
            except TypeError as e:
                print(f"TypeError encountered: {e}")
                error_folder = os.path.join(rootdir, 'error_file')
                if not os.path.exists(error_folder):
                    os.makedirs(error_folder)
                indexfile = fp+'_index'
                shutil.move(fp, os.path.join(error_folder, os.path.basename(fp)))
                shutil.move(indexfile, os.path.join(error_folder, os.path.basename(indexfile))) #move index file
                print(os.path.basename(fp))
            except ValueError as e:
                print(f"ValueError encountered: {e}")
                error_folder = os.path.join(rootdir, 'error_file')
                if not os.path.exists(error_folder):
                    os.makedirs(error_folder)
                indexfile = fp+'_index'
                shutil.move(fp, os.path.join(error_folder, os.path.basename(fp)))
                shutil.move(indexfile, os.path.join(error_folder, os.path.basename(indexfile))) #move index file
                print(os.path.basename(fp))
            except IndexError as e:
                print(f"IndexError encountered: {e}")
                error_folder = os.path.join(rootdir, 'error_file')
                if not os.path.exists(error_folder):
                    os.makedirs(error_folder)
                indexfile = fp+'_index'
                shutil.move(fp, os.path.join(error_folder, os.path.basename(fp)))
                shutil.move(indexfile, os.path.join(error_folder, os.path.basename(indexfile))) #move index files
                print(os.path.basename(fp))
                    
        results_folder = os.path.join(rootdir, 'results')
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
            if CSVfile:
                df_point_log.to_csv(os.path.join(results_folder, 'df_point_log.csv'), index=False)
        return df_point_log
    

def remove_nan_and_get_indices(input_list):
    nan_indices = [i for i, x in enumerate(input_list) if np.isnan(x)]
    cleaned_list = [x for x in input_list if not np.isnan(x)]
    return cleaned_list, nan_indices


def load_map_file_square_tdms (directory) :
    from pyfmreader import loadfile
    from pyfmreader.ps_nex.parseTDMS import grab_tdms
    # Grab TDMS files 
    _, tdms_files = grab_tdms(directory)
    # Initialize lists to store the parameters
    z_setpoint_trig_v_list = []
    mapping_index_list = []
    map_x_index_list = []
    map_y_index_list = []
    # map_x_pos_v_list = []
    # map_y_pos_v_list = []
    impulse_xy_v = 0.1
    estimate_xy = True

    # Open each force curve in the map and read the Zposition contact. If file is corrupt, NaN will be placed
    for files in tdms_files:
        # start_time = time.time()  # Start timing
        try:
            psnex_file = loadfile(files)  # load the file
            # Get metadata and force curve object
            metadata = psnex_file.filemetadata
            z_setpoint_trig_v = metadata['curve_properties']['0'][0]['segment_0_Z_position_setpoint_trigger_(V)']
            mapping_index = metadata['mapping_index']
            map_x_index = metadata['mapping_position_row']
            map_y_index = metadata['mapping_position_col']
            # map_x_pos_v = metadata['X_cur_position_V']
            # map_y_pos_v = metadata['Y_cur_position_V']
        
            # Z height contact and setpoint
            # z_setpoint_trig_v  = metadata['curve_properties']['0'][0]['segment_0_Z_position_setpoint_trigger_(V)']
            
            # Get sensitivities if not already loaded
            if 'z_stage_sens_nm_V' not in locals():
                z_stage_sens_nm_V = metadata['curve_properties']['0'][0]['z_stage_sensitivity']
            if 'x_sens_nm' not in locals():
                x_sens_nm = metadata['system_X_piezo_sensitivity_(nm/V)']
            if 'y_sens_nm' not in locals():
                y_sens_nm = metadata['system_Y_piezo_sensitivity_(nm/V)']
            if 'x_sens_um' not in locals():
                x_sens_um = x_sens_nm / 1000
            if 'y_sens_um' not in locals():
                y_sens_um = y_sens_nm / 1000
            
            # Append the parameters to the lists
            z_setpoint_trig_v_list.append(float(z_setpoint_trig_v))
            mapping_index_list.append(mapping_index)
            map_x_index_list.append(map_x_index)
            map_y_index_list.append(map_y_index)
            # map_x_pos_v_list.append(map_x_pos_v)
            # map_y_pos_v_list.append(map_y_pos_v)
        except Exception as e:
            print(f"Error loading file {files}: {e}")
            # Append NaN to the lists if there is an error
            z_setpoint_trig_v_list.append(np.nan)
            mapping_index_list.append(np.nan)
            map_x_index_list.append(np.nan)
            map_y_index_list.append(np.nan)
            # map_x_pos_v_list.append(np.nan)
            # map_y_pos_v_list.append(np.nan)
        # end_time = time.time()  # End timing
        # iteration_time = end_time - start_time
        # print(f"Time for iteration: {iteration_time:.4f} seconds")

    z_height_um_list = [z * z_stage_sens_nm_V * 1e-3 for z in z_setpoint_trig_v_list]

    if estimate_xy:
        x_pos_um_list = [x * x_sens_um * impulse_xy_v for x in map_x_index_list]
        y_pos_um_list = [y * y_sens_um * impulse_xy_v for y in map_y_index_list]
    # else:
    #     x_pos_um_list = [x * x_sens_um * impulse_xy_v for x in map_x_pos_v_list]
    #     y_pos_um_list = [y * y_sens_um * impulse_xy_v for y in map_y_pos_v_list]



    estimated_square = np.sqrt(len(tdms_files))
    print (f'# of files analyzed: {len(tdms_files)}, square root is : {estimated_square}')
    map_x_pix = int(estimated_square)
    map_y_pix = int(estimated_square)

    data = {
    'z_height_v': z_setpoint_trig_v_list,
    'curve_index': mapping_index_list,
    'x_index': map_x_index_list,
    'y_index': map_y_index_list,
    # 'x_pos_v': map_x_pos_v_list,
    # 'y_pos_v': map_y_pos_v_list,
    'z_height_um': z_height_um_list,
    # 'Z_height_um_zero': map_test_um_zeroed,
    'x_pos_um': x_pos_um_list,
    'y_pos_um': x_pos_um_list, 
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)


    # Sort the dataframe
    df_sorted = df.sort_values(by='curve_index')
    x_pos_um_sorted = df_sorted['x_pos_um'].values
    y_pos_um_sorted = df_sorted['y_pos_um'].values
    z_height_um_sorted = df_sorted['z_height_um'].values

    # Get final data files
    x_axis = x_pos_um_sorted[:map_x_pix].round(1)
    y_axis = np.reshape(y_pos_um_sorted, (map_x_pix, map_y_pix))[:,0].round(1)
    z_pos_2d = np.reshape(z_height_um_sorted, (map_x_pix, map_y_pix)) 

    # remove the last column from all 

    # print(x_axis)

    # Mirror flip the 2D array along the vertical axis
    z_pos_2d_flipped = np.flip(z_pos_2d, axis=1)

    # zero height to represent the actual height since the height is calculated as the approach 
    map_test_um_zeroed = -z_pos_2d_flipped + z_pos_2d_flipped.max().max()
    df_sorted['Z_height_um_zero'] = map_test_um_zeroed.flatten()
    df_dict = {col: df_sorted[col].to_numpy() for col in df_sorted.columns}

    # need to fix this because the dictionary items have a key for each value....
  

    # # plot heatmap
    # fig, ax = plt.subplots(figsize=(20, 10))  # Adjust the figsize as needed
    # a = sns.heatmap(map_test_um_zeroed, xticklabels=x_axis, yticklabels=y_axis, ax=ax)
    # a.invert_yaxis()

    # a.set_xlabel('X axis (um)')
    # a.set_ylabel('Y axis (um)')
    # # a.set_title(f'Heatmap: {filename}')

    # colorbar = a.collections[0].colorbar
    # colorbar.set_label('Z axis (um)')
    # colorbar.ax.yaxis.label.set_rotation(90)  # Rotate the Z axis label
    # _ = plt.xticks(rotation=45)
    # _ = plt.yticks(rotation=0)

        # Create a dictionary with the parameters




    return df_dict

def load_map_file_square_tdms_v2 (directory) :
    from pyfmreader import loadfile
    from pyfmreader.ps_nex.parseTDMS import grab_tdms
    # Grab TDMS files 
    _, tdms_files = grab_tdms(directory)
    # Initialize lists to store the parameters
    z_setpoint_trig_v_list = []
    mapping_index_list = []
    map_x_index_list = []
    map_y_index_list = []
    filepaths = []
    # map_x_pos_v_list = []
    # map_y_pos_v_list = []
    impulse_xy_v = 0.1
    estimate_xy = True

    # Open each force curve in the map and read the Zposition contact. If file is corrupt, NaN will be placed
    for files in tdms_files:
        # start_time = time.time()  # Start timing
        try:
            psnex_file = loadfile(files)  # load the file
            # Get metadata and force curve object
            metadata = psnex_file.filemetadata
            z_setpoint_trig_v = metadata['curve_properties']['0'][0]['segment_0_Z_position_setpoint_trigger_(V)']
            mapping_index = metadata['mapping_index']
            map_x_index = metadata['mapping_position_row']
            map_y_index = metadata['mapping_position_col']
            # map_x_pos_v = metadata['X_cur_position_V']
            # map_y_pos_v = metadata['Y_cur_position_V']
        
            # Z height contact and setpoint
            # z_setpoint_trig_v  = metadata['curve_properties']['0'][0]['segment_0_Z_position_setpoint_trigger_(V)']
            
            # Get sensitivities if not already loaded
            if 'z_stage_sens_nm_V' not in locals():
                z_stage_sens_nm_V = metadata['curve_properties']['0'][0]['z_stage_sensitivity']
            if 'x_sens_nm' not in locals():
                x_sens_nm = metadata['system_X_piezo_sensitivity_(nm/V)']
            if 'y_sens_nm' not in locals():
                y_sens_nm = metadata['system_Y_piezo_sensitivity_(nm/V)']
            if 'x_sens_um' not in locals():
                x_sens_um = x_sens_nm / 1000
            if 'y_sens_um' not in locals():
                y_sens_um = y_sens_nm / 1000
            
            # Append the parameters to the lists
            z_setpoint_trig_v_list.append(float(z_setpoint_trig_v))
            mapping_index_list.append(mapping_index)
            map_x_index_list.append(map_x_index)
            map_y_index_list.append(map_y_index)
            filepaths.append(files)
            # map_x_pos_v_list.append(map_x_pos_v)
            # map_y_pos_v_list.append(map_y_pos_v)
        except Exception as e:
            print(f"Error loading file {files}: {e}")
            # Append NaN to the lists if there is an error
            z_setpoint_trig_v_list.append(np.nan)
            mapping_index_list.append(np.nan)
            map_x_index_list.append(np.nan)
            map_y_index_list.append(np.nan)
            filepaths.append(files)
            # map_x_pos_v_list.append(np.nan)
            # map_y_pos_v_list.append(np.nan)
        # end_time = time.time()  # End timing
        # iteration_time = end_time - start_time
        # print(f"Time for iteration: {iteration_time:.4f} seconds")

    z_height_um_list = [z * z_stage_sens_nm_V * 1e-3 for z in z_setpoint_trig_v_list]

    if estimate_xy:
        x_pos_um_list = [x * x_sens_um * impulse_xy_v for x in map_x_index_list]
        y_pos_um_list = [y * y_sens_um * impulse_xy_v for y in map_y_index_list]
    # else:
    #     x_pos_um_list = [x * x_sens_um * impulse_xy_v for x in map_x_pos_v_list]
    #     y_pos_um_list = [y * y_sens_um * impulse_xy_v for y in map_y_pos_v_list]



    estimated_square = np.sqrt(len(tdms_files))
    print (f'# of files analyzed: {len(tdms_files)}, square root is : {estimated_square}')
    map_x_pix = int(estimated_square)
    map_y_pix = int(estimated_square)

    data = {
    'filepaths' : filepaths,
    'curve_index': mapping_index_list,
    'x_index': map_x_index_list,
    'y_index': map_y_index_list,
    # 'x_pos_v': map_x_pos_v_list,
    # 'y_pos_v': map_y_pos_v_list,
    'z_height_um': z_height_um_list,
    'z_height_v': z_setpoint_trig_v_list,
    # 'Z_height_um_zero': map_test_um_zeroed,
    'x_pos_um': x_pos_um_list,
    'y_pos_um': y_pos_um_list, 
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)


    # Sort the dataframe
    df_sorted = df.sort_values(by='curve_index')
    x_pos_um_sorted = df_sorted['x_pos_um'].values
    y_pos_um_sorted = df_sorted['y_pos_um'].values
    z_height_um_sorted = df_sorted['z_height_um'].values

    # Get final data files
    x_axis = x_pos_um_sorted[:map_x_pix].round(1)
    y_axis = np.reshape(y_pos_um_sorted, (map_x_pix, map_y_pix))[:,0].round(1)
    z_pos_2d = np.reshape(z_height_um_sorted, (map_x_pix, map_y_pix)) 


    # Mirror flip the 2D array along the vertical axis
    z_pos_2d_flipped = np.flip(z_pos_2d, axis=1)

    # zero height to represent the actual height since the height is calculated as the approach 
    map_test_um_zeroed = -z_pos_2d_flipped + z_pos_2d_flipped.max().max()
    df_sorted['Z_height_um_zero'] = map_test_um_zeroed.flatten()

    

    # df_dict = {col: df_sorted[col].to_numpy() for col in df_sorted.columns}
    
    return df_sorted




def load_map_file_csv (map_path, z_sens_um = 6, 
                       x_sens_um = 5.685, y_sens_um = 3.960, impulse_v = 1,
                        dx_v = 0.14, dy_v = 0.14, flipAxis = True, zmin = None, zmax = None): 
    print(f'Analyzing map: {map_path}')
    map_test = pd.read_csv(map_path,header=None)
   
    map_test = map_test.dropna()
    # convert values
    map_test_um =map_test * z_sens_um

    map_test_um_zeroed = -map_test_um + map_test_um.max().max()
    x_axis = (map_test_um.columns * dx_v * x_sens_um).round(1)
    y_axis = (map_test_um.index * dy_v * y_sens_um).round(1)

    # Define the z-axis range for the colormap
    if zmin is not None :
        z_min = zmin
    else:
        z_min = map_test_um_zeroed.min().min()
    if zmax is not None :
        z_max = zmax
    else:
        z_max = map_test_um_zeroed.max().max()
    
    # Set the range for the colormap
    vmin = z_min
    vmax = z_max

    fig, ax = plt.subplots(figsize=(20, 10))  # Adjust the figsize as needed
    # a = sns.heatmap(map_test_um_zeroed, xticklabels=x_axis, yticklabels=y_axis, ax=ax)
    a = sns.heatmap(map_test_um_zeroed, xticklabels=x_axis.round(2), yticklabels=y_axis.round(2), ax=ax, cmap="YlOrBr", vmin=vmin, vmax=vmax)
    a.invert_yaxis()
    if flipAxis:
        xy_axis = -1
    else:
        xy_axis = 1

    a.set_aspect((y_sens_um / x_sens_um) ** (xy_axis*1))
    print(f'y_sens_um: {y_sens_um}, x_sens_um: {x_sens_um}')
    a.set_xlabel('X axis (um)', fontsize=20)
    a.set_ylabel('Y axis (um)', fontsize=20)
    colorbar = a.collections[0].colorbar
    colorbar.set_label('Z axis (um)', fontsize=18)

    colorbar.ax.yaxis.label.set_rotation(90)  # Rotate the Z axis label
    _ = plt.xticks(ticks=np.arange(0, len(x_axis), 5), labels=x_axis[::5].round(2), rotation=45, fontsize=18)
    _ = plt.yticks(ticks=np.arange(0, len(y_axis), 5), labels=y_axis[::5].round(2), rotation=0, fontsize=18)
    # set font size for Z ticks
    colorbar.ax.tick_params(labelsize=20)
    filename = os.path.basename(map_path)
    a.set_title(f'Heatmap: {filename}', fontsize=20)

    # Show minor ticks
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', length=4, color='black')

    # Set transparent background
    fig.patch.set_alpha(0.0)

    # colorbar = a.collections[0].colorbar
    # colorbar.set_label('Z axis (um)')
    # colorbar.ax.yaxis.label.set_rotation(90)  # Rotate the Z axis label
    # _ = plt.xticks(rotation=45)

    # Limit the y-axis to show only values up to 2.9 µm
    ylim = 2.9
    # ax.set_ylim(0, np.where(y_axis <= ylim)[0][-1] + 1)

    # Limit the x-axis to show only values up to 15.0 µm
    xlim = 15.0
    # ax.set_xlim(0, np.where(x_axis <= xlim)[0][-1] + 1)

    # Show minor ticks
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', length=4, color='black')

    return map_test_um_zeroed, x_axis, y_axis



def compute_time_difference(folder_path: str):
    """
    Compute, then Print time difference between the earliest and latest .tdms files
    in a given folder, based on timestamp in the filename.
    Args:
        folder_path (str): The path to the directory containing the tdms files to check

    Returns:
        None, prints time taken
    """
    folder = Path(folder_path)
    pattern = re.compile(r"(\d{4}\.\d{2}\.\d{2}_\d{2}\.\d{2}\.\d{2}\.\d{2})")
    timestamps = []

    for file in folder.glob("*.tdms"):
        match = pattern.search(file.name)
        if match:
            timestamp_str = match.group(1)
            dt = datetime.strptime(timestamp_str, "%Y.%m.%d_%H.%M.%S.%f")
            timestamps.append((file.name, dt))
        else:
            print(f"Skipped: {file.name} (no timestamp found)")

    if len(timestamps) >= 2:
        timestamps.sort(key=lambda x: x[1])
        t_first = timestamps[0][1]
        t_last = timestamps[-1][1]
        delta = t_last - t_first
        delta_seconds = delta.total_seconds()
        hours, remainder = divmod(delta_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = delta.microseconds // 10000  # Two-digit ms

        print(f"\nTime difference between first and last file HH:MM:SS:MS: "
              f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}:{milliseconds:02}")
        return (int(hours), int(minutes), int(seconds), milliseconds)
    else:
        print("Not enough files to compute a time difference.")
        return None