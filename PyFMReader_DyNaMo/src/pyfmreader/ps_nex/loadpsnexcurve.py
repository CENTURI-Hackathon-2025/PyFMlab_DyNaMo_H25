#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 18:07:32 2024

@author: yogehs
"""
# File containing the loadPSNEXcurve function,
# used to load single force curves from JPK files.

from struct import unpack
from itertools import groupby
import numpy as np
from nptdms import TdmsFile

from ..utils.forcecurve import ForceCurve
from ..utils.segment import Segment

 
#from pyfmreader.utils.forcecurve import ForceCurve
#from pyfmreader.utils.segment import Segment

def loadPSNEXcurve(file_metadata,curve_index = 0):
    """
    Function used to load the data of a single force curve from a PSNEX file.

            Parameters:
                    file_metadata (dict): Dictionary containing the file metadata.

                    curve_index (int): Index of curve to load.
            
            Returns:
                    force_curve (utils.forcecurve.ForceCurve): ForceCurve object containing the loaded data.
    """
    file_id = file_metadata['Entry_filename']
    curve_properties = file_metadata['curve_properties']
    height_channel_key = file_metadata['height_channel_key']
    deflection_chanel_key = file_metadata['deflection_chanel_key']
    tdms_file_ps_nex_file = TdmsFile.open(file_metadata['file_path'])  # alternative TdmsFile.read(path1+fname[ibead])
    tick_time_s = file_metadata['instrument_tick_time_(s)']
    #please add it inthe file metadata
    z_stage_sens_m = 6000*10**-9

    force_curve = ForceCurve(curve_index, file_id)

    curve_indices = file_metadata["Entry_tot_nb_curve"] 
    num_segment = file_metadata['num_segments']
    #TODO please fix this 
    num_segment_arr = [0,2]
    index = 1 if curve_indices == 0 else 3
    
    tdms_groups = tdms_file_ps_nex_file.groups()  ;    tdms_psnex_fc = tdms_groups[0]
 
    deflection = tdms_psnex_fc[deflection_chanel_key][:]
    height = tdms_psnex_fc[height_channel_key][:]*z_stage_sens_m
    seg_pos_array =[[0,0]] * len(num_segment_arr)
    #for the offset, and the final time array  
    t0 = 0;time_fc =  np.array([])
    
    #TALK TO ENZO, remind him 
    #TODO delays in the system , a bool maybe to trigger this 
    tick_sampling_rate_time_s= 2e-06
    z_sensor_delay = 1e-04
    num_pts_rm = int(z_sensor_delay/tick_sampling_rate_time_s)
    
    sizes_seg = np.array([
    curve_properties[str(curve_index)][i][f"segment_{i}_nb_points_cal"]
    for i in num_segment_arr])
    #TODO what the helllis this 
    num_pts_rm_time = num_pts_rm- (len(height) - np.sum(sizes_seg))

    start_indices = np.concatenate(([0], np.cumsum(sizes_seg[:-1])))
    end_indices = start_indices + sizes_seg


     
    deflection = deflection[:-num_pts_rm]
    height = height[num_pts_rm:]
    #find size of each seg




    #finding the seg_pos_array from max z height 
    
    for idx in range(len(num_segment_arr)):
        start_pos,end_pos = start_indices[idx],end_indices[idx]
        print(start_pos,end_pos)

        segment_id = num_segment_arr[idx]
        segment_raw_data = {}
        segment_formated_data = {}
        
        segment_type = curve_properties[str(curve_index)][segment_id][f"segment_{segment_id}_type"]
        segment_duration = curve_properties[str(curve_index)][segment_id][f"segment_{segment_id}_duration_(ticks)"]*tick_time_s
        segment_num_points = curve_properties[str(curve_index)][segment_id][f"segment_{segment_id}_nb_points_cal"]

        # TO DO: Time can be exported, handle this situation.
        #segment_formated_data["time"] = np.linspace(0, segment_duration, segment_num_points, endpoint=False)
        segment_formated_data["time"] = np.linspace(0, segment_duration, end_pos-start_pos, endpoint=False)

        segment_formated_data[height_channel_key] = height[start_pos:end_pos]
        segment_formated_data['vDeflection'] = deflection[start_pos:end_pos]


        segment = Segment(file_id, segment_id, segment_type)
        segment.segment_formated_data = segment_formated_data
        
        segment.segment_metadata = curve_properties[str(curve_index)][segment_id]
        #TODO what is the set point mode 
        #segment.force_setpoint_mode = JPK_SETPOINT_MODE
        
        segment.nb_point = segment_num_points
        
        segment.nb_col = len(segment_formated_data.keys())
        
        segment.force_setpoint = segment.segment_metadata[f"segment_{segment_id}_setpoint_(V)"]
        segment.velocity = segment.segment_metadata[f"segment_{segment_id}_ramp_speed_nm/s"]
        
        segment.sampling_rate = segment.segment_metadata[f"segment_{segment_id}_sampling_rate_(S/s)"]
        segment.z_displacement = segment.segment_metadata[f"segment_{segment_id}_Z_retract_length_(V)"]
        print(segment.segment_type)
        if segment.segment_type == "App":
            #if we overshoot in the appracoh 
            
            if np.nanargmax(height) != end_pos:
                print("overshoot in the approach, accounted for   ")
                end_indices[idx] = np.nanargmax(height)
                start_indices[idx+1] = np.nanargmax(height)

                end_pos = end_indices[idx]
                segment_formated_data["time"] = np.linspace(0, segment_duration, end_pos-start_pos, endpoint=False)

                segment_formated_data[height_channel_key] = height[start_pos:end_pos]
                segment_formated_data['vDeflection'] = deflection[start_pos:end_pos]

            force_curve.extend_segments.append((int(segment.segment_id), segment))
            
            print("success")
        elif segment.segment_type == "Ret":
            #TODO rem half points from each time segment for aligning  
            segment_formated_data["time"] = segment_formated_data["time"][:-(num_pts_rm_time)]
            force_curve.retract_segments.append((int(segment.segment_id), segment))
        elif segment.segment_type == "Con":
            force_curve.pause_segments.append((int(segment.segment_id), segment))
        elif segment.segment_type == "Modulation":
            force_curve.modulation_segments.append((int(segment.segment_id), segment))

    return force_curve