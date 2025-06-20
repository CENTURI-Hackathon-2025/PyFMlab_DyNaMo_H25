#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 18:07:01 2024

@author: yogehs
"""
from .parsepsnexheader import parsePSNEXheader, parsePSNEXsegmentheader

def loadPSNEXfile(filepath, UFF):
    """
    Function used to load the metadata of a PS_nex file.

            Parameters:
                    filepath (str): Path to the PS_nex file.
                    UFF (uff.UFF): UFF object to load the metadata into.
            
            Returns:
                    UFF (uff.UFF): UFF object containing the loaded metadata.
    """
    UFF.filemetadata = parsePSNEXheader(filepath)

    #key for the channel of ht and defleciton

    # UFF.filemetadata['found_vDeflection'] = True
    # UFF.filemetadata['height_channel_key'] = "Zpiezo_stage_sensor_(V)"
    #UFF.filemetadata['height_channel_key'] = "Zpiezo_stage_Vout_(V)"

    # UFF.filemetadata['deflection_chanel_key'] = "Deflection_quotient_(V)"
    curve_properties = {}

    curve_indices =  UFF.filemetadata["Entry_tot_nb_curve"] 

    index = 1 if curve_indices == 0 else 3

    for i in range( UFF.filemetadata["num_segments"] ):
        if index == 3:
            #curve_id = segment_group[0].split("/")[1]
            curve_id =  UFF.filemetadata["curve_id"] 
        else:
            curve_id = '0'
        segment_id = i
        if not curve_id in curve_properties.keys():
            curve_properties.update({curve_id:{}})

        curve_properties = parsePSNEXsegmentheader(filepath,curve_properties, segment_id, UFF, curve_id)

    UFF.filemetadata['curve_properties'] = curve_properties
    
    #TODO what the hall have you done 
    # UFF.isFV = UFF.filemetadata["mapping_bool"]
    UFF.filemetadata['isFV'] = False
    # UFF.filemetadata['num_x_pixels'] = 32
    # UFF.filemetadata['num_y_pixels'] = 32
    # UFF.filemetadata['scan_size_x'] = 0
    # UFF.filemetadata['scan_size_y'] = 0
    UFF.filemetadata['file_type'] = 'PSNEX.tdms'

    return UFF




    