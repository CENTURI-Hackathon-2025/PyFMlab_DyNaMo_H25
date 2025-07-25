# File containing the function loadfile,
# used as an entry point to load different
# AFM data format files.

import os
from nptdms import TdmsFile

from .constants import *
from .jpk.loadjpkfile import loadJPKfile
from .jpk.loadjpkthermalfile import loadJPKThermalFile
from .nanosc.loadnanoscfile import loadNANOSCfile
from .ps_nex.loadpsnexfile import loadPSNEXfile
from .load_uff import loadUFFtxt
from .uff import UFF
from .uff import CachedZipStore



def loadfile(filepath):
    """
    Load AFM file. 
    
    Supported formats:
        - JPK --> .jpk-force, .jpk-force-map, .jpk-qi-data
        - JPK Thermal --> .tnd
        - NANOSCOPE --> .spm, .pfc, .00X
        - UFF --> .uff
        - PS-NEX --> .tdms 

            Parameters:
                    filepath (str): Path to the file.
            
            Returns:
                    If JPK, NANOSCOPE OR UFF:
                        UFF (uff.UFF): Universal File Format object containing loaded data.
                    If JPK Thermal:
                        Amplitude (m^2/V) (np.array),
                        Frequencies (Hz) (np.array),
                        Fit-Data (m^2/V) (np.array),
                        Parameters (dict)


    """
    split_path = filepath.split(os.extsep)
    # Depending on the configuration of the OS, JPK files have the following
    # extension: .jpk-force.zip
    if split_path[-1] == 'zip': filesuffix = split_path[-2]
    else: filesuffix = split_path[-1]

    uffobj = UFF()


    if filesuffix[1:].isdigit() or filesuffix in nanoscfiles:
        return loadNANOSCfile(filepath, uffobj)

    elif filesuffix in jpkfiles:
        #storing the file into cached zip store
        uffobj.zipbuffer= CachedZipStore()
        uffobj.zipbuffer.load(filepath)
        
        return loadJPKfile(filepath, uffobj, filesuffix)

    
    elif filesuffix in ufffiles:
        return loadUFFtxt(filepath, uffobj)
    
    elif filesuffix in jpkthermalfiles:
        return loadJPKThermalFile(filepath)
    
    elif filesuffix in psnexfiles:

        tdms_file = TdmsFile.read_metadata(filepath)
        if 'PSnex' in tdms_file['Force Curve'].properties.get("instrument"):
            print("PSnex is the best")

            return loadPSNEXfile(filepath, uffobj)
        else:
            print('here you can you use any tdms file reading ')

    else:
        Exception(f"Can not load file: {filepath}")



