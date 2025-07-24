# File containing the UFF class.
# Used to store data and metadata.

from zipfile import ZipFile
from io import BytesIO
from functools import lru_cache

# from .jpk.loadjpkfile import zipbuffer
from .constants import *
from .jpk.loadjpkcurve import loadJPKcurve
from .jpk.loadjpkimg import computeJPKPiezoImg
from .nanosc.loadnanosccurve import loadNANOSCcurve
from .nanosc.loadnanoscimg import loadNANOSCimg
from .ps_nex.loadpsnexcurve import loadPSNEXcurve
from .load_uff import loadUFFcurve
from .save_uff import saveUFFtxt

class CachedZipStore:
    def __init__(self):
        self._zip_path = None
        self._zipfile = None

    def load(self, path):
        self._zip_path = path
        self._zipfile = ZipFile(self._zip_path, mode='r')

    def namelist(self):
        if not self._zipfile:
            raise RuntimeError("ZIP not loaded")
        return self._zipfile.namelist()

    @lru_cache(maxsize=128)
    def get_file(self, name):
        if not self._zipfile:
            raise RuntimeError("ZIP not loaded")
        with self._zipfile.open(name) as f:
            return f.read()

    def close(self):
        if self._zipfile:
            self._zipfile.close()
            self._zipfile = None

    def get_file_size(self, name):
        if not self._zipfile:
            raise RuntimeError("ZIP not loaded")
        return self._zipfile.getinfo(name).file_size
    def __del__(self):
        # Ensure the ZIP file is closed if not already
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



    

class UFF:
    """
    Class used to store the data and metadata of an AFM file.

            Properties:
                    filemetadata (dict): Dictionary containing the file metadata.
                    isFV (bool): Flag indicating if the file is a Force Volume or not.
                    piezoimg (np.array): 2D np.array containing the piezo image of the file.
                    imagedata (dict): dictionary containing additional image data.
            
            Methods:
                    getcurve
                    getpiezoimg
                    to_txt

    """
    def __init__(self):
        self.filemetadata=None
        # JPK Specific Atributes
        self._sharedataprops=None
        self._groupedpaths=None
        # FV Specific Atribtues
        self.isFV=None
        self.piezoimg=None
        # In files like JPK scans you may
        # have additional image data.
        self.imagedata=None
        self.zipbuffer=None
    

    def getcurve(self, curveidx):
        """
        Function used to load a single curve from a file.
        
        Supported formats:
            - JPK --> .jpk-force, .jpk-force-map, .jpk-qi-data
            - NANOSCOPE --> .spm, .pfc
            - UFF --> .uff
            - PS-NEX --> .tdms 


                Parameters:
                        curveidx (int): Index of curve to load.
                
                Returns:
                        FC (utils.forcecurve.ForceCurve): ForceCurve object containing the force curve data.
        """
        file_type = self.filemetadata['file_type']
        if file_type in jpkfiles:
            curvepaths = self._groupedpaths[curveidx]
            FC = loadJPKcurve(
                curvepaths, self.zipbuffer, curveidx, self.filemetadata)
        elif file_type[1:].isdigit() or file_type in nanoscfiles:
            FC = loadNANOSCcurve(curveidx, self.filemetadata)

        elif file_type in ufffiles:
            FC = loadUFFcurve(self.filemetadata)
        elif file_type in psnexfiles:
            FC = loadPSNEXcurve(self.filemetadata,curveidx) 
        return FC
    
    def getpiezoimg(self):
        """
        Function used to compute the piezo image of a file.

        It is required that the file is a Force Volume.
        
        Supported formats:
            - JPK --> .jpk-force-map, .jpk-qi-data
            - NANOSCOPE --> .spm, .pfc

                Parameters: None
                
                Returns:
                        piezoimg (np.array): 2D array containing the piezo image of the file.
        """
        file_type = self.filemetadata['file_type']
        if file_type in jpkfiles:
            self.piezoimg = computeJPKPiezoImg(self)
        elif file_type[1:].isdigit() or file_type in nanoscfiles:
            self.piezoimg = loadNANOSCimg(self.filemetadata)
        return self.piezoimg
    
    def to_txt(self, savedir):
        """
        Function used to save the loaded data into a txt file following the UFF.

                Parameters:
                        savedir (str): Path to save the txt UFF file.
                
                Returns: None
        """
        if self.isFV:
            for curveidx in range(self.filemetadata['Entry_tot_nb_curve']):
                saveUFFtxt(self, self, savedir, curveidx)
        else:
            saveUFFtxt(self, self, savedir)
        
