# Import logging and get global logger
import logging
logger = logging.getLogger()
# Import for multiprocessing
import concurrent.futures
# Get loadfile function from PyFMReader
from pyfmreader import loadfile
# Get constants
import pyfmgui.const as const

def load_single_file(filepath):
    try:
        file = loadfile(filepath)
        file_id = file.filemetadata['Entry_filename']
        file_type = file.filemetadata['file_type']
        is_map_file = file.filemetadata.get('mapping_bool') # check if the file is part of a map file
        if file.isFV and file_type in const.nanoscope_file_extensions:
            file.getpiezoimg()
        # elif is_map_file and file_type in const.psnex_file_extension:
        #     print("Entered getpiezoimg for PS-NEX file")
        #     None
        #     file.getpiezoimg()
        return (file_id, file)
    except Exception as error:
        logger.info(f'Failed to load {filepath} with error: {error}')

def loadfiles(session, filelist, progress_callback, range_callback, step_callback):
    files_to_load = [path for path in filelist if path not in session.loaded_files_paths]
    loaded_files = []
    count = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(load_single_file, filepath) for filepath in files_to_load]
        loaded_files = []
        for future in concurrent.futures.as_completed(futures):
            loaded_files.append(future.result())
            count += 1
            progress_callback.emit(count)

    # Remove NONE values
    loaded_files = [r for r in loaded_files if r is not None]

    # Loop and save files in the session
    for file_id, file in loaded_files:
        try:
            session.loaded_files[file_id] = file
        except Exception as e:
            logger.info(f'Failed to add file {file_id} to loaded_files with error: {e}')
