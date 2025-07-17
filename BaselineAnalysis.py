# Script written by Nicolas Sallin to perform some Baseline analysis
# A jupyter notebook, BaselineAnalysis.ipynb, present some possibilities
# Contact nicolas.sallin@unibe.ch or @Nicolas Sallin on Slack

import sys
import subprocess
import os

import json
import numpy as np
import logging
import argparse
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Detector properties
N_ADC = 4
N_chan_ADC = 64

N_sipm_side = 60
N_TPC = 2
N_side = 2


# Setup module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default level

# Avoid adding multiple handlers on re-import
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def set_log_level(level_name: str):
    """
    Change log level at runtime.

        input: level_name, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    
    """
    level = getattr(logging, level_name.upper(), None)
    if isinstance(level, int):
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
        logger.info(f"Log level changed to {level_name.upper()}")
    else:
        logger.error(f"Invalid log level: {level_name}")


class BaselineFiles:
    """
    Manage the input files and the resulting outputs of baselineWvfms
    """

    def __init__(self, input_json, output_path=None, log_level=None):

        if log_level != None:
            set_log_level(log_level)

        # Load from JSON file
        with open(input_json, 'r') as f_json:
            self.input_files_dict = json.load(f_json)

        # Set the output path
        if (output_path == None):
            self.output_path = os.path.join(os.path.dirname(__file__) ,f'Baselines_{os.path.splitext(os.path.basename(input_json))[0]}/')
        else:
            self.output_path = os.path.abspath(output_path)

        # Compute the number of files
        self.N_files = 0
        for folder in self.input_files_dict.keys():
            self.N_files += len(self.input_files_dict[folder])


        # Initialize the baselines array
        self.baselines = np.zeros((self.N_files, N_ADC, N_chan_ADC))
        self.baselines_std = np.zeros((self.N_files, N_ADC, N_chan_ADC))


        self.filesDate = np.full(self.N_files, 'None', dtype='<U10')
        self.filesAltID = np.full(self.N_files, 'None', dtype='<U16')

        logger.info(f'The input json file: {input_json}, {self.N_files} data files were found')
        logger.info(f'The ouput folder: {self.output_path}')
                


    def get_baselines(self, Nevent = -1, mode='file'):
        n_file = 0

        if mode=='event' and Nevent>0:
            self.baselines_event = np.zeros((self.N_files, Nevent, N_ADC, N_chan_ADC))
            self.baselines_event_std = np.zeros((self.N_files, Nevent, N_ADC, N_chan_ADC))

        for folder in self.input_files_dict.keys():
            for file in self.input_files_dict[folder]:

                bl_wvfms = baselineWvfms(folder, file['file_name'], self.output_path)
                bl_wvfms.compute_baselines(Nevent=Nevent)

                logger.debug(f'shape of bl_wvfms.baselines {bl_wvfms.baselines.shape}, shape of bl_wvfms.baselines_event {bl_wvfms.baselines_event.shape}')
                
                np.copyto(self.baselines[n_file], bl_wvfms.baselines)
                np.copyto(self.baselines_std[n_file], bl_wvfms.baselines_std)

                if mode=='event':
                    np.copyto(self.baselines_event[n_file], bl_wvfms.baselines_event)
                    np.copyto(self.baselines_event_std[n_file], bl_wvfms.baselines_event_std)


                logger.debug(f"date: {file['date']}, alt. name: {file['alt_name']}")
                self.filesDate[n_file] = file['date']
                self.filesAltID[n_file] = file['alt_name']

                del bl_wvfms

                n_file += 1

        logger.debug(f'{n_file} files over {self.N_files} files were computed')       

        logger.info(f'The baseline were computed')

class baselineWvfms:
    ''' 
        Class to compute the baseline for the FSD waveforms

        Inputs to this class are as follows:

            - filedir          (str):   Path to input file
            - filename         (str):   Name of input flow file
            - ouput_path       (str):   Path where to save the figures, if None: save in LAr_evd/FSD_eventDisplay/ (default: None)

        Class methods:

            - dumpWvfms()           :   Dump the waveforms as png at the output path
            
    '''

    # Initialize the class
    def __init__(self, filedir, filename, output_path=None, log_level=None):
        
        if log_level != None:
            set_log_level(log_level)

        self.file = os.path.join(filedir,filename)

        # Open files
        f = h5py.File(self.file, 'r')

        logger.debug(f'Processing {self.file}')

       
        # Set the output path
        if (output_path != None):
            self.output_path = os.path.abspath(output_path)
        else:
            self.output_path = os.path.join(os.path.dirname(__file__) ,f'evD_{self.filename}/')

        logger.debug(f'The ouput folder: {self.output_path}')

        # Load light events, waveform datasets and light geometry info if using
        self.light_events = f['light/events/data']
        self.light_wvfms = f['light/wvfm/data']['samples']

        self.baselines_event = np.zeros(self.light_wvfms.shape[:-1])

        # Defined some module properties
        self.N_sipm_side = int(60)
        self.N_side_tpc = 2
        self.N_tpc = 2
        self.N_sipm_lightModule = 6
        self.N_LCM_lightModule = 3
        self.time_tick = 16*10**-9 # [s]

        logger.debug(f"Number of events in the selection: {len(self.light_events)}")
        logger.debug(f"Shape of the waveforms array: {self.light_wvfms.shape}")


    def compute_baselines(self, mode='file', Nevent = -1):
        """
        Compute the baseline from a baseline run: take the mean of all the data point

        Args:
            mode (str): Define the mode of the computation
                'file' (default): One baseline per channel is computed
                'event'         : One baseline per event and per channel is computed
            Nevent (int):   Number of event includes in the computation. (default: -1, all events)

        Return:
            None
        """

        if Nevent == -1:
            Nevent = self.light_wvfms.shape[0]
        else:
            Nevent = np.min([Nevent, self.light_wvfms.shape[0]])

        if mode=='file':
            logger.debug(f"Computing the mean baselines of the file {self.file} with {Nevent} events")
            self.baselines_event = np.mean(self.light_wvfms[:Nevent], axis=-1)
            self.baselines_event_std = np.std(self.light_wvfms[:Nevent], axis=-1)

            logger.debug(f"Shape of baselines_event: {self.baselines_event.shape}")

            self.baselines = np.mean(self.baselines_event, axis=0)
            self.baselines_std = np.std(self.baselines_event, axis=0)

        elif mode == 'event':
            logger.debug(f"Computing the baselines for each event of the file {self.file}")
            self.baselines_event = np.mean(self.light_wvfms, axis= -1)
            self.baselines_event_std = np.std(self.light_wvfms, axis= -1)

        return None
    
    # def compute_distrib_baselines(self, mode='file'):
    #     """
    #     Compute the distribution of the baseline from a baseline run

    #     Args:
    #         mode (str): Define the mode of the computation
    #             'file' (default): One baseline per channel is computed
    #             'event'         : One baseline per event and per channel is computed

    #     Return:
    #         None
    #     """

    #     if mode=='file':
    #         logger.debug(f"Computing the mean baselines of the file {self.file}")
    #         self.baselines_event = np.mean(self.light_wvfms, axis=-1)
    #         self.baselines = np.mean(self.baselines_event, axis=0)

    #     elif mode == 'event':
    #         logger.debug(f"Computing the baselines for each event of the file {self.file}")
    #         self.baselines_event = np.mean(self.light_wvfms, axis= -1)

    #     return None

    def plot_wvfms(self, event, adc, chan, xlim = None):
        '''
        Plot the light waveforms.

        Args:
            
            
        Return:
            None
        '''
        # Compute the x-axis coordinate
        x_ticks = np.arange(0, self.light_wvfms[0].shape[-1],  1)

        if (xlim==None):
            xlim = [x_ticks[0], x_ticks[-1]]


        # Setup the plot
        fig_wvfm = plt.figure(figsize=[12.8, 4.8])
        ax_wvfm = fig_wvfm.subplots()
        if (xlim != None):
            ax_wvfm.set_xlim(xlim)
        ax_wvfm.set_xlabel('ticks')
        ax_wvfm.set_ylabel('ADC unit')
        ax_wvfm.grid(True)

        ax_wvfm.plot(x_ticks, self.light_wvfms[event][adc,chan], label=f'Event {event}, ADC {adc}, Chan. {chan}', marker='.', ls='')
        ax_wvfm.plot(x_ticks, self.light_wvfms[event][adc,chan], marker='', ls='-', c='r', alpha=0.3)

        ax_wvfm.hlines(self.baselines[event, adc, chan], xlim[0], xlim[1], color='k', ls='--', label='Baseline')

        return None
    

def parse_args():
    """
        Parse the arguments
    
    Args:
        None

    Return:
        parser.parse_args():    Namespace containing the arguments's name and their values
    """
    parser = argparse.ArgumentParser(description="Compute the baselines of the given files")
    parser.add_argument("-i", "--input_json", type=str, required=True, help="json file describing the files to analyse")
    parser.add_argument("-o", "--output_folder",type=str, required=True, help="Output folder")
    parser.add_argument("-ll", "--log_level",type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Log level")
    
    return parser.parse_args()


if __name__ == "__main__":

    # Parse the arguments
    args = parse_args()

    # Set the log level
    if args.log_level != 'INFO':
        set_log_level(args.log_level)
    
    logger.debug(f'Argument parsed: {args}')
    
    # Initialize the Baselinefiles
    baseline_files = BaselineFiles(os.path.normpath(args.input_json), os.path.normpath(args.output_folder))
