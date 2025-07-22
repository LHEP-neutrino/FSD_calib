# Script written by Nicolas Sallin to perform some gain analysis
# A jupyter notebook, SNR_VGA.ipynb, present some possibilities
#how to use the different function
# Contact nicolas.sallin@unibe.ch or @Nicolas Sallin on Slack for
#any question

import sys
import subprocess
import os

import json
import numpy as np
import logging
import argparse
import h5py
import matplotlib.pyplot as plt



sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Detector properties
N_ADC = 1 #4
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


class LEDFiles:
    """
    Manage the input files and the resulting outputs of gainAnalysis
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

        self.filesDate = np.full(self.N_files, 'None', dtype='<U10')
        self.filesAltID = np.full(self.N_files, 'None', dtype='<U16')

        logger.info(f'The input json file: {input_json}, {self.N_files} data files were found')
        logger.info(f'The ouput folder: {self.output_path}')
                


    def get_gains(self, Nevent = -1, mode='minimal'):
        '''
        Get the gain for each of the files provided in the input, use <Nevent> to compute the fingerplots.

        The 'minimal' mode will store only the gain values, the 'full' mode will store the fingerplots.
        '''
        n_file = 0

        if mode=='full' :
            self.fingerplots = np.zeros((self.N_files, N_ADC, N_chan_ADC))

        self.gains = np.zeros((self.N_files, N_ADC, N_chan_ADC))
            

        for folder in self.input_files_dict.keys():
            for file in self.input_files_dict[folder]:

                calibClass = calibClass(folder, file['file_name'], self.output_path)
                calibClass.compute_fingerplots(Nevent=Nevent)
                calibClass.compute_gains(Nevent=Nevent)

                logger.debug(f'Shape of calibClass.fingerplots {calibClass.fingerplots.shape}, shape of calibClass.gains {calibClass.gains.shape}')
                
                np.copyto(self.gains[n_file], calibClass.gains)

                if mode=='full':
                    np.copyto(self.fingerplots[n_file], calibClass.fingerplots)

                logger.debug(f"Gain computed for file with date: {file['date']}, alt. name: {file['alt_name']}")
                self.filesDate[n_file] = file['date']
                self.filesAltID[n_file] = file['alt_name']

                del calibClass

                n_file += 1

        logger.debug(f'{n_file} files over {self.N_files} files were computed')       

        logger.info(f'The gains were computed')

class calibClass:
    ''' 
        Class to compute the gain for the SiPM waveforms

        Inputs to this class are as follows:

            - filedir          (str):   Path to input file
            - filename         (str):   Name of input flow file
            - ouput_path       (str):   Path where to save the figures, if None: save in LAr_evd/FSD_eventDisplay/ (default: None)

        Class methods:

            
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
            self.output_path = os.path.join(os.path.dirname(__file__) ,f'evD_{filename}/')

        logger.debug(f'The ouput folder: {self.output_path}')

        # Load light events, waveform datasets and light geometry info if using
        self.light_events = f['light/events/data']
        self.light_wvfms = f['light/wvfm/data']['samples']

        Nevents = len(self.light_wvfms)

        self.peaks_idx = [[[None for _ in range(N_chan_ADC)] for _ in range(N_ADC)] for _ in range(Nevents)]
        self.fingerplots = [[None for _ in range(N_chan_ADC)] for _ in range(N_ADC)]

        # Defined some module properties
        self.N_sipm_side = int(60)
        self.N_side_tpc = 2
        self.N_tpc = 2
        self.N_sipm_lightModule = 6
        self.N_LCM_lightModule = 3
        self.time_tick = 16*10**-9 # [s]

        logger.debug(f"Number of events in the selection: {len(self.light_events)}")
        logger.debug(f"Shape of the waveforms array: {self.light_wvfms.shape}")

    def _extract_peak(self, wvfm, minWidth, mode='default', search_int = None, cut=None):
        '''
            extract the peak of the waveforms in the between range[0] and range[1].

            Args:
                minWidth:   Minimal width of a peak (e.g. a 5 ticks peaks have two values lower than the peak summit on each side)
                mode:       Mode of the returned peaks
                    - 'default':    return only the peak above the mean
                search_int:      indices of the intervall of the wvfm in which to search for peak
        '''
        peak_idx = []
        is_peak = True
        Npt_peakSide = int(minWidth/2)
        mean = np.mean(wvfm)

        if search_int == None:
            search_int = np.array([0, len(wvfm)-1])

        else:
            search_int = np.sort(search_int)
            if search_int[0] < 0 or search_int[1] > len(wvfm)-1:
                raise ValueError('The given search interval is out of bound')
            search_int += [-Npt_peakSide, Npt_peakSide]
        
        for ix in range(search_int[0]+Npt_peakSide+1, search_int[1]-Npt_peakSide-1):
            for i in range(1, Npt_peakSide+1):
                if (wvfm[ix-i] > wvfm[ix] or wvfm[ix+i] > wvfm[ix]):
                    is_peak = False
                    break

            if (is_peak==True):
                peak_idx.append(ix)
            else:
                is_peak = True

        peak_idx = np.array(peak_idx, dtype=int)
        
        if (mode=='default'):
           # Only return the peak above the mean
           aboveMean_idx = np.where(wvfm[peak_idx]>mean)[0]
           peak_idx = peak_idx[aboveMean_idx]

        if (cut == '15ticks'):
            peak_idx = np.array([peak_idx[i] for i in range(1, len(peak_idx)-1) if peak_idx[i-1] + 15 < peak_idx[i] < peak_idx[i+1] - 15], dtype=int)

        return peak_idx, mean 
    

    
    def findPeak_wvfms(self, event, adc, chan, minWidth=5, search_int=None, cut=None):
        '''
        Find the number of peak and store it in self.Npeaks

        Args:
            
            
        Return:
            None
        '''
        if isinstance(event, int):
            peak_idx, *peak_info = self._extract_peak(self.light_wvfms[event][adc,chan], minWidth, search_int=search_int, cut=cut)
            self.peaks_idx[event][adc][chan] = peak_idx

        elif isinstance(event, list):
            for i_event in range(event[0], event[1]):
                peak_idx, *peak_info = self._extract_peak(self.light_wvfms[i_event][adc,chan], minWidth, search_int=search_int, cut=cut)
                
                self.peaks_idx[i_event][adc][chan] = peak_idx

        return None

    def compute_gain(self):
        '''
        Compute the gain from the fit parameters
        '''

        logger.debug(f"Computing the gain of the file {self.file}")
        # self.gains = ...

        return None

    def fit_fingerplots(self, p0 = None):
        '''
        Fit the fingerplots with the initial parameters 'p0'.

        Args:
            p0 (list): list of initial parameters, [mu 1pe peak, sigma 1pe peak, gain, pedestal]
                if None : 
        '''

        logger.debug(f"Fitting the fingerplots of the file {self.file}")

        #1st: make sure that the finger plots were set
        # if self.fingerplots == 0 :
        #     raise ValueError('The fingerplots were not computed. run compute_fingerplot first or manually set calibClass.fingerplots')

        #2nd: find the initial parameters
        # if p0 == None :
        #   mu0 = ...
        #   sig0 = ...
        #   gain0 = ...
        #   ped0 = ...
        #else:
        # mu0, sig0, gain0, ped0 = p0

        #3rd: fit with multigaussian

        #4th: store the parameters, the peak means, the sig of the peaks


        return None
    
    def compute_fingerplots(self, Nevent = -1, Int_window=[0, -1], Nbins=200, mode='integral', cut=None):
        '''
        Plot the distribution of integrated waveforms, so called fingers plot

        Args:
            Nevent (int)                    : Number of event include in the computation (default: -1, all events)
            Int_window (np.array or list)   : Integaration window, the waveform will be integrated from Int_window[0] to Int_window[1]
            mode (str)                      : Mode of computation
                - 'integral' (default) : Integrate the waveform in the given window
                - 'amplitude'          : Maximal amplitude of the peak
                - 'amplitude_peaks'    : Includes all the peaks found by the peak finder

                - 'fit_int'            : Integral of the fitted waveform, TODO
                - 'fit_amp'            : Max. amplitude of the fitted waveform, TODO
            cut (str)                       : Cut(s) applied to the select event
                - None (default)       : No cut
                - 1peak                : Only select event with one peak in 'Int_window' 
                - 15ticks              : Only select peak at least 15 ticks away from neighbouring peaks   
        '''

        # Cut variable
        minWidth = 9

        if Nevent == -1 or Nevent > self.light_wvfms.shape[0]:
            Nevent = self.light_wvfms.shape[0]

        logger.debug(f"Computing the fingerplots of the file {self.file} with {Nevent} events")

        
        if (mode == 'integral'):
            for i_adc in range(N_ADC):
                for j_chan in range(N_chan_ADC):
                    if (cut == '1peak'):
                        self.findPeak_wvfms([0, Nevent], i_adc, j_chan, minWidth=minWidth, search_int=Int_window)
                        event_mask = np.full((Nevent), True, dtype=bool)
                        for k_event in range(Nevent):
                            event_mask[k_event] = (len(self.peaks_idx[k_event][i_adc][j_chan])==1)
                        event_mask = np.where(event_mask)[0]
                        intsWvfm = np.sum(self.light_wvfms[event_mask, i_adc, j_chan, Int_window[0]:Int_window[1]], axis=-1)

                    else:
                        intsWvfm = np.sum(self.light_wvfms[:Nevent, i_adc, j_chan, Int_window[0]:Int_window[1]], axis=-1)

                    self.fingerplots[i_adc][j_chan] = np.histogram(intsWvfm, bins=Nbins)
        
        elif (mode == 'amplitude'):
            for i_adc in range(N_ADC):
                for j_chan in range(N_chan_ADC):
                    if (cut == '1peak'):
                        self.findPeak_wvfms([0, Nevent], i_adc, j_chan, minWidth=minWidth, search_int=Int_window)
                        event_mask = np.full((Nevent), True, dtype=bool)
                        for k_event in range(Nevent):
                            event_mask[k_event] = (len(self.peaks_idx[k_event][i_adc][j_chan])==1)
                        event_mask = np.where(event_mask)[0]
                        ampWvfm = np.max(self.light_wvfms[event_mask, i_adc, j_chan, Int_window[0]:Int_window[1]], axis=-1)


                    else:
                        ampWvfm = np.max(self.light_wvfms[:Nevent, i_adc, j_chan, Int_window[0]:Int_window[1]], axis=-1)

                    self.fingerplots[i_adc][j_chan] = np.histogram(ampWvfm, bins=Nbins)

        elif (mode == 'amplitude_peaks'):
            for i_adc in range(N_ADC):
                for j_chan in range(N_chan_ADC):
                    self.findPeak_wvfms([0, Nevent], i_adc, j_chan, minWidth=minWidth, search_int=Int_window, cut=cut)

                    ampsWvfm = np.array([])
                    for k_event in range(Nevent):
                        # print(f'event {k_event}: {self.light_wvfms[k_event, i_adc, j_chan, self.peaks_idx[k_event][i_adc][j_chan]]}')
                        ampsWvfm = np.concatenate((ampsWvfm, self.light_wvfms[k_event, i_adc, j_chan, self.peaks_idx[k_event][i_adc][j_chan]]), axis=None)

                    # print(ampsWvfm)
                    self.fingerplots[i_adc][j_chan] = np.histogram(ampsWvfm, bins=Nbins)

        logger.debug(f'Finger plot computed with {Nevent} events in mode "{mode}"')

        return None


    def extract_gain(self, Nevent = -1):
        """
        Compute the gain from a calib run

        Args:
            Nevent (int):       Number of event includes in the computation. (default: -1, all events)

        Return:
            None
        """

        if Nevent == -1:
            Nevent = self.light_wvfms.shape[0]
        else:
            Nevent = np.min([Nevent, self.light_wvfms.shape[0]])

        logger.debug(f"Computing the gains of the file {self.file} with {Nevent} events")
        
        # self.compute_fingerplots()
        # self.fit_fingerplots()
        # self.compute_gain()
        
        # logger.debug(f"Shape of gains: {self.gains.shape}")

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

def plot_fingerplot(counts, bins, title=''):

    fig = plt.figure(figsize=[10, 6])
    ax = fig.subplots()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]

    ax.bar(bin_centers, counts, width=width, color='skyblue', label=f'N = {np.sum(counts)}')

    ax.set_title(title, fontsize=8)
    ax.legend()

    fig.show()

    return None


if __name__ == "__main__":

    # Parse the arguments
    args = parse_args()

    # Set the log level
    if args.log_level != 'INFO':
        set_log_level(args.log_level)
    
    logger.debug(f'Argument parsed: {args}')
    
    # Initialize the Baselinefiles
    LED_files = LEDFiles(os.path.normpath(args.input_json), os.path.normpath(args.output_folder))

    #TODO: complete