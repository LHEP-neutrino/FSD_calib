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
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
from datetime import datetime


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Detector properties
N_ADC = 1 #4
N_chan_ADC = 6 #64

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
            self.output_path = os.path.join(os.path.dirname(__file__) ,f"calib_{input_json.split('.')[0]}")
        else:
            self.output_path = os.path.abspath(output_path)

        os.makedirs(self.output_path, exist_ok=True)

        # Compute the number of files
        self.N_files = 0
        for folder in self.input_files_dict.keys():
            self.N_files += len(self.input_files_dict[folder])

        self.filesDate = np.full(self.N_files, 'None', dtype='<U10')
        self.filesAltID = np.full(self.N_files, 'None', dtype='<U16')
        self.fit_status = np.zeros((self.N_files, N_ADC, N_chan_ADC), dtype=bool)
        self.SNRs = np.zeros((self.N_files, N_ADC, N_chan_ADC))
        self.gains = np.zeros((self.N_files, N_ADC, N_chan_ADC))

        logger.info(f'The input json file: {input_json}, {self.N_files} data files were found')
        logger.info(f'The ouput folder: {self.output_path}')
                


    # def get_gains(self, Nevent = -1, mode='minimal'):
    #     '''
    #     Get the gain for each of the files provided in the input, use <Nevent> to compute the fingerplots.

    #     The 'minimal' mode will store only the gain values, the 'full' mode will store the fingerplots.
    #     '''
    #     n_file = 0

    #     if mode=='full' :
    #         self.fingerplots = np.zeros((self.N_files, N_ADC, N_chan_ADC))

    #     self.gains = np.zeros((self.N_files, N_ADC, N_chan_ADC))
            

    #     for folder in self.input_files_dict.keys():
    #         for file in self.input_files_dict[folder]:

    #             calibClass = calibClass(folder, file['file_name'], self.output_path)
    #             calibClass.compute_fingerplots(Nevent=Nevent)
    #             calibClass.compute_gains(Nevent=Nevent)

    #             logger.debug(f'Shape of calibClass.fingerplots {calibClass.fingerplots.shape}, shape of \
    #                          calibClass.gains {calibClass.gains.shape}')
                
    #             np.copyto(self.gains[n_file], calibClass.gains)

    #             if mode=='full':
    #                 np.copyto(self.fingerplots[n_file], calibClass.fingerplots)

    #             logger.debug(f"Gain computed for file with date: {file['date']}, alt. name: {file['alt_name']}")
    #             self.filesDate[n_file] = file['date']
    #             self.filesAltID[n_file] = file['alt_name']

    #             del calibClass

    #             n_file += 1

    #     logger.debug(f'{n_file} files over {self.N_files} files were computed')       

    #     logger.info(f'The gains were computed')

    def get_PEs_SNR(self, baselines, baselines_std, Nevent = -1, int_window = [110, 135], Nbins=100, mode='integral', cut=None, save_plots=False):
        
        nFile = 0
        for folder in self.input_files_dict.keys():
            for file in self.input_files_dict[folder]:

                calib = calibClass(filedir=folder, filename=file['file_name'], output_path=self.output_path)
                calib.extract_PEs(baselines=baselines[nFile], Nevent=Nevent, int_window=int_window, Nbins=Nbins, mode=mode, cut=cut,save_plots=save_plots, show_p0_plots=False)
                
                np.copyto(self.fit_status[nFile], calib.fit_status)
                np.copyto(self.gains[nFile], calib.gains)

                calib.compute_SNR(baselines=baselines[nFile], baselines_std=baselines_std[nFile])

                np.copyto(self.SNRs[nFile], calib.SNRs)

                nFile += 1
            # print(f'For file {file}, nPEs = {calibClass.nPEs}')

        return None
    

class calibClass:
    ''' 
        Class to compute the gain for the SiPM waveforms

        Inputs to this class are as follows:

            - filedir (str):        Path to input file
            - filename (str):       Name of input flow file
            - ouput_path (str):     Path where to save the figures, 
                                    if None: save in LAr_evd/FSD_eventDisplay/ (default: None)

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
        if (output_path is not None):
            self.output_path = os.path.join(os.path.abspath(output_path),f"calib_{filename.split('.')[0]}")
        else:
            self.output_path = os.path.join(os.path.dirname(__file__) ,f"calib_{filename.split('.')[0]}")
        
        os.makedirs(self.output_path, exist_ok=True)

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
                minWidth:   Minimal width of a peak (e.g. a 5 ticks peaks have two values lower than the peak summit 
                            on each side)
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
            peak_idx = np.array([peak_idx[i]
                                for i in range(1, len(peak_idx)-1) if peak_idx[i-1]+15 < peak_idx[i] < peak_idx[i+1]-15],
                                dtype=int)

        return peak_idx, mean 
    

    
    def findPeak_wvfms(self, event, adc, chan, minWidth=5, search_int=None, cut=None):
        '''
        Find the number of peak and store it in self.Npeaks

        Args:
            
            
        Return:
            None
        '''
        if isinstance(event, int):
            peak_idx, *peak_info = self._extract_peak(self.light_wvfms[event][adc,chan], minWidth, search_int=search_int,
                                                       cut=cut)
            self.peaks_idx[event][adc][chan] = peak_idx

        elif isinstance(event, list):
            for i_event in range(event[0], event[1]):
                peak_idx, *peak_info = self._extract_peak(self.light_wvfms[i_event][adc,chan], minWidth,
                                                           search_int=search_int, cut=cut)
                
                self.peaks_idx[i_event][adc][chan] = peak_idx

        return None


    def _compute_fingerplots_p0(self, counts, bin_centers, width, posRatio_noisePeak = 0.4):
        '''
        Compute the initial parameter 'p0' for  the fingerplots fit and the bounds of the fit parameters.

        Args:
            counts (np.array):      The values of the histogram (returned from np.histogram)
            bin_centers (np.array): Center of the bins corresponding to 'counts'
            width (int):            Width of the bins w.r.t. counts and bin_centers


        Return:
            fit_p0 (np.array):       The initial parameters for the fingerplot fit
            fit_bounds (np.array):   The bounds of the fit parameters
            fit_lim (np.array):      An array containing the range where the fit will be computed
        '''
        # Get the peaks position above <height>
        peaks, properties = find_peaks(counts, distance=5, width=1, prominence=20, height=np.mean(counts))

        Npeaks_fit = len(peaks)

        # Add a second peak for each main peak, representing the an observed noise effect (cross-talk?)
        # Compute the noise peak position w.r.t. the distance between the main peaks
        noise_peaks = np.array((peaks[1:]-peaks[:-1])*posRatio_noisePeak, dtype=int)
        # Add the noise peak of the last main peak
        noise_peaks = np.append(noise_peaks, noise_peaks[-1])
        # Get the absolute position
        noise_peaks += peaks

        # Get the p0 and the fit bounds
        # 2 gaussian per peak and 3 params per gaussian
        self.Nparams_peak = 6
        fit_p0 = np.zeros(Npeaks_fit*self.Nparams_peak)
        fit_bounds = np.zeros((2,Npeaks_fit*self.Nparams_peak))
        
        for i_peak in range(Npeaks_fit):
            # Amplitue main gaussian
            fit_p0[i_peak*self.Nparams_peak] = counts[peaks[i_peak]]
            fit_bounds[0][i_peak*self.Nparams_peak] = fit_p0[i_peak*self.Nparams_peak]*0.95
            fit_bounds[1][i_peak*self.Nparams_peak] = fit_p0[i_peak*self.Nparams_peak]*1.1
            # Mean main gaussian
            fit_p0[i_peak*self.Nparams_peak+1] = bin_centers[0]+width*peaks[i_peak]
            fit_bounds[0][i_peak*self.Nparams_peak+1] = fit_p0[i_peak*self.Nparams_peak+1]-width
            fit_bounds[1][i_peak*self.Nparams_peak+1] = fit_p0[i_peak*self.Nparams_peak+1]+width
            # Std main gaussian
            fit_p0[i_peak*self.Nparams_peak+2] = properties['widths'][i_peak]*width*0.5
            fit_bounds[0][i_peak*self.Nparams_peak+2] = fit_p0[i_peak*self.Nparams_peak+2]*0.8
            fit_bounds[1][i_peak*self.Nparams_peak+2] = fit_p0[i_peak*self.Nparams_peak+2]*2
            # Amplitue secondary gaussian
            fit_p0[i_peak*self.Nparams_peak+3] = counts[noise_peaks[i_peak]]
            fit_bounds[0][i_peak*self.Nparams_peak+3] = fit_p0[i_peak*self.Nparams_peak+3]*0.6
            fit_bounds[1][i_peak*self.Nparams_peak+3] = fit_p0[i_peak*self.Nparams_peak+3]*1.1
            # Mean secondary gaussian
            fit_p0[i_peak*self.Nparams_peak+4] = bin_centers[0]+width*noise_peaks[i_peak]
            fit_bounds[0][i_peak*self.Nparams_peak+4] = fit_p0[i_peak*self.Nparams_peak+4]-width
            fit_bounds[1][i_peak*self.Nparams_peak+4] = fit_p0[i_peak*self.Nparams_peak+4]+2*width
            # Amplitue secondary gaussian
            # Std secondary gaussian
            fit_p0[i_peak*self.Nparams_peak+5] = (properties['widths'][i_peak]*width*0.5)
            fit_bounds[0][i_peak*self.Nparams_peak+5] = fit_p0[i_peak*self.Nparams_peak+5]*0.8
            fit_bounds[1][i_peak*self.Nparams_peak+5] = fit_p0[i_peak*self.Nparams_peak+5]*5

            # logger.debug(f" prop: {properties['left_ips'][0]}")
            if properties['left_ips'][0] > 0 and counts[int(properties['left_ips'][0]-1)]>0:
                    fit_lim_min = properties['left_ips'][0]-1
            else:
                fit_lim_min = properties['left_ips'][0]

            if peaks[-1]+int((peaks[-1]-peaks[-2])*0.5)+1 < len(counts):
                fit_lim_max = peaks[-1]+int((peaks[-1]-peaks[-2])*0.5)+1
            else: 
                fit_lim_max = len(counts)


        return fit_p0, fit_bounds, np.array([fit_lim_min, fit_lim_max], dtype=int)

    def fit_fingerplots(self, show_p0_plots=False):
        '''
        Fit the fingerplots. If the fit fails, saved the p0 parameters instead.

        Args:
            
        '''
        logger.debug(f"Fitting the fingerplots of the file {self.file}")
        
        self.fit_lim = np.array([[np.array([0, self.light_wvfms.shape[-1]], dtype=int) for _ in range(N_chan_ADC)] 
                        for _ in range(N_ADC)])
        self.fitted_params = np.empty((N_ADC, N_chan_ADC), dtype=object)
        self.fitted_pcov = np.empty((N_ADC, N_chan_ADC), dtype=object)
        self.reduced_chi_squared = np.array([[np.array([0, 0], dtype=float) for _ in range(N_chan_ADC)] 
                        for _ in range(N_ADC)])
        self.fit_status = np.zeros((N_ADC, N_chan_ADC), dtype=bool)

        for i_adc in range(N_ADC):
            for j_chan in range(N_chan_ADC):

                counts, bins = self.fingerplots[i_adc][j_chan]

                bin_centers = (bins[:-1] + bins[1:]) / 2
                width = bins[1] - bins[0]
                
                # Compute p0
                logger.debug(f"Compute fingerplots p0 of ADC {i_adc}, chan. {j_chan}")
                fit_p0, fit_bounds, fit_lim = self._compute_fingerplots_p0(counts=counts, bin_centers=bin_centers,
                                                                            width=width)
                self.fit_lim[i_adc][j_chan] = fit_lim
                if (show_p0_plots == True):
                    plot_fingerplot(counts, bins, title=f'Fingerplot and p0 for ADC {i_adc}, chan. {j_chan}', show=True, fit_params=fit_p0, fit_xlim=self.fit_lim[i_adc][j_chan])
                
                # Fit the fingerplots
                try:
                    self.fitted_params[i_adc][j_chan], self.fitted_pcov[i_adc][j_chan] = curve_fit(multi_gaussian,
                                                   bin_centers[fit_lim[0]:fit_lim[1]], 
                                                   counts[fit_lim[0]:fit_lim[1]], p0=fit_p0, 
                                                   bounds=fit_bounds, maxfev = 100000, 
                                                   sigma=np.sqrt(counts[fit_lim[0]:fit_lim[1]]))
                    chi_squared = np.sum(((counts[fit_lim[0]:fit_lim[1]] - multi_gaussian(bin_centers[fit_lim[0]:fit_lim[1]], *self.fitted_params[i_adc][j_chan]))**2 / counts[fit_lim[0]:fit_lim[1]]))
                    dof = len(counts[fit_lim[0]:fit_lim[1]]) - len(self.fitted_params[i_adc][j_chan])  # degrees of freedom
                    # print(f'dof: {len(counts[fit_lim[0]:fit_lim[1]])} and {len(self.fitted_params[i_adc][j_chan])}, fit_lim: {fit_lim}')
                    logger.debug(f"Fingerplots of ADC {i_adc}, chan. {j_chan} fitted")
                except:
                    logger.warning(f"The fingerplot were not fitted for ADC {i_adc}, chan. {j_chan}, the saved parmaters are the p0s")
                    self.fitted_params[i_adc][j_chan] = fit_p0
                    chi_squared = 0
                    dof = 0
                    self.fit_status[i_adc][j_chan] = 1

                self.reduced_chi_squared[i_adc][j_chan] = np.array([chi_squared, dof])

        logger.info(f"The fingerplots of the file {self.file} were fitted.")
        

        return None
    
    def compute_fingerplots(self, Nevent = -1, int_window=[0, -1], Nbins=150, mode='integral', cut=None):
        '''
        Plot the distribution of integrated waveforms, so called fingers plot

        Args:
            Nevent (int)                    : Number of event include in the computation (default: -1, all events)
            int_window (np.array or list)   : Integaration window, the waveform will be integrated from int_window[0]
                                              to int_window[1]
            mode (str)                      : Mode of computation
                - 'integral' (default) : Integrate the waveform in the given window
                - 'amplitude'          : Maximal amplitude of the peak
                - 'amplitude_peaks'    : Includes all the peaks found by the peak finder

                - 'fit_int'            : Integral of the fitted waveform, TODO
                - 'fit_amp'            : Max. amplitude of the fitted waveform, TODO
            cut (str)                       : Cut(s) applied to the select event
                - None (default)       : No cut
                - 1peak                : Only select event with one peak in 'int_window' 
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
                        self.findPeak_wvfms([0, Nevent], i_adc, j_chan, minWidth=minWidth, search_int=int_window)
                        event_mask = np.full((Nevent), True, dtype=bool)
                        for k_event in range(Nevent):
                            event_mask[k_event] = (len(self.peaks_idx[k_event][i_adc][j_chan])==1)
                        event_mask = np.where(event_mask)[0]
                        intsWvfm = np.sum(self.light_wvfms[event_mask, i_adc, j_chan, int_window[0]:int_window[1]], axis=-1)

                    else:
                        intsWvfm = np.sum(self.light_wvfms[:Nevent, i_adc, j_chan, int_window[0]:int_window[1]], axis=-1)

                    self.fingerplots[i_adc][j_chan] = np.histogram(intsWvfm, bins=Nbins)
        
        elif (mode == 'amplitude'):
            for i_adc in range(N_ADC):
                for j_chan in range(N_chan_ADC):
                    if (cut == '1peak'):
                        self.findPeak_wvfms([0, Nevent], i_adc, j_chan, minWidth=minWidth, search_int=int_window)
                        event_mask = np.full((Nevent), True, dtype=bool)
                        for k_event in range(Nevent):
                            event_mask[k_event] = (len(self.peaks_idx[k_event][i_adc][j_chan])==1)
                        event_mask = np.where(event_mask)[0]
                        ampWvfm = np.max(self.light_wvfms[event_mask, i_adc, j_chan, int_window[0]:int_window[1]], axis=-1)


                    else:
                        ampWvfm = np.max(self.light_wvfms[:Nevent, i_adc, j_chan, int_window[0]:int_window[1]], axis=-1)

                    self.fingerplots[i_adc][j_chan] = np.histogram(ampWvfm, bins=Nbins)

        elif (mode == 'amplitude_peaks'):
            for i_adc in range(N_ADC):
                for j_chan in range(N_chan_ADC):
                    self.findPeak_wvfms([0, Nevent], i_adc, j_chan, minWidth=minWidth, search_int=int_window, cut=cut)

                    ampsWvfm = np.array([])
                    for k_event in range(Nevent):
                        ampsWvfm = np.concatenate((ampsWvfm, self.light_wvfms[k_event, i_adc, j_chan, 
                                                    self.peaks_idx[k_event][i_adc][j_chan]]), axis=None)

                    self.fingerplots[i_adc][j_chan] = np.histogram(ampsWvfm, bins=Nbins)

        logger.info(f'The finger plots were computed with {Nevent} events in mode "{mode}"')

        return None

    def compute_gains(self, mode='integral'):
        
        self.gains = np.empty((N_ADC, N_chan_ADC))
        self.gains_std = np.empty_like(self.gains)

        for i_adc in range(N_ADC):
            for j_chan in range(N_chan_ADC):
                if self.fit_status[i_adc][j_chan] == 0:
                    Npeaks_fit = int(len(self.fitted_params[i_adc][j_chan])/self.Nparams_peak)
                    peak_diffs = np.array([self.fitted_params[i_adc][j_chan][k_peak*self.Nparams_peak+1]-self.fitted_params[i_adc][j_chan][(k_peak+1)*self.Nparams_peak+1] for k_peak in range(Npeaks_fit-1)])
                    if (mode in ['amplitude', 'amplitude_peaks']):
                        self.gains[i_adc][j_chan] = abs(np.mean(peak_diffs))
                        self.gains_std[i_adc][j_chan] = np.std(peak_diffs)
                    else : 
                        self.gains[i_adc][j_chan] = abs(np.mean(peak_diffs[1:]))
                        self.gains_std[i_adc][j_chan] = np.mean(peak_diffs[1:])
                else:
                    logger.debug(f'Skipped ADC {i_adc}, chan. {j_chan} due to failed fitting')

        return None

    def save_fitted_plots(self, mode='default', fitting_mode=None, pedestals=None):
        if (mode=='default'):
            for i_adc in range(N_ADC):
                for j_chan in range(N_chan_ADC):
                    plot_fingerplot(*self.fingerplots[i_adc][j_chan], 
                                    title=f'Fingerplot of ADC {i_adc}, chan. {j_chan}', output_folder=self.output_path,
                                    plot_name=f'fingerplot_ADC{i_adc}_chan{j_chan}.png')
                    
        elif (mode =='gain'):
            for i_adc in range(N_ADC):
                for j_chan in range(N_chan_ADC):
                    if self.fit_status[i_adc][j_chan] == 0:
                        if (fitting_mode is not None):
                            title=f"Fingerplot of ADC {i_adc}, chan. {j_chan} with Gain, '{fitting_mode}' method"
                        else: 
                            title=f'Fingerplot of ADC {i_adc}, chan. {j_chan} with Gain'
                        plot_fingerplot(*self.fingerplots[i_adc][j_chan], title=title,
                                        output_folder=self.output_path, plot_name=f'fingerplot_wGain_ADC{i_adc}_chan{j_chan}.png',
                                        fit_params=self.fitted_params[i_adc][j_chan], Nparams_peak = self.Nparams_peak,
                                        fit_xlim=self.fit_lim[i_adc][j_chan], reduced_chi_squared=self.reduced_chi_squared[i_adc][j_chan],
                                        gain=self.gains[i_adc][j_chan])
                    else: 
                        title=f"Fingerplot of ADC {i_adc}, chan. {j_chan} with p0s (failed fit), '{fitting_mode}' method"
                        plot_fingerplot(*self.fingerplots[i_adc][j_chan], title=title,
                                        output_folder=self.output_path, plot_name=f'fingerplot_wp0Fail_ADC{i_adc}_chan{j_chan}.png',
                                        fit_params=self.fitted_params[i_adc][j_chan], Nparams_peak = self.Nparams_peak,
                                        fit_xlim=self.fit_lim[i_adc][j_chan])
        elif (mode == 'pe'):
            for i_adc in range(N_ADC):
                for j_chan in range(N_chan_ADC):
                    if self.fit_status[i_adc][j_chan] == 0:
                        if (fitting_mode is not None):
                            title=f"Fingerplot of ADC {i_adc}, chan. {j_chan} with Gain and PEs, '{fitting_mode}' method"
                        else: 
                            title=f'Fingerplot of ADC {i_adc}, chan. {j_chan} with Gain and PEs'
                        plot_fingerplot(*self.fingerplots[i_adc][j_chan],
                                        title=title,
                                        output_folder=self.output_path,
                                        plot_name=f'fingerplot_wPE_ADC{i_adc}_chan{j_chan}.png',
                                        fit_params=self.fitted_params[i_adc][j_chan], Nparams_peak = self.Nparams_peak,
                                        fit_xlim=self.fit_lim[i_adc][j_chan], reduced_chi_squared=self.reduced_chi_squared[i_adc][j_chan],
                                        gain=self.gains[i_adc][j_chan], pedestal=pedestals[i_adc][j_chan], nPEs=self.nPEs[i_adc][j_chan])

                    else: 
                        title=f"Fingerplot of ADC {i_adc}, chan. {j_chan} with p0s (failed fit), '{fitting_mode}' method"
                        plot_fingerplot(*self.fingerplots[i_adc][j_chan], title=title,
                                        output_folder=self.output_path, plot_name=f'fingerplot_wp0Fail_ADC{i_adc}_chan{j_chan}.png',
                                        fit_params=self.fitted_params[i_adc][j_chan], Nparams_peak = self.Nparams_peak,
                                        fit_xlim=self.fit_lim[i_adc][j_chan])
        return None


    def extract_gains(self, Nevent = -1, int_window=[0,-1], Nbins=150, mode='integral', cut=None, save_plots=False, show_p0_plots=False):
        """
        Compute the gain from a calib run

        Args:
            Nevent (int):       Number of event includes in the computation. (default: -1, all events)

        Return:
            None
        """

        logger.info(f"Computing the gains of the file {self.file} with {Nevent} events")
        logger.debug(f"The light waveforms variable have the shape {self.light_wvfms.shape}")

        if Nevent == -1:
            Nevent = self.light_wvfms.shape[0]
        else:
            Nevent = np.min([Nevent, self.light_wvfms.shape[0]])


        # Compute the fingerplots
        self.compute_fingerplots(Nevent = Nevent, int_window=int_window, Nbins=Nbins, mode=mode, cut=cut)

        # Fit the fingerplots
        self.fit_fingerplots(show_p0_plots)
            
        # Compute the gain
        self.compute_gains(mode=mode)
        
        # Save the files
        if (save_plots == True):
            self.save_fitted_plots(mode='gain', fitting_mode=mode)
            logger.debug(f"Gain plots stored in {self.output_path}")
        
        logger.debug(f"Gains computed")

        return None
    
    def compute_PEs(self, pedestals):
        
        logger.info(f"Computing the PEs")
    
        self.nPEs = np.empty((N_ADC, N_chan_ADC), dtype=object)
        for i_adc in range(N_ADC):
            for j_chan in range(N_chan_ADC):
                if self.fit_status[i_adc][j_chan] == 0:
                    Npeaks_fit = int(len(self.fitted_params[i_adc][j_chan])/self.Nparams_peak)
                    self.nPEs[i_adc][j_chan]=np.empty((Npeaks_fit))
                    for k_peak in range(Npeaks_fit):
                        self.nPEs[i_adc][j_chan][k_peak] = abs((self.fitted_params[i_adc][j_chan][k_peak*self.Nparams_peak+1]-pedestals[i_adc][j_chan])/self.gains[i_adc][j_chan])
                else:
                    logger.debug(f"The PEs were not computed for ADC {i_adc}, chan. {j_chan} due to a failed fit")
        
        logger.debug(f"The PEs were computed")
        return None

    def extract_PEs(self, baselines, Nevent = -1, int_window=[100,120], Nbins=150, mode='integral', cut=None, save_plots=False, show_p0_plots=False):
        # Extract the gains

        self.extract_gains(Nevent=Nevent, int_window=int_window, Nbins=Nbins, mode=mode, cut=cut, save_plots=False, show_p0_plots=show_p0_plots)
        
        # Compute the PE
        if (mode == 'amplitude'):
            pedestals = baselines
        elif (mode == 'integral' and int_window is not None):
            pedestals = baselines*(int_window[1]-int_window[0])

        self.compute_PEs(pedestals=pedestals)

        # Save the files
        if (save_plots == True):
            self.save_fitted_plots(mode='pe', fitting_mode=mode, pedestals=pedestals)
            logger.debug(f"PE plots stored in {self.output_path}")

        logger.info(f'{np.sum(self.fit_status)} fits failed ({np.sum(self.fit_status)*100/self.fit_status.size:.1f}%)')

        return None

    def compute_SNR(self, baselines, baselines_std):
        logger.debug('Start the computation of SNR')
        self.SNRs = np.zeros((N_ADC,N_chan_ADC))
        for i_adc in range(N_ADC):
            for j_chan in range(N_chan_ADC):
                if (self.fit_status[i_adc, j_chan] == 0):
                    A_1stPeak = self.fitted_params[i_adc, j_chan][1]
                    sig_1stPeak = self.fitted_params[i_adc, j_chan][2]

                    baseline = baselines[i_adc, j_chan]
                    sig_baseline = baselines_std[i_adc, j_chan]

                    # print(f'Amplitude peak: {A_1stPeak} sig {sig_1stPeak}')
                    # print(f'Baseline: {baseline} sig {sig_baseline}')

                    self.SNRs[i_adc, j_chan] = (A_1stPeak-baseline)/(np.sqrt(sig_1stPeak**2+sig_baseline**2)) 
                    # print(f'SNR of adc {i_adc}, chan. {j_chan}: {SNR}')
                else:
                    self.SNRs[i_adc, j_chan] = 0

        logger.debug('The SNR was computed')
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
    parser.add_argument("-ll", "--log_level",type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 
                                                                                'CRITICAL'], help="Log level")
    
    return parser.parse_args()

def plot_fingerplot(counts, bins, title=None, show=False, output_folder=None, plot_name=None, plot_xlim=None, fit_params=None, Nparams_peak=6, fit_xlim=None, reduced_chi_squared=None, gain=None, pedestal=None, nPEs=None):
    """
        Plot the finger plot.
    
    Args:
        counts (np.array):      The values of the histogram (see np.histogram docs)
        bins (np.array):        The bin edges of the histogram (see np.histogram docs)
        # mode (str):             Mode of the plot
        #     - 'default':            Plot the fingerplot 
        #     - 'fit'    :            Plot the fingerplot with the fit function passed by fit_params
        #     - 'gain' :              Plot the gain additionally to the fit function  
        # title (str):            Title of the plot, overwrite the default title
        show (bool):            Show the plot
        output_folder (str):    Output folder, if None the plot is not saved 
        fit_params (np.array):  Parameters of the multi_gaussian function to plot the fitting function. Required 
                                in mode 'fit'
        plot_xlim (list):       X-axis limits of the plot


    Return:
        None
    """
    logger.debug(f'Plotting the finger plot')

    fig = plt.figure(figsize=[10, 6])
    ax = fig.subplots()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]

    ax.bar(bin_centers, counts, width=width, color='skyblue', label=f'Fingerplot with {np.sum(counts)} entries', zorder= 5)

    default_title= 'Fingerplot'
    if (fit_params is not None and fit_xlim is not None):
        Npeaks = int(len(fit_params)/Nparams_peak)
        
        x_fit = np.linspace(bin_centers[fit_xlim[0]], bin_centers[fit_xlim[1]], 1000)

        ax.plot(x_fit, multi_gaussian(x_fit, *fit_params), color='r', ls='-', label='Fitted function (multi-gaussian)', zorder= 10)
        
        default_title= 'Fingerplot with a multigaussian fit function'

        for i_peak in range(Npeaks):
            ax.plot([], [],'', label=f"Peak {i_peak + 1}: $\mu$ = {fit_params[i_peak*Nparams_peak+1]:.1f}, $\sigma$ = {fit_params[i_peak*Nparams_peak+2]:.1f}", color="None", zorder= 20)
            
        if (gain != None):
            if (reduced_chi_squared is not None):
                ax.plot([], [], '', label=f"Gain: {gain:.1f}, $red. \chi^2$ = {reduced_chi_squared[0]:.1f}/{reduced_chi_squared[1]}", color="None", zorder= 30)

            else:
                ax.plot([], [], '', label=f"Gain: {gain:.1f}", color="None", zorder= 30)

            if (nPEs is not None):
                y_coord_text = []
                for i_peak in range(len(nPEs)):
                    y_coord_text.append(multi_gaussian(fit_params[i_peak*Nparams_peak+1], *fit_params)+15)
                    ax.text(fit_params[i_peak*Nparams_peak+1], y_coord_text[i_peak] , f"{int(np.round(nPEs[i_peak],))} PE", 
                            fontsize=10, ha='center', va='bottom', zorder= 20)

                ax.set_ylim([0, np.max(y_coord_text)+20])

            if (pedestal is not None):
                ax.vlines(x=pedestal, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], label=f"Pedestal: {pedestal:.1f}", color="green", ls='--', zorder= 30)

        ax.vlines(x=bin_centers[fit_xlim], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color = "C1", label='Fitting bounds', zorder= 10)


        ax.add_patch(Rectangle((x_fit[0],ax.get_ylim()[0]), abs(x_fit[-1]-x_fit[0]), ax.get_ylim()[1]-ax.get_ylim()[0], color='C1', alpha=0.1, zorder= 0))

    if (title==None):
        ax.set_title(default_title, fontsize=8)
    else:
        ax.set_title(title, fontsize=8)

    if np.all(plot_xlim):
        ax.set_xlim(plot_xlim)

    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels (alphabetic order)
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels)

    ax.set_xlabel('ADC counts')
    ax.set_ylabel('Number of entries')
    ax.grid(True)


    if (show==True):
        fig.show()
    
    if (output_folder != None):
        date_str = datetime.today().strftime('%Y%m%d')
        if (plot_name != None):
            output_plot = os.path.join(output_folder, f'{date_str}_{plot_name}')
        else:
            output_plot = os.path.join(output_folder ,f'{date_str}_fingerplot.png')
    
        fig.savefig(output_plot)
        logger.info(f'File {os.path.basename(output_plot)} saved in {os.path.dirname(output_plot)}')

    return None


def multi_gaussian(x, *params):
    """
    Compute the sum of multiple Gaussians.
    Each Gaussian has 3 parameters: amplitude, mean, std_dev.
    
    Args:
        x:          input array
        *params:    variable length parameters [A1, mu1, sigma1, A2, mu2, sigma2, ..., An, mun, sigman]
        
    Return:
        y:          sum of Gaussians evaluated at x
    """
    y = np.zeros_like(x, dtype=float)
    num_gaussians = len(params) // 3
    
    for i in range(num_gaussians):
        A = params[3*i]
        mu = params[3*i + 1]
        sigma = params[3*i + 2]

        y += A * np.exp(-((x - mu)**2) / (2 * sigma**2))
        
    return y


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