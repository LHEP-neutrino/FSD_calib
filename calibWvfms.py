import sys
import subprocess

# Function to install missing packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure setuptools is installed to use pkg_resources
try:
    import pkg_resources
except ImportError:
    install('setuptools')
    import pkg_resources

# Ensure all non-standard packages are installed
required_packages = [
    'numpy', 'h5py', 'pandas', 'matplotlib'
]    
# , 'sqlalchemy', 'cmasher', 'IPython', 'PyMuPDF', 'pillow', 'uproot', 'h5flow', 'ipywidgets'
# ]

installed_packages = {pkg.key for pkg in pkg_resources.working_set}
missing_packages = [pkg for pkg in required_packages if pkg not in installed_packages]

if missing_packages:
    for package in missing_packages:
        install(package)

# Import modules
#import pymupdf
import numpy as np
#import pandas as pd
# from datetime import datetime
# import ipywidgets as widgets
#from io import BytesIO
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
#from src.proto_nd_flow.util.lut import LUT
# from h5flow.core import resources
# import itertools
#import math
import h5py
#import cmasher as cmr
#from IPython.display import display, clear_output
#import matplotlib as mpl
import matplotlib.pyplot as plt
#from matplotlib import cm, colors
#import matplotlib.image as mpimg
#from matplotlib.patches import Rectangle
#from matplotlib.colors import Normalize
#from PIL import Image
# from math import fabs
#from time import sleep
# import uproot
from scipy.optimize import curve_fit


class calibWvfms:
    ''' 
        Class to test and set up a calibration routine for the FSD waveforms

        Inputs to this class are as follows:

            - filedir          (str):   Path to input file
            - filename         (str):   Name of input flow file
            - ouput_path       (str):   Path where to save the figures, if None: save in LAr_evd/FSD_eventDisplay/ (default: None)

        Class methods:

            - dumpWvfms()           :   Dump the waveforms as png at the output path
            
    '''

    # Initialize the class
    def __init__(self, filedir, filename, output_path=None):
        
        # Open files
        f = h5py.File(filedir+filename, 'r')

        # Set general class-level variables from inputs
        self.filedir = filedir
        self.filename = filename
        
        # Set the output path
        if (output_path is None):
            self.output_path = os.path.join(os.path.dirname(__file__) ,f'evD_{self.filename}/')
        else:
            self.output_path = os.path.abspath(output_path)


        # self.run_info = f['run_info']
        # self.is_mc = self.run_info.attrs['is_mc']

        # Load light events, waveform datasets and light geometry info if using
        self.light_events = f['light/events/data']
        self.light_wvfms = f['light/wvfm/data']['samples']
        # self.light_event_wvfm_ref = f['light/events/ref']['light/wvfm']['ref']
        # self.light_event_wvfm_region = f['light/events/ref']['light/wvfm']['ref_region']

        # self.sipm_abs_pos = LUT.from_array(f["geometry_info/sipm_abs_pos"].attrs["meta"],f["geometry_info/sipm_abs_pos/data"])
        # self.sipm_rel_pos = LUT.from_array(f["geometry_info/sipm_rel_pos"].attrs["meta"],f["geometry_info/sipm_rel_pos/data"])
        # self.light_det_id = LUT.from_array(f["geometry_info/det_id"].attrs["meta"],f["geometry_info/det_id/data"])

        # self.all_sipm_pos = f["geometry_info/sipm_abs_pos/data"]["data"][1:]
        # self.sipm_unique_x = np.unique([pos[0] for pos in self.all_sipm_pos])
        # self.sipm_unique_z = np.unique([pos[2] for pos in self.all_sipm_pos])
        # self.sipm_unique_y = np.unique([pos[1] for pos in self.all_sipm_pos])

        self.Npeaks = np.zeros(self.light_wvfms.shape[:-1])

        
        # Defined some module properties
        self.N_sipm_side = int(60)
        self.N_side_tpc = 2
        self.N_tpc = 2
        self.N_sipm_lightModule = 6
        self.N_LCM_lightModule = 3
        self.time_tick = 16*10**-9 # [s]

        # Information about the selection:
        print(f'Processing file {filedir+filename}')
        print(f'The output path is set to {self.output_path}')
        print(f"Number of events in the selection: {len(self.light_events)}")

    def _extract_raiseEdge(self, wvfm, threshold):
        '''
        threshold in ADC unit
        '''
        start_idx = []
        # end

        Npt_beforeThresold = 2

        for ix in range(Npt_beforeThresold, len(wvfm)):
            if (wvfm[ix-1] < threshold and wvfm[ix-2] < threshold and wvfm[ix] >= threshold):
                start_idx.append(ix)

            # if (wvfm[ix-1] > threshold and wvfm[ix-2] > threshold and wvfm[ix] <= threshold):
        

        return start_idx
    
    def _extract_peak(self, wvfm, minWidth, verbose=False):
        '''
        minWidth: Minimal width of a peak (e.g. a 5 ticks peaks have two values lower than the peak summit on each side)
        '''
        peak_idx = []
        is_peak = True
        mean = np.mean(wvfm)

        Npt_peakSide = int(minWidth/2)


        for ix in range(Npt_peakSide+1, len(wvfm)-Npt_peakSide-1):
            for i in range(1, Npt_peakSide+1):
                if (wvfm[ix-i] > wvfm[ix] or wvfm[ix+i] > wvfm[ix]):
                    is_peak = False
                    break

            if (is_peak==True):
                peak_idx.append(ix)
            else:
                is_peak = True

        peak_idx = np.array(peak_idx)
        
        if (verbose==False):
           aboveMean_idx = np.where(wvfm[peak_idx]>mean)[0]
           peak_idx = peak_idx[aboveMean_idx]

        return peak_idx, mean 
    

    
    def findPeak_wvfms(self, event, adc, chan, minWidth=5, verbose=False):
        '''
        Find the number of peak and store it in self.Npeaks

        Args:
            
            
        Return:
            None
        '''

        peak_idx, *peak_info = self._extract_peak(self.light_wvfms[event][adc,chan], minWidth, verbose)
        
        if (verbose==True):
            print(f'{len(peak_idx)} peaks were found with a minWidth of {minWidth}.')

        # if (verbose==True):
        #     mean = peak_info[0]
        #     aboveMean_idx = peak_idx[np.where(self.light_wvfms[event][adc,chan][peak_idx] > mean)[0]]
        #     belowMean_idx = peak_idx[np.where(self.light_wvfms[event][adc,chan][peak_idx] <= mean)[0]]
        #     ax_wvfm.plot(aboveMean_idx, self.light_wvfms[event][adc,chan][aboveMean_idx], color='g', marker='x', ls='')
        #     ax_wvfm.plot(belowMean_idx, self.light_wvfms[event][adc,chan][belowMean_idx], color='r', marker='x', ls='')

        #     ax_wvfm.hlines(peak_info[0], xlim[0], xlim[1], color='r', ls='--')

        #     self.Npeaks[event][adc,chan]=len(aboveMean_idx)

        
        self.Npeaks[event][adc,chan]=len(peak_idx)

        return None


    def plot_wvfms(self, event, adc, chan, xlim = None, threshold=None, peakFinder=False, minWidth=5, baseline=None, verbose=False):
        '''
        Plot the light waveforms.

        Args:
            
            
        Return:
            None
        '''
        # Compute the x-axis coordinate
        x_ticks = np.arange(0, self.light_wvfms[0].shape[-1],  1)


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

        if (threshold != None):
            start_idx = self._extract_raiseEdge(self.light_wvfms[event][adc,chan], threshold) 
            ylim = ax_wvfm.get_ylim()
            print(f' There are {len(start_idx)} triggers with a threshold at {threshold}')
            for idx in start_idx:
                ax_wvfm.vlines(idx, ylim[0], ylim[1], color='k', ls='--')

            ax_wvfm.hlines(threshold, xlim[0], xlim[1], color='r', ls='--')

        if (peakFinder==True):
            peak_idx, *peak_info = self._extract_peak(self.light_wvfms[event][adc,chan], minWidth, verbose)
            

            if (verbose==True):
                mean = peak_info[0]
                aboveMean_idx = peak_idx[np.where(self.light_wvfms[event][adc,chan][peak_idx] > mean)[0]]
                belowMean_idx = peak_idx[np.where(self.light_wvfms[event][adc,chan][peak_idx] <= mean)[0]]
                ax_wvfm.plot(aboveMean_idx, self.light_wvfms[event][adc,chan][aboveMean_idx], color='g', marker='x', ls='')
                ax_wvfm.plot(belowMean_idx, self.light_wvfms[event][adc,chan][belowMean_idx], color='r', marker='x', ls='')

                ax_wvfm.hlines(peak_info[0], xlim[0], xlim[1], color='r', ls='--')

                self.Npeaks[event][adc,chan]=len(aboveMean_idx)

                print(f'{len(aboveMean_idx)} peaks were found above the mean with a minWidth of {minWidth} and {len(belowMean_idx)} were cut out.')

            else:
                ax_wvfm.plot(peak_idx, self.light_wvfms[event][adc,chan][peak_idx], color='r', marker='x', ls='')
                self.Npeaks[event][adc,chan]=len(peak_idx)
                print(f'{len(peak_idx)} peaks were found with a minWidth of {minWidth} above the mean.')

        if (baseline != None):
            ax_wvfm.hlines(baseline, xlim[0], xlim[1], color='k', ls='--', label='Baseline')

        return None
    
    def fingerPlot_Amp_wvfms(self, minWidth=5, Nbins=100):
        fig_fP_Amp = plt.figure()#figsize=[12.8, 4.8])
        ax_fP_Amp = fig_fP_Amp.subplots()
        ax_fP_Amp.set_xlabel('Peak height [ADC unit]')
        ax_fP_Amp.set_ylabel('Number of Peak')
        ax_fP_Amp.grid(True)


        amp = []

        for adc in range(1):#Nadc):
            # for chan in range(1):#Nchan):
            chan=0
            for event in range(180):
                peak_idx, *peak_info = self._extract_peak(self.light_wvfms[event][adc,chan], minWidth)
                for idx in peak_idx:
                    amp.append(self.light_wvfms[event][adc,chan][idx])

        hAmp, hAmp_bins = np.histogram(amp, bins=Nbins) 
        # Compute bin centers
        hAmp_bins_centers = (hAmp_bins[:-1] + hAmp_bins[1:]) / 2

        # Fit the function to the bin centers and counts
        p0=[700, -26900, 100, 300, -26800, 50, 150, -26600, 200]
        hAmp_popt, hAmp_pcov = curve_fit(_multigauss, hAmp_bins_centers, hAmp, p0=p0)

        ax_fP_Amp.stairs(hAmp, hAmp_bins, fill=True, zorder=5, label=f'ADC {0}, Chan. {10}')

        x_fit = np.linspace(hAmp_bins_centers.min(), hAmp_bins_centers.max(), 1000)
        ax_fP_Amp.plot(x_fit, _multigauss(x_fit, *hAmp_popt), 'r-', label='Fitted Function', zorder=10)
        print(hAmp_popt)

        ax_fP_Amp.legend()
        
        return None
    

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
    def __init__(self, filedir, filename, output_path=None):
        
        # Open files
        f = h5py.File(filedir+filename, 'r')

        # Set general class-level variables from inputs
        self.filedir = filedir
        self.filename = filename
        
        # Set the output path
        if (output_path is None):
            self.output_path = os.path.join(os.path.dirname(__file__) ,f'evD_{self.filename}/')
        else:
            self.output_path = os.path.abspath(output_path)


        # self.run_info = f['run_info']
        # self.is_mc = self.run_info.attrs['is_mc']

        # Load light events, waveform datasets and light geometry info if using
        self.light_events = f['light/events/data']
        self.light_wvfms = f['light/wvfm/data']['samples']
        # self.light_event_wvfm_ref = f['light/events/ref']['light/wvfm']['ref']
        # self.light_event_wvfm_region = f['light/events/ref']['light/wvfm']['ref_region']

        # self.sipm_abs_pos = LUT.from_array(f["geometry_info/sipm_abs_pos"].attrs["meta"],f["geometry_info/sipm_abs_pos/data"])
        # self.sipm_rel_pos = LUT.from_array(f["geometry_info/sipm_rel_pos"].attrs["meta"],f["geometry_info/sipm_rel_pos/data"])
        # self.light_det_id = LUT.from_array(f["geometry_info/det_id"].attrs["meta"],f["geometry_info/det_id/data"])

        # self.all_sipm_pos = f["geometry_info/sipm_abs_pos/data"]["data"][1:]
        # self.sipm_unique_x = np.unique([pos[0] for pos in self.all_sipm_pos])
        # self.sipm_unique_z = np.unique([pos[2] for pos in self.all_sipm_pos])
        # self.sipm_unique_y = np.unique([pos[1] for pos in self.all_sipm_pos])

        self.baselines = np.zeros(self.light_wvfms.shape[:-1])

        
        # Defined some module properties
        self.N_sipm_side = int(60)
        self.N_side_tpc = 2
        self.N_tpc = 2
        self.N_sipm_lightModule = 6
        self.N_LCM_lightModule = 3
        self.time_tick = 16*10**-9 # [s]

        # Information about the selection:
        print(f'Processing file {filedir+filename}')
        print(f'The output path is set to {self.output_path}')
        print(f"Number of events in the selection: {len(self.light_events)}")

    def compute_baselines(self):
        self.baselines = np.mean(self.light_wvfms, axis= -1)

        return None
    
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
    
def _gauss(x, norm, mu, sigma):
        return norm * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
def _multigauss(x, norm1, mu1, sig1, norm2, mu2, sig2, norm3, mu3, sig3):
    y = 0.
    y += _gauss(x, norm1, mu1, sig1)
    y += _gauss(x, norm2, mu2, sig2)
    y += _gauss(x, norm3, mu3, sig3)
    return y
