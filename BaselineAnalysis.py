# Script written by Nicolas Sallin to perform some Baseline analysis
# A jupyter notebook, BaselineAnalysis.ipynb, present some possibilities
# Contact nicolas.sallin@unibe.ch or @Nicolas Sallin on Slack

import json
import os
import numpy as np


class BaselineFiles:
    """
    
    """

    def __init__(self, input_json, output_path=None):  #, input_folderPath=None):

        if input_json != None:
            # Load from JSON file
            with open(input_json, 'r') as f_json:
                self.input_files_dict = json.load(f_json)

        # Set the output path
        if (output_path == None):
            self.output_path = os.path.join(os.path.dirname(__file__) ,f'Baselines_{os.path.splitext(os.path.basename(input_json))[0]}/')
        else:
            self.output_path = os.path.abspath(output_path)
                

    def print_files(self):
        # for file in self.files_dict.keys():
        #     print(file)
        return None


    def get_baselines(self):
        for folder in self.input_files_dict.keys():
            for file in self.input_files_dict[folder]:
                # print('folder', folder)
                # print('file_name',file['file_name'])

                bl_wvfms = baselineWvfms(folder, file['file_name'], self.output_path)
                bl_wvfms.compute_baselines()
                print(bl_wvfms.baselines)

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
        if (output_path != None):
            self.output_path = os.path.abspath(output_path)
        else:
            self.output_path = os.path.join(os.path.dirname(__file__) ,f'evD_{self.filename}/')

        # Load light events, waveform datasets and light geometry info if using
        self.light_events = f['light/events/data']
        self.light_wvfms = f['light/wvfm/data']['samples']

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