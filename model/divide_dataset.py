import os
import pandas as pd
import numpy as np

"""
    Script used for dividing the dataset in train and test data
"""

DATASET_LOCATION = '../metadata/all_data.csv'
MIDI_FILES_LOCATION = '../../new_data1/'
TESTING_DATA_FILE = '../metadata/testing_data_midi.csv'
VALIDATION_DATA_FILE = '../metadata/validation_data_midi.csv'

dataset_csv = pd.read_csv(
    DATASET_LOCATION,
    usecols=[0, 1, 2])

all_data = []

for root, directory, files in os.walk(MIDI_FILES_LOCATION):
    for midi_dir in directory:
        for midi_root, midi_directory, midi_files in os.walk(MIDI_FILES_LOCATION + midi_dir):
            all_data += midi_files

all_data_frame = pd.DataFrame(data=np.array(all_data), columns=['midi'])
testing_data = all_data_frame.sample(frac=0.2)
validation_data = testing_data.sample(frac=0.5)
testing_data.to_csv(index=False, path_or_buf=TESTING_DATA_FILE)
validation_data.to_csv(index=False, path_or_buf=VALIDATION_DATA_FILE)
