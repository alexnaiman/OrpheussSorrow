from pretty_midi import PrettyMIDI, instrument_name_to_program, Instrument, Note
import numpy as np
import os
import pandas
from cv2 import imwrite
from math import ceil
from sklearn.preprocessing import normalize

# The range of notes we want to be is [0,96]
# and we want to make them equally distant to the limits of notes interval [0,127](-> [16, 111])
max_note_range = 96
note_difference = (128 - max_note_range) / 2
notes_per_bar = 96
number_of_measures = 16


class MidiUtils:
    SCORE_FILE = "../other_datasets/match_scores.json"
    LMD_MATCHED_DIR = "lmd_matched"
    DATASET_DIR = "../data"
    EMOTION_DATASET = None
    EMOTION_DATASET_LOCATION = '../my_dataset.csv'
    DEBUGGING = True

    @staticmethod
    def midi_to_input_vector(midi_filename):
        """
        Converts a midi_file to a midi piano roll matrix -> our type of input.
        :param midi_filename: string -> The name of the midi file we want to convert.
        :return: array of nparray representing our piano roll -> So, a matrix of type (X,96,96) where
                X is the number of piano rolls generated
        """
        midi_file = PrettyMIDI(midi_filename)
        beat = midi_file.resolution  # number of ticks in a beat
        measure = 4 * beat  # number of ticks in a measure
        notes = {}
        for instrument in midi_file.instruments:
            if False:  # instrument.is_drum:
                continue
            for note in instrument.notes:
                # get tick of note and map it to interval [0, 96]
                note_pitch = note.pitch - note_difference
                if not 0 <= note_pitch < max_note_range:
                    print("@@@@@@@@@@@@@@@")
                    print("NoteWarning: note %d outside interval; skipping note" % note_pitch)
                    print("@@@@@@@@@@@@@@@")
                    continue
                start_tick = midi_file.time_to_tick(note.start)
                start_pixel_time = start_tick * notes_per_bar // measure
                end_pixel_time = start_pixel_time + 1  # make the note end after one tick
                if note_pitch not in notes:
                    notes[int(note_pitch)] = [[start_pixel_time, end_pixel_time]]
                else:
                    notes[int(note_pitch)].append([start_pixel_time, end_pixel_time])

        data = []
        for note_pitch in notes:
            for start, end in notes[note_pitch]:
                sample_ix = int(start // notes_per_bar)
                while len(data) <= sample_ix:
                    data.append(np.zeros((notes_per_bar, max_note_range), dtype=np.uint8))
                sample = data[sample_ix]
                start_ix = int(start - sample_ix * notes_per_bar)
                # make the note 1 tick long
                # for easing the training we do not care at the moment about notes longer than 1 tick
                # so we only have two possibilities
                #  - 1, if the note is played at the
                #  - 0, otherwise
                sample[start_ix, note_pitch] = 1
        empty_piano_roll = np.zeros_like(data[0])
        # appending necessary empty piano rolls such that the list is
        required_number_piano_rolls = int(ceil(len(data) / number_of_measures)) * number_of_measures
        additional_empty_piano_rolls = (required_number_piano_rolls - len(data)) * [empty_piano_roll]
        data += additional_empty_piano_rolls
        return data

    @staticmethod
    def find_note_range_limits(piano_roll_matrices):
        """
        Returns the maximum and minimum note pitch
        :param piano_roll_matrices: the piano roll matrix of form (x,96,96)
        :return:
        """
        merged_matrix = np.zeros_like(piano_roll_matrices[0])
        for matrix in piano_roll_matrices:
            # merge all matrices doing max element wise
            merged_matrix = np.maximum(merged_matrix, matrix)
        # flatten the matrix
        merged_matrix = np.amax(merged_matrix, axis=0)
        min_limit = np.where(merged_matrix == 1)[0][0]
        max_limit = np.where(merged_matrix == 1)[0][-1]
        return min_limit, max_limit

    @staticmethod
    def augment_data_trough_note_shifting(msd, piano_roll, max_shift_value=6):
        """
        Augments an existing piano roll (midi_data) through note shifting. Increase and decrease each note pitch by
        max_shift_value/2
        :param msd: id of track to get the emotion input
        :param piano_roll: a piano roll -> matrix(x,96,96)
        :param max_shift_value: the maximum number we can shift with
        :return: tuple of form (augmented piano roll, length of the augmented piano roll, emotion data)
        """
        min_limit, max_limit = MidiUtils.find_note_range_limits(piano_roll)
        min_shift = -min(max_shift_value, min_limit)
        max_shift = min(max_shift_value, max_note_range - max_limit)
        augmented_midi = []
        augmented_midi_length = []
        augmented_emotional_value = MidiUtils.EMOTION_DATASET[MidiUtils.EMOTION_DATASET["MSD_track_id"] == msd][
            ['valence', 'arousal']].to_numpy()
        augmented_emotional_data = []

        for shift in range(min_shift, max_shift):
            for i in range(len(piano_roll)):
                midi_matrix = np.zeros_like(piano_roll[i])
                midi_matrix[:, min_limit + shift:max_limit + shift] = piano_roll[i][:, min_limit:max_limit]
                augmented_midi.append(midi_matrix)
            augmented_midi_length.append(len(piano_roll))

        # we want to add the emotion data for each piano_roll -> (valence, arousal) for each 16 samples of type (96,96)
        # emotional data length => total_length / 16 for each shift
        emotional_data_length = (len(piano_roll) // number_of_measures) * len(range(min_shift, max_shift))
        augmented_emotional_data.append([augmented_emotional_value[0]] * emotional_data_length)
        return augmented_midi, augmented_midi_length, augmented_emotional_data

    @staticmethod
    def dump_input_vector():
        """
        Dumps all the input data for our model to learn
        It dumps it to three files
        -   train_data.npy
        -   train_length.npy
        -   train_data_emotion.npy
        """
        MidiUtils.EMOTION_DATASET = pandas.read_csv(
            '../my_dataset.csv',
            usecols=[0, 1, 2])
        train_data = []
        train_data_length = []
        train_data_emotion = []
        my_break = False
        for msd in MidiUtils.EMOTION_DATASET["MSD_track_id"]:
            directory = os.path.join(MidiUtils.DATASET_DIR, msd)
            for root, directory, files in os.walk(directory):
                midi_files = list(filter(lambda f: f.endswith('.mid') or f.endswith('.midi'), files))
                for file in midi_files:
                    # file_path = os.path.join(root, file)
                    file_path = './test.midi'
                    try:
                        midi_data = MidiUtils.midi_to_input_vector(file_path)
                    except Exception as e:
                        print(e)
                        print("Path: %s" % file_path)
                        continue
                    augmented_data, augmented_data_length, augmented_emotion_data = MidiUtils.augment_data_trough_note_shifting(
                        msd, midi_data, 1)  # )
                    train_data += augmented_data
                    train_data_length += augmented_data_length
                    train_data_emotion += augmented_emotion_data
                    if MidiUtils.DEBUGGING:
                        my_break = True
            if my_break:
                break
        if not sum(train_data_length) == len(train_data):
            raise Exception("Error: Data corrupted")
        train_data = np.array(train_data, dtype=np.uint8)
        MidiUtils.get_full_piano_roll("./piano_rolls", train_data, 0.5)
        train_data_length = np.array(train_data_length, dtype=np.uint32)
        train_data_emotion = np.array(train_data_emotion[0], dtype=np.float)
        np.save('dumps/train_data.npy', train_data)
        np.save('dumps/train_length.npy', train_data_length)
        np.save('dumps/train_data_emotion.npy', train_data_emotion)

    @staticmethod
    def get_one_piano_roll(piano_roll_filename, midi_data, threshold=None):
        if threshold is not None:
            image_matrix = np.where(midi_data > threshold, 0, 1)
        else:
            image_matrix = 1.0 - midi_data
        imwrite(piano_roll_filename, image_matrix * 255)

    @staticmethod
    def get_full_piano_roll(directory, midi_data, threshold=None):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i in range(midi_data.shape[0]):
            filename = directory + "/piano_roll" + str(i) + ".png"
            MidiUtils.get_one_piano_roll(filename, midi_data[i], threshold)

    @staticmethod
    def input_vector_to_midi(piano_roll, midi_filename, thresh_hold=0.5):
        """
        Transforms a piano roll midii data to a MIDI file and writes it
        :param piano_roll: piano roll matrix of type (x,96,96)
        :param midi_filename: the filename for the midi
        :param thresh_hold: the value above which we accept a note as a valid one
        """
        midi = PrettyMIDI(resolution=480)
        piano_program = instrument_name_to_program("Acoustic Grand Piano")
        piano = Instrument(program=piano_program)
        beat = midi.resolution  # number of ticks in a beat
        measure = 4 * beat  # number of ticks in a measure
        note_ticks = measure / notes_per_bar  # AICI GANDESTE_TE LA FORMULA
        for i, matrix in enumerate(piano_roll):
            for y in range(matrix.shape[0]):
                for x in range(matrix.shape[1]):
                    note = x + (128 - max_note_range) // 2
                    if matrix[y, x] >= thresh_hold:
                        start = int((i * 96 + y + 1) * note_ticks)
                        start_time = midi.tick_to_time(start)
                        end = int(start + note_ticks * 4)
                        end_time = midi.tick_to_time(end)
                        note_midi = Note(127, note, start_time, end_time)
                        piano.notes.append(note_midi)
        midi.instruments.append(piano)
        midi.write(midi_filename)

    @staticmethod
    def normalize_data():
        emotion_data = np.load('./dumps/train_data_emotion.npy')

        emotion_data = np.concatenate((emotion_data, [[-3, -3], [3, 3]]))
        emotion_data = normalize(emotion_data, axis=0, norm='max')
        emotion_data = emotion_data[:-2, :]
        np.save("./dumps/train_emotion_normalized_data", emotion_data)

# MidiUtils.dump_input_vector()
# MidiUtils.normalize_data()
