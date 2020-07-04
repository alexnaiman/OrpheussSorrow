import random
import numpy as np
import midi_utils

import pandas

from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, TimeDistributed, concatenate
from keras.layers.normalization import BatchNormalization

from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras import backend as K
from keras import callbacks
import matplotlib.pyplot as plt
import pickle

from model_utils import SaveOnEpochCallback

EPOCHS = 1000
LEARNING_RATE = 0.001
CONTINUE_TRAIN = False
PLAY_ONLY = False
DROP_OUT_RATE = 0.1
BATCH_MOMENTUM = 0.9
BATCH_SIZE = 350
NUMBER_FEATURES = 48
JUST_PLAY = True
TRAIN_DATA = '../dumps/train_data.npy'
TRAIN_DATA_LENGTH = '../dumps/train_length.npy'
TRAIN_EMOTION_DATA = '../dumps/train_emotion_normalized_data.npy'
TEST_DATA = '../dumps/test_data.npy'
TEST_EMOTION_DATA = '../dumps/test_emotion_normalized_data.npy'

MODEL = "190"

np.random.seed(0)
random.seed(0)


class OrpheusModel:

    def __init__(self):
        """
        Helper class that creates, loads, and trains model containing also other utils
        """
        self.samples = None
        self.test_samples = None
        self.emotion_data = None
        self.test_emotion_data = None
        self.model = None
        self.get_predicted_song = None
        self.encoder = None
        self.mean = None
        self.eigen_values = None
        self.eigen_vectors = None

    def load_data(self):
        """
        Function that loads our training and validation data
        """
        print("...getting data ready")
        self.samples = np.load(TRAIN_DATA)
        self.emotion_data = np.load(TRAIN_EMOTION_DATA)
        print("...getting validation data ready")
        self.test_samples = np.load(TEST_DATA)
        self.test_emotion_data = np.load(TEST_EMOTION_DATA)
        print("...reshaping our data")
        self.samples = np.reshape(self.samples, (
            self.samples.shape[0] // midi_utils.NUMBER_OF_MEASURES, midi_utils.NUMBER_OF_MEASURES,
            self.samples.shape[1],
            self.samples.shape[2]))
        self.test_samples = np.reshape(self.test_samples, (
            self.test_samples.shape[0] // midi_utils.NUMBER_OF_MEASURES, midi_utils.NUMBER_OF_MEASURES,
            self.test_samples.shape[1],
            self.test_samples.shape[2]))

    def train(self):
        """
        Starts training the model
        :return:
        """
        # self.load_data()
        saver = SaveOnEpochCallback()
        csv_logger = callbacks.CSVLogger('training.log')
        history = self.model.fit([self.samples, self.emotion_data], self.samples, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                 callbacks=[saver, csv_logger],
                                 validation_data=([self.test_samples, self.test_emotion_data], self.test_samples))
        with open('../metadata/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    def create_model(self, should_plot=False):
        """
        Creates and compiles our model
        :param should_plot: if true, will plot the structure of the model
        """
        shape = list(self.samples.shape)
        print("...creating model")

        layer_input = Input(shape=(shape[1:]))
        print(K.int_shape(layer_input))

        layer = Reshape((midi_utils.NUMBER_OF_MEASURES, -1))(layer_input)
        print(K.int_shape(layer))

        layer = TimeDistributed(Dense(1844, activation='relu'))(layer)
        print(K.int_shape(layer))

        layer = TimeDistributed(Dense(369, activation='relu'))(layer)
        print(K.int_shape(layer))

        layer = Flatten()(layer)
        print(K.int_shape(layer))

        layer = Dense(1181, activation='relu')(layer)
        print(K.int_shape(layer))

        layer = Dense(236, activation='relu')(layer)
        print(K.int_shape(layer))

        layer = Dense(NUMBER_FEATURES, activation='relu')(layer)
        layer = BatchNormalization(momentum=BATCH_MOMENTUM)(layer)
        print(K.int_shape(layer))

        emotion_input_layer = Input(shape=[2])
        print(K.int_shape(emotion_input_layer))

        emotion_layer = BatchNormalization(
            momentum=BATCH_MOMENTUM)(emotion_input_layer)
        print(K.int_shape(emotion_input_layer))

        merged_layer = concatenate([layer, emotion_layer], name='pre_encoder')
        print(K.int_shape(merged_layer))

        layer = Dense(236, name='encoder')(merged_layer)
        layer = BatchNormalization(momentum=BATCH_MOMENTUM)(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(DROP_OUT_RATE)(layer)
        print(K.int_shape(layer))

        layer = Dense(1181, name='encoder')(merged_layer)
        layer = BatchNormalization(momentum=BATCH_MOMENTUM)(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(DROP_OUT_RATE)(layer)
        print(K.int_shape(layer))

        layer = Dense(midi_utils.NUMBER_OF_MEASURES * 369)(layer)
        print(K.int_shape(layer))

        layer = Reshape((midi_utils.NUMBER_OF_MEASURES, 369))(layer)
        print(K.int_shape(layer))

        layer = TimeDistributed(BatchNormalization(
            momentum=BATCH_MOMENTUM))(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(DROP_OUT_RATE)(layer)

        layer = TimeDistributed(Dense(1844))(layer)
        layer = TimeDistributed(BatchNormalization(
            momentum=BATCH_MOMENTUM))(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(DROP_OUT_RATE)(layer)
        print(K.int_shape(layer))

        layer = TimeDistributed(
            Dense(shape[2] * shape[3], activation='sigmoid'))(layer)
        print(K.int_shape(layer))

        layer = Reshape((midi_utils.NUMBER_OF_MEASURES,
                         shape[2], shape[3]))(layer)
        print(K.int_shape(layer))

        model = Model([layer_input, emotion_input_layer], layer)
        model.compile(RMSprop(lr=LEARNING_RATE),
                      loss="binary_crossentropy", metrics=["accuracy"])

        self.model = model
        if should_plot:
            plot_model(model, to_file='orpheus_model.png',
                       show_shapes=True, dpi=48)
            plot_model(model, to_file='orpheus_model2.png',
                       show_shapes=True, dpi=192)

    def load_model(self, path='../dumped_models/model_{}.hd5'.format(MODEL)):
        """
        Loads a pre-trained model
        :param path: location of the model
        """
        self.model = load_model(path)

        self.get_predicted_song = K.function([self.model.get_layer('pre_encoder').input, K.learning_phase()],
                                             [self.model.layers[-1].output])
        self.encoder = Model(inputs=self.model.input, outputs=self.model.get_layer(
            'batch_normalization_1').output)

    def get_random_song_roll(self, emotion_value=np.array([[-10, 10]])):
        """
        Generates a song with given emotion values
        :param emotion_value: nparray of form [1,2]
        :return: array of piano rolls of shape (16, 96, 96)
        """
        normal_distributed_vector = [
            np.random.normal(0.0, 1.0, (1, NUMBER_FEATURES))]

        song = self.get_predicted_song(
            [[[normal_distributed_vector], [emotion_value]], 0])[0]

        piano_roll = np.reshape(
            song, (song.shape[0] * song.shape[1], song.shape[2], song.shape[3]))

        return midi_utils.filter_out_midi(piano_roll).tolist()

    def get_normalized_data(self):
        """
        Saves some metadata related to our dataset using singular value decomposition
        Saves:
            - mean of our data features
            - eigen values of our data space
            - eigen ectors of our data space
        """
        self.load_data()
        self.load_model()

        encoded_data = np.squeeze(self.encoder.predict(
            [self.samples, self.emotion_data]))
        mean = np.mean(encoded_data, axis=0)
        covariance = np.cov((encoded_data - mean).T)
        u, s, eigen_vectors = np.linalg.svd(covariance)
        self.eigen_values = np.sqrt(s)
        self.eigen_vectors = eigen_vectors
        np.save('../metadata/{}/mean.npy'.format(MODEL), mean)
        np.save('../metadata/{}/eigen_values.npy'.format(MODEL), self.eigen_values)
        np.save('../metadata/{}/eigen_vectors.npy'.format(MODEL),
                self.eigen_vectors)

    def load_normalized_values(self):
        """
        Loads our metadata
        :return:
        """
        self.mean = np.load('../metadata/{}/mean.npy'.format(MODEL))
        self.eigen_values = np.load(
            '../metadata/{}/eigen_values.npy'.format(MODEL))
        self.eigen_vectors = np.load(
            '../metadata/{}/eigen_vectors.npy'.format(MODEL))

    def get_normalized_song_for_features(self, features):
        """
        Uses our metadata to obtain a normalized array of features for the given features
        :param features: array of features of type (1, NUMBER_OF_FEATURES)
        :return: normalized array of features (1, NUMBER_OF_FEATURES)
        """
        return self.mean + np.dot(features * self.eigen_values, self.eigen_vectors)

    def get_song_by_features(self, features, emotion_value=np.array([[-3, 3]]), thresh_hold=0.375, normalized=True):
        """
        Generates a song using the given features
        :param features: array of features (1, NUMBER_OF_FEATURES)
        :param emotion_value: emotion values [valence, arousal]
        :param thresh_hold: the threshold above a note should be to be rendered
        :param normalized: if we should normalize the feature array or not
        :return: a sparse piano roll -> array of type [roll, time, note] representing the "coordinates" of our notes
        """
        normalized_features = features
        if normalized:
            normalized_features = self.get_normalized_song_for_features(features)[
                0]
        song = self.get_predicted_song(
            [[[normalized_features], [emotion_value]], 0])[0]

        # used when debugging/testing the data
        # dumped_song = np.reshape(song, (song.shape[0] * song.shape[1], song.shape[2], song.shape[3]))
        # midi_utils.input_vector_to_midi(dumped_song, 'composed/test{}.mid'.format(0.25), thresh_hold=0.25)
        # midi_utils.input_vector_to_midi(dumped_song, 'composed/test{}.mid'.format(0.375), thresh_hold=0.375)
        # midi_utils.input_vector_to_midi(dumped_song, 'composed/test{}.mid'.format(0.5), thresh_hold=0.5)

        filtered_piano_roll = midi_utils.filter_out_midi(song, thresh_hold)
        _, roll, x, y = np.where(filtered_piano_roll > 0)
        sparse_piano_roll = list(zip(roll.tolist(), x.tolist(), y.tolist()))
        return sparse_piano_roll

    def get_random_song_midi(self, path='../composed/test{}.mid', emotion_value=np.array([[-3, 3]])):
        """
        Dumps a random song into a MIDI file
        :param path: the location of the MIDI
        :param emotion_value: emotion values -> [valence, arousal]
        """
        # normal_distributed_vector = [np.random.normal(0.0, 1.0, (1, NUMBER_FEATURES))]
        normal_distributed_vector = [np.full((1, NUMBER_FEATURES), 0.75)]

        song = self.get_predicted_song(
            [[[normal_distributed_vector], [emotion_value]], 0])[0]
        song = np.reshape(
            song, (song.shape[0] * song.shape[1], song.shape[2], song.shape[3]))

        midi_utils.input_vector_to_midi(
            song, path.format(0.25), thresh_hold=0.25)
        midi_utils.input_vector_to_midi(
            song, path.format(0.375), thresh_hold=0.375)
        midi_utils.input_vector_to_midi(
            song, path.format(0.5), thresh_hold=0.5)

    @staticmethod
    def plot_loss():
        """
        Plot loss of last trained model
        :return:
        """
        logs = pandas.read_csv('../metadata/trainingv2.log')  # short run data
        history = pickle.load(
            open('../metadata/trainHistoryDict', "rb"))  # long run ata

        fig, axes = plt.subplots(nrows=2, ncols=1)
        axes[0].set_title("Long run view on loss on our model")
        logs.plot(x="epoch", y=["loss", "val_loss"], ax=axes[0])
        axes[0].legend(["Train", "Test"])
        axes[0].set_ylabel("loss")
        axes[0].set_xlabel("epoch")

        axes[1].set_title("Zoomed in view on loss on our model")
        axes[1].plot(history['loss'])
        axes[1].plot(history['val_loss'])
        axes[1].set_ylabel('loss')
        axes[1].set_xlabel('epoch')
        axes[1].legend(['Train', 'Test'])
        plt.show()

    @staticmethod
    def plot_accuracy():
        history = pickle.load(
            open('../metadata/trainHistoryDict', "rb"))  # long run ata
        plt.title("View on accuracy on our model")
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Test'])
        plt.show()

    @staticmethod
    def parse_json_data(data):
        features = data['features']
        arousal = data['arousal']
        valence = data['valence']
        threshold = data['threshold']
        multiplier = data['multiplier']

        if multiplier is None:
            multiplier = 1

        print(threshold)
        print(valence)
        print(arousal)

        features = np.asarray([features], dtype=np.float32)
        features = features / 100
        features = [features]

        arousal = ((arousal - 50) / 50) * multiplier
        valence = ((valence - 50) / 50) * multiplier
        threshold = threshold / 100

        print("parsed")
        print(threshold)
        print(valence)
        print(arousal)

        emotion_data = np.array([[valence, arousal]])
        return features, emotion_data, threshold

    def evaluate_model(self):
        self.load_model()
        self.load_data()
        results = self.model.evaluate([self.test_samples, self.test_emotion_data], self.test_samples,
                                      batch_size=BATCH_SIZE)
        print(results)


if __name__ == '__main__':
    orpheus = OrpheusModel()
    orpheus.evaluate_model()
    # orpheus.load_data()
    # orpheus.create_model(should_plot=True)
    # orpheus.plot_loss()
    # orpheus.get_random_song_midi()
    # orpheus.get_normalized_data()
    # OrpheusModel.plot_loss()
    # OrpheusModel.plot_accuracy()
