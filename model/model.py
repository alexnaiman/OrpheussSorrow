import sys, random, os
import numpy as np
from matplotlib import pyplot
import pydot
import cv2
from midi_utils import number_of_measures
from sklearn.preprocessing import normalize

from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, TimeDistributed, concatenate
from keras.layers.normalization import BatchNormalization

from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras import backend as K

# TODO: TRY RENAME THESE
from model_utils import SaveOnEpochCallback

EPOCHS = 2000
LEARNING_RATE = 0.001
CONTINUE_TRAIN = False
PLAY_ONLY = False
DROP_OUT_RATE = 0.1
BATCH_MOMENTUM = 0.9
BATCH_SIZE = 350
NUMBER_FEATURES = 120


class MidiModelUtils:
    # TODO: try to remake this function
    @staticmethod
    def plot_loss(values, file_name, above=True):
        pyplot.clf()
        axes = pyplot.gca()
        axes.yaxis.set_tick_position('both')
        axes.yaxis.grid(True)
        pyplot.plot(values)
        pyplot.ylim([0.0, 0.009])
        pyplot.xlabel('Epoch')
        loc = ('upper right' if above else 'lower right')
        pyplot.draw()
        pyplot.savefig(file_name)

    # TODO: try to remake this function
    @staticmethod
    def save_config(model):
        with open('config.txt', 'w') as f:
            f.write('LR:          ' + str(LEARNING_RATE) + '\n')
            f.write('BN_M:        ' + str(BATCH_MOMENTUM) + '\n')
            f.write('BATCH_SIZE:  ' + str(BATCH_SIZE) + '\n')
            f.write('DO_RATE:     ' + str(DROP_OUT_RATE) + '\n')
            # f.write('num_songs:   ' + str(NUMBER_RANDOM_Songs) + '\n')
            f.write('optimizer:   ' + type(model.optimizer).__name__ + '\n')

    @staticmethod
    def start_training():
        os.environ['KERAS_BACKEND'] = "theano"
        K.set_image_data_format('channels_first')
        np.random.seed(0)
        random.seed(0)

        print("Loading Data...")
        y_samples = np.load('./dumps/train_data.npy')
        y_lengths = np.load('./dumps/train_length.npy')
        y_emotion_data = np.load('./dumps/train_emotion_normalized_data.npy')

        print("Reshaping Data...")
        y_samples = np.reshape(y_samples, (
            y_samples.shape[0] // number_of_measures, number_of_measures, y_samples.shape[1], y_samples.shape[2]))
        # y_emotion_data = np.reshape(y_emotion_data,
        #                             (y_emotion_data.shape[0] // number_of_measures, number_of_measures, 2))

        num_samples = y_samples.shape[0] * y_samples.shape[1]
        num_songs = y_lengths.shape[0]
        print("Loaded " + str(num_samples) + " samples from " + str(num_songs) + " songs.")
        print(np.sum(y_lengths))
        assert (np.sum(y_lengths) == num_samples)

        shape = list(y_samples.shape)

        print("Creating Model")
        layer_input = Input(shape=(shape[1:]))
        print(K.int_shape(layer_input))

        layer = Reshape((number_of_measures, -1))(layer_input)
        print(K.int_shape(layer))

        layer = TimeDistributed(Dense(2000, activation='relu'))(layer)
        print(K.int_shape(layer))

        layer = TimeDistributed(Dense(200, activation='relu'))(layer)
        print(K.int_shape(layer))

        layer = Flatten()(layer)
        print(K.int_shape(layer))

        layer = Dense(1600, activation='relu')(layer)
        print(K.int_shape(layer))

        layer = Dense(NUMBER_FEATURES, activation='relu')(layer)
        layer = BatchNormalization(momentum=BATCH_MOMENTUM)(layer)
        print(K.int_shape(layer))

        emotion_input_layer = Input(shape=[2])
        print(K.int_shape(emotion_input_layer))

        emotion_layer = BatchNormalization(momentum=BATCH_MOMENTUM)(emotion_input_layer)
        print(K.int_shape(emotion_input_layer))

        merged_layer = concatenate([layer, emotion_layer], name='pre_encoder')
        print(K.int_shape(merged_layer))

        layer = Dense(1600, name='encoder')(merged_layer)
        layer = BatchNormalization(momentum=BATCH_MOMENTUM)(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(DROP_OUT_RATE)(layer)
        print(K.int_shape(layer))

        layer = Dense(number_of_measures * 200)(layer)
        print(K.int_shape(layer))

        layer = Reshape((number_of_measures, 200))(layer)
        print(K.int_shape(layer))

        layer = TimeDistributed(BatchNormalization(momentum=BATCH_MOMENTUM))(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(DROP_OUT_RATE)(layer)
        print(K.int_shape(layer))

        layer = TimeDistributed(Dense(shape[2] * shape[3], activation='sigmoid'))(layer)
        print(K.int_shape(layer))

        layer = Reshape((number_of_measures, shape[2], shape[3]))(layer)
        print(K.int_shape(layer))

        model = Model([layer_input, emotion_input_layer], layer)
        model.compile(RMSprop(lr=LEARNING_RATE), loss="binary_crossentropy")

        plot_model(model, to_file='model.png', show_shapes=True)

        get_predicted_song = K.function([model.get_layer('encoder').input, K.learning_phase()],
                                        [model.layers[-1].output])

        encoder = Model(inputs=model.input, outputs=model.get_layer('pre_encoder').output)

        saver = SaveOnEpochCallback()
        # random_vec =
        history = model.fit([y_samples, y_emotion_data], y_samples, batch_size=BATCH_SIZE, epochs=5, callbacks=[saver], )
        print("aici")


MidiModelUtils.start_training()
