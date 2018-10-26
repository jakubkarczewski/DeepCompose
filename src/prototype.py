import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import scipy.io.wavfile as wav
import scipy.signal as signal
import os
import sys

import keras
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

sys.path.append('../utils')
from midi_io import *

examples_directory = "music_examples"
examples = []

for file in os.listdir(examples_directory):
    if file.endswith(".mid"):
        m = MidiParser(examples_directory+"/"+file, 6)
        examples.append(m.output_piano_roll_representation)


def generate_sets(data_array, length_of_window):
    """Prepare training sets for LSTM."""
    length_of_sequence = data_array.shape[0]

    train_x = []
    train_y = []

    for i in range(0,length_of_sequence-length_of_window-1):
        train_x.append(data_array[i:i+length_of_window,:])
        train_y.append(data_array[i+length_of_window+1,:])

    return np.array(train_x), np.array(train_y)


def retrieve_one_hot(predicted_vector):
    """Retrieve one-hot encoding of vector form LSTM's output."""
    output_vector = np.zeros(predicted_vector.shape)
    output_vector[0,np.argmax(predicted_vector)] = 1
    return output_vector.astype(int).tolist()[0]


def generate_model(examples, learning_epochs_number, num_of_prev_samples):
    """Generate neural model."""

    train_x = []
    train_y = []

    for example in examples:
        number_of_inputs = len(example[0])
        example_train_x, example_train_y = generate_sets(np.array(example),
                                                         num_of_prev_samples)
        train_x.append(example_train_x)
        train_y.append(example_train_y)

    train_x = np.vstack(train_x)
    train_y = np.vstack(train_y)

    model = Sequential()
    model.add(LSTM(160, input_shape=(num_of_prev_samples,
                                     number_of_inputs), return_sequences=True))
    model.add(LSTM(160))
    model.add(Dense(number_of_inputs))

    optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999,
                                      epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    model.fit(train_x,train_y,epochs=learning_epochs_number,batch_size=128,
              verbose=2)

    return model


def generate_melody(examples, model, number_of_previous_samples, melody_length):
    # initialization of previous output state memory list
    example_init_index = np.random.randint(0,
                                           len(examples))

    previous_outputs_memory = []
    for i in range(0,number_of_previous_samples):
        previous_outputs_memory.append(examples[example_init_index][i])

    # generation stage
    output_vectors = []
    for i in range(0,melody_length):
        predicted_sample = retrieve_one_hot(model.predict(
            np.array([np.array(previous_outputs_memory)])))

        previous_outputs_memory.pop(0)
        previous_outputs_memory.append(predicted_sample)

        output_vectors.append(predicted_sample)

    return output_vectors

if __name__ == '__main__':
    # sample length
    number_of_previous_samples = 15

    # output lenghgt
    length_of_melody = 250

    learning_epochs_number = 50

    neural_network_model = generate_model(examples,learning_epochs_number,
                                          number_of_previous_samples)

    # neural_network_model.save("new_network_model.model")
    # name_of_file_with_model_to_be_imported = "100_ep.model"
    # neural_network_model = load_model(name_of_file_with_model_to_be_imported)

    N = 10
    for n in range(N):
        print('-------------------------------\n')
        print('generating melody number %i'%n)
        generated_melody = generate_melody(examples,
                                           neural_network_model,
                                           number_of_previous_samples,
                                           length_of_melody)

        # from one hot to notes
        melody_descriptor = condensate_pianoroll(generated_melody, 60, 2.5)

        # print notes
        for note in melody_descriptor: print(note)

        # save to midi
        output_file_name = 'output_examples/ann_composition_%s.mid' \
                           % str(n).zfill(2)
        save_melody_to_midi(melody_descriptor, output_file_name)
