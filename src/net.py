import argparse

import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.utils import np_utils

from midi import Polyphonic_pianoroll, parse_midi, save_midi


class Net:
    """Class representing neural network."""
    def __init__(self, min_dur, batch_size, epochs, gen_len, gen_num):
        self.min_dur = min_dur
        self.batch_size = batch_size
        self.epochs = epochs
        self.gen_len = gen_len
        self.gen_num = gen_num
        self.x = None
        self.y = None
        self.data_x = None
        self.data_y = None
        self.song = None
        self.pianoroll = None
        self.int_to_chord = None
        self.model = None

    def _preprocess_midi(self, midi_filename, seq_length=16):
        tracks = parse_midi(midi_filename, self.min_dur)

        self.song = Polyphonic_pianoroll(tracks)
        self.pianoroll = list(map(tuple, self.song.polyphonic_pianoroll))  # tuple is hashable

        chords = sorted(list(set(self.pianoroll)))
        chord_to_int = dict((c, i) for i, c in enumerate(chords))

        self.int_to_chord = dict((i, c) for i, c in enumerate(chords))

        self.data_x = []
        self.data_y = []
        for i in range(0, len(self.pianoroll) - seq_length, 1):
            seq_in = self.pianoroll[i:i + seq_length]
            seq_out = self.pianoroll[i + seq_length]
            self.data_x.append([chord_to_int[char] for char in seq_in])
            self.data_y.append(chord_to_int[seq_out])

        self.x = numpy.reshape(self.data_x, (len(self.data_x), seq_length, 1))
        self.x = self.x / float(len(chords))
        self.y = np_utils.to_categorical(self.data_y)

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(self.batch_size, input_shape=(self.x.shape[1], self.x.shape[2])))
        model.add(Dropout(0.2)) # maybe parametrize?
        model.add(Dense(self.y.shape[1], activation='softmax'))
        self.model = model

    def run_training(self, midi_filename):
        # todo: here we need to learn from many files but this is for later
        self._preprocess_midi(midi_filename)
        self._build_model()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        filepath = "%s-md=%s-bs=%s-improvement-{epoch:02d}-{loss:.4f}.hdf5" % (midi_filename, self.min_dur,
                                                                               self.batch_size)
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(self.x, self.y, epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks_list)

    def run_inference(self, midi_filename, weights_path):
        self._preprocess_midi(midi_filename)
        self._build_model()
        self.model.load_weights(weights_path)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

        for n in range(self.gen_num):
            pattern = self.data_x[numpy.random.randint(0, len(self.data_x) - 1)]
            self.song.polyphonic_pianoroll = []

            for i in range(self.gen_len):
                x = numpy.reshape(pattern, (1, len(pattern), 1))
                x = x / float(len(self.pianoroll))
                prediction = self.model.predict(self.x, verbose=0)
                index = numpy.random.choice(len(prediction[0]), 1, p=prediction[0])
                result = self.int_to_chord[index[0]]
                self.song.polyphonic_pianoroll.append(result)
                pattern.append(index[0])
                pattern = pattern[1:len(pattern)]

            self.song.polyphonic_pianoroll = list(map(list, self.song.polyphonic_pianoroll))
            self.song.back_to_pianorolls()
            save_midi(self.song.pianorolls, 'generated-%s.mid' % n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        help="training or inference.")
    parser.add_argument("--midi_filename", type=str, required=True,
                        help="Name of input MIDI file.")
    parser.add_argument("--min_dur", type=int, default=40,
                        help="Min dur value")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Size of training batch")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="echo the string you use here")
    parser.add_argument("--gen_len", type=int, default=1000,
                        help="echo the string you use here")
    parser.add_argument("--gen_num", type=int, default=5,
                        help="echo the string you use here")
    parser.add_argument("--weights_path", type=str,
                        help="echo the string you use here")
    args = parser.parse_args()

    net = Net(args.min_dur, args.batch_size, args.epochs, args.gen_len, args.gen_num)

    if args.mode == "training":
        net.run_training(args.midi_filename)
    elif args.mode == "inference":
        assert args.weights_path, "Need to specify weights path.gen_num"
        net.run_inference(args.midi_filename, args.weights_path)
    else:
        raise Exception("Wrong mode, must be either training or inference.")
