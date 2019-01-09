import argparse
from os.path import isdir, isfile, join
from os import listdir
from pickle import dump, load

import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.utils import np_utils

from midi import Polyphonic_pianoroll, parse_midi, save_midi

# inference: --mode inference --midi_input all_files/Piano_tuxguitar/Beethoven_sonata14_mond_1_format0.mid --weights_path model_guitar.hdf5
# training: --mode training --midi_input all_files/Guitar_tuxguitar/


class Net:
    """Class representing neural network."""
    def __init__(self, min_dur, batch_size, epochs, gen_len, gen_num, seq_len=16):
        self.min_dur = min_dur
        self.batch_size = batch_size
        self.epochs = epochs
        self.gen_len = gen_len
        self.gen_num = gen_num
        self.seq_len = seq_len
        self.x = None
        self.y = None
        self.data_x = None
        self.data_y = None
        self.song = None
        self.pianoroll = None
        self.chord_to_int = None
        self.int_to_chord = None
        self.model = None
        # todo: we need shape that will be good for training and INFERENCE
        # todo: something like (A_const, B_const, N_variable)
        # todo: there is something like this in files from Kurowski
        self.shape_tuple = None

    def _preprocess_midi_inference(self, midi_filename, seq_length=16):
        tracks = parse_midi(midi_filename, self.min_dur)

        self.song = Polyphonic_pianoroll(tracks)
        self.pianoroll = list(map(tuple, self.song.polyphonic_pianoroll))  # tuple is hashable

        chords = sorted(list(set(self.pianoroll)))
        # chord_to_int = dict((c, i) for i, c in enumerate(chords))
        #
        # self.int_to_chord = dict((i, c) for i, c in enumerate(chords))

        self.data_x = []
        self.data_y = []
        for i in range(0, len(self.pianoroll) - seq_length, 1):
            seq_in = self.pianoroll[i:i + seq_length]
            seq_out = self.pianoroll[i + seq_length]
            self.data_x.append([self.chord_to_int[char] for char in seq_in])
            self.data_y.append(self.chord_to_int[seq_out])

        self.x = numpy.reshape(self.data_x, (len(self.data_x), self.seq_len, 1))
        self.x = self.x / float(len(chords))
        self.y = np_utils.to_categorical(self.data_y, num_classes=len(self.chord_to_int.keys()))

    def _preprocess_midi_training(self, midi_dir):

        self.data_x = []
        self.data_y = []

        # todo: sklejac chordsy
        for filename in listdir(midi_dir):
            if filename.endswith('.mid'):

                tracks = parse_midi(join(midi_dir, filename), self.min_dur)

                self.song = Polyphonic_pianoroll(tracks)
                self.pianoroll = list(map(tuple, self.song.polyphonic_pianoroll))  # tuple is hashable

                chords = sorted(list(set(self.pianoroll)))

                # for i, chord in enumerate(chords):
                #     if chord not in self.int_to_chord.values():
                #         self.chord_to_int[chord] = i
                #         self.int_to_chord[i] = chord

                # chord_to_int = dict((c, i) for i, c in enumerate(chords))

                # self.int_to_chord = dict((i, c) for i, c in enumerate(chords))

                for i in range(0, len(self.pianoroll) - self.seq_len, 1):
                    seq_in = self.pianoroll[i:i + self.seq_len]
                    seq_out = self.pianoroll[i + self.seq_len]
                    self.data_x.append([self.chord_to_int[char] for char in seq_in])
                    self.data_y.append(self.chord_to_int[seq_out])

        # todo: shape must be as specified above, working both for training and inference
        # todo: this solution is BAD, it can worki only for training
        # if not self.shape_tuple:
        #     self.shape_tuple = (len(self.data_x), self.seq_len, 1)
        # self.x = numpy.reshape(self.data_x, self.shape_tuple)
        self.x = numpy.reshape(self.data_x, (len(self.data_x), self.seq_len, 1))
        self.x = self.x / float(len(chords))
        self.y = np_utils.to_categorical(self.data_y, num_classes=len(self.chord_to_int.keys()))

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(self.batch_size, input_shape=(self.seq_len, 1)))
        model.add(Dropout(0.2)) # maybe parametrize?
        model.add(Dense(self.y.shape[1], activation='softmax'))
        self.model = model

    # def _serialize_mappings(self):
    #     for filename, obj in zip(("int_to_chord.pkl", "chord_to_int.pkl"), (self.int_to_chord, self.chord_to_int)):
    #         with open(filename, 'wb') as f:
    #             dump(obj, f)

    def _load_mappings(self):
        for i, filename in enumerate(("int_to_chord.pkl", "chord_to_int.pkl")):
            with open(filename, 'rb') as f:
                if i == 0:
                    self.int_to_chord = load(f)
                elif i == 1:
                    self.chord_to_int = load(f)
                else:
                    raise Exception

    def run_training(self, midi_dir):
        self._load_mappings()
        self._preprocess_midi_training(midi_dir)
        self._build_model()
        # self._serialize_mappings()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        filepath = "md=%s-bs=%s-improvement-{epoch:02d}-{loss:.4f}.hdf5" % (self.min_dur, self.batch_size)
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(self.x, self.y, epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks_list)

    def run_inference(self, midi_filename, weights_path):
        self._load_mappings()
        self._preprocess_midi_inference(midi_filename)
        self._build_model()
        self.model.load_weights(weights_path)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

        for n in range(self.gen_num):
            pattern = self.data_x[numpy.random.randint(0, len(self.data_x) - 1)]
            self.song.polyphonic_pianoroll = []

            for i in range(self.gen_len):
                x = numpy.reshape(pattern, (1, len(pattern), 1))
                x = x / float(len(self.pianoroll))
                prediction = self.model.predict(x, verbose=0)
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
    parser.add_argument("--midi_input", type=str, required=True,
                        help="Name of input MIDI file (inference) or dir with midi files (training).")
    parser.add_argument("--min_dur", type=int, default=40,
                        help="Min dur value")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Size of training batch")
    parser.add_argument("--epochs", type=int, default=5,
                        help="echo the string you use here")
    parser.add_argument("--gen_len", type=int, default=100,
                        help="echo the string you use here")
    parser.add_argument("--gen_num", type=int, default=5,
                        help="echo the string you use here")
    parser.add_argument("--weights_path", type=str,
                        help="echo the string you use here")
    args = parser.parse_args()

    net = Net(args.min_dur, args.batch_size, args.epochs, args.gen_len, args.gen_num)

    if args.mode == "training":
        assert isdir(args.midi_input)
        net.run_training(args.midi_input)
    elif args.mode == "inference":
        assert isfile(args.midi_input)
        assert args.weights_path, "Need to specify weights path.gen_num"
        net.run_inference(args.midi_input, args.weights_path)
    else:
        raise Exception("Wrong mode, must be either training or inference.")
