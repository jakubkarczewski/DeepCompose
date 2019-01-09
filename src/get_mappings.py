import argparse
from os import listdir
from os.path import join
from pickle import dump


from midi import Polyphonic_pianoroll, parse_midi, save_midi


def main(training_dir, inference_path):
    all_chords = []

    for filename in listdir(training_dir):
        if filename.endswith('.mid'):

            tracks = parse_midi(join(training_dir, filename), 40)

            song = Polyphonic_pianoroll(tracks)
            pianoroll = list(map(tuple, song.polyphonic_pianoroll))  # tuple is hashable

            chords = sorted(list(set(pianoroll)))

            for chord in chords:
                if chord not in all_chords:
                    all_chords.append(chord)

    tracks = parse_midi(inference_path, 40)

    song = Polyphonic_pianoroll(tracks)
    pianoroll = list(map(tuple, song.polyphonic_pianoroll))  # tuple is hashable

    chords = sorted(list(set(pianoroll)))

    for chord in chords:
        if chord not in all_chords:
            all_chords.append(chord)

    int_to_chord = {i: chord for i, chord in enumerate(all_chords)}
    chord_to_int = {chord: i for i, chord in enumerate(all_chords)}

    for filename, obj in zip(("int_to_chord.pkl", "chord_to_int.pkl"), (int_to_chord, chord_to_int)):
        with open(filename, 'wb') as f:
            dump(obj, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_dir", type=str, required=True,
                        help="dir with training data.")
    parser.add_argument("--inference_path", type=str, required=True,
                        help="path to file 4 inference")
    args = parser.parse_args

