from os import listdir
from os.path import join
from pickle import dump


from midi import Polyphonic_pianoroll, parse_midi, save_midi

all_chords = []
midi_dir = 'all_files/all_songs/'

for filename in listdir(midi_dir):
    if filename.endswith('.mid'):

        tracks = parse_midi(join(midi_dir, filename), 40)

        song = Polyphonic_pianoroll(tracks)
        pianoroll = list(map(tuple, song.polyphonic_pianoroll))  # tuple is hashable

        chords = sorted(list(set(pianoroll)))

        # for i, chord in enumerate(chords):
        #     if chord not in int_to_chord.values():
        #         index = i + len(int_to_chord.values())
        #         chord_to_int[chord] = index
        #         int_to_chord[index] = chord

        for chord in chords:
            if chord not in all_chords:
                all_chords.append(chord)

int_to_chord = {i: chord for i, chord in enumerate(all_chords)}
chord_to_int = {chord: i for i, chord in enumerate(all_chords)}

for filename, obj in zip(("int_to_chord.pkl", "chord_to_int.pkl"), (int_to_chord, chord_to_int)):
    with open(filename, 'wb') as f:
        dump(obj, f)
