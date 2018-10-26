import mido
from mido import Message, MidiFile, MidiTrack
import numpy as np


def save_melody_to_midi(notes_sequence, file_name):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(Message('program_change', program=1, time=0))

    for note in notes_sequence:
        track.append(Message('note_on', note=int(note.pitch), velocity=64,
                             time=0))
        track.append(Message('note_off', note=int(note.pitch), velocity=127,
                             time=int(np.floor(mid.ticks_per_beat
                                               * note.duration))))

    mid.save(file_name)


class PianoString:
    """Single "string" of a piano roll associated with a single note of given
     pitch.
     """
    def __init__(self, note):
        self.note = note
        self.is_active = False

    def make_active(self):
        self.is_active = True

    def make_inactive(self):
        self.is_active = False


class PianoRoll:
    def __init__(self, beginning_note,  size):
        self.beginning_note = beginning_note
        self.size = size
        self.strings = []

        self._initialize_strings()

    def _initialize_strings(self):
        """Has to be evoked after size and beginning key of the PianoRoll
         are set to correct values"""
        for i in range(0, self.size):
            # in terms of MIDI note number
            current_note = self.beginning_note + i
            self.strings.append(PianoString(current_note))

    def _get_string_by_note(self, note):
        for string in self.strings:
            if string.note == note:
                return string
        raise RuntimeError("unsupported MIDI note number"
                           " requested from PianoRoll object")

    def play_string(self, note):
        string = self._get_string_by_note(note)
        string.make_active()

    def mute_string(self, note):
        string = self._get_string_by_note(note)
        string.make_inactive()

    def get_piano_roll_vector(self):
        output_vector = []
        for string in self.strings:
            if string.is_active:
                output_vector += [1]
            else:
                output_vector += [0]
        return output_vector


class Note:
    """Pitch encoded as in MIDI, duration 1.0 = quarter note"""
    def __init__(self, pitch, duration):
        self.pitch = pitch
        self.duration = duration

    def __str__(self):
        return "note object, pitch: "+str(self.pitch)+", duration: "\
               + str(self.duration)


class MidiParser:
    def __init__(self, file_name,  time_steps_per_beat):
        self.file_name = file_name
        self.file_handle = mido.MidiFile(file_name)

        self.time_steps_per_beat = time_steps_per_beat
        self.delta_time = self.file_handle.ticks_per_beat / time_steps_per_beat

        self.piano_roll = PianoRoll(43, 52)

        self.current_time = 0
        self.output_piano_roll_representation = []
        self.output_note_object_representation = []
        self._parse_file()

    def _parse_file(self):
        for track in self.file_handle.tracks:

            # import works for first track containing information
            # related to MIDI notes
            if not self._track_contains_note_messages(track):
                continue

            for msg in track:
                if msg.type in ["note_on", "note_off"]:
                    delay_in_midi_ticks = msg.time
                    epochs_to_skip = int(delay_in_midi_ticks /
                                                  self.delta_time)

                    if (msg.type == "note_on" and msg.velocity == 0)\
                            or msg.type == "note_off":
                        self.output_note_object_representation.append(
                            Note(msg.note, msg.time))

                    for i in range(0, epochs_to_skip):
                        self.output_piano_roll_representation.append(
                            self.piano_roll.get_piano_roll_vector())

                    self._execute_message(msg)
            break

    def _track_contains_note_messages(self, midi_track):
        for msg in midi_track:
            if msg.type in ["note_on", "note_off"]:
                return True
        return False

    def _execute_message(self, midi_message):
        if midi_message.type == "note_on":
            if midi_message.velocity == 0:
                self.piano_roll.mute_string(midi_message.note)
                return
            self.piano_roll.play_string(midi_message.note)
        elif midi_message.type == "note_off":
            self.piano_roll.mute_string(midi_message.note)

    # def result(self):
    #     return self.output_piano_roll_representation
    #
    # def note_result(self):
    #     return self.output_note_object_representation


def inspect_midi_file(input_file_name):
    input_file_handle = mido.MidiFile(input_file_name)
    print("ticks per beat: ", input_file_handle.ticks_per_beat)
    for i, track in enumerate(input_file_handle.tracks):
        print('Track {}: {}'.format(i, track.name))
        for msg in track:
            print(msg)

# def read_midi_file(file_name, time_steps_per_beat):
#     m = MidiParser(file_name, time_steps_per_beat)
#     return m.result()


def condensate_pianoroll(pianoroll_array, beginning_note, ticks_per_quarter):
    def  convert_from_one_hot(vector):
        for i in range(0, len(vector)):
            if vector[i] == 1:
                return i
    
    # DBG_LIMIT       = 20
    # dbg_cnt         = 0
    
    first_element = convert_from_one_hot(pianoroll_array[0]) + beginning_note
    
    piano_roll_intermediate_descriptors = []
    piano_roll_intermediate_descriptors.append({
        'one_hot_output_number': first_element, 'number_repetitions': 1})

    for i in range(1,len(pianoroll_array)):
        
        current_element_number = convert_from_one_hot(pianoroll_array[i])\
                                 + beginning_note
        if piano_roll_intermediate_descriptors[-1]['one_hot_output_number']\
                == current_element_number:
            piano_roll_intermediate_descriptors[-1]['number_repetitions'] += 1
        else:
            piano_roll_intermediate_descriptors.append({
                'one_hot_output_number': current_element_number,
                'number_repetitions': 1})

        # DBG DBG DBG
        # if dbg_cnt >= DBG_LIMIT:
            # break
        # dbg_cnt += 1

    output = []
    for entry in piano_roll_intermediate_descriptors:
        output.append(Note(entry['one_hot_output_number'],
                           np.floor(entry['number_repetitions'] /
                                    float(ticks_per_quarter))))

    return output
