from mido import Message, MidiFile, MidiTrack, MetaMessage

MIN_DUR = 10


class Pianoroll(object):
    def __init__(self, pitch_range, program_change, msgs):
        self.pitch_range = pitch_range
        self.program_change = program_change
        self.msgs = msgs
        if msgs:
            self.roll = messages_to_pianoroll(self.pitch_range, self.msgs)


def messages_to_pianoroll(pitch_range, msgs):
    pattern, stack = [0] * (len(pitch_range) + 1), []
    storage, pause, modulo = 0, 0, 0
    status = msgs[0].type
    for m in msgs:
        if m.type == 'note_on':
            if status == 'note_off':
                if storage > 0:
                    stack.append(pattern.copy())
                    x = pattern
                    x[-1] = 1
                    stack.extend([x] * (storage - 1))
                    pattern = [0] * (len(pitch_range) + 1)
                    storage = 0
                status = 'note_on'
            pause += m.time // MIN_DUR
            modulo += m.time % MIN_DUR
            if modulo > MIN_DUR and stack:
                stack.extend([stack[-1].copy()] * (modulo // MIN_DUR))
                modulo = 0
        else:
            if status == 'note_on':
                if pause > 0:
                    stack.append(pattern.copy())
                    x = pattern
                    x[-1] = 1
                    stack.extend([x] * (pause - 1))
                    pattern = [0] * (len(pitch_range) + 1)
                    pause = 0
                status = 'note_off'
            storage += m.time // MIN_DUR
            modulo += m.time % MIN_DUR
            if modulo > MIN_DUR and stack:
                stack.extend([stack[-1].copy()] * (modulo // MIN_DUR))
                modulo = 0
            pattern[pitch_range.index(m.note)] = 1
    if storage > 0:
        stack.append(pattern.copy())
        x = pattern
        x[-1] = 1
        stack.extend([x] * (storage - 1))
    return stack


def parse_midi(path, min_dur=5):
    global MIN_DUR
    MIN_DUR = min_dur
    mid = MidiFile(path)
    pianorolls, tempo_msgs = [], []
    tic_f = 480 / mid.ticks_per_beat

    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo_msgs.append(msg)

    if not tempo_msgs:
        tempo_msgs = [MetaMessage(type='set_tempo', tempo=500000, time=0)]

    for i, track in enumerate(mid.tracks):
        program_change = Message(type='program_change', channel=0, program=0)
        msgs, pitch_range = [], []
        cur_time = 0
        cur_tempo = tempo_msgs.copy()
        tempo = cur_tempo[0].tempo / 500000
        for msg in track:
            cur_time += msg.time

            if msg.type == 'note_on' or msg.type == 'note_off':
                if len(cur_tempo) > 1 and cur_time > cur_tempo[0].time + cur_tempo[1].time:
                    cur_tempo.pop(0)
                    tempo = cur_tempo[0].tempo / 500000
                    if cur_tempo[0].time == 0:
                        cur_time = 0

                if msg.note not in pitch_range:
                    pitch_range.append(msg.note)

                new_time = int(msg.time * tempo * tic_f)
                round_factor = new_time % 10
                if round_factor != 0:
                    if round_factor < 5:
                        new_time -= round_factor
                    else:
                        new_time += (10 - round_factor)

                if msg.type == 'note_on' and msg.velocity < 5:
                    new_type = 'note_off'
                else:
                    new_type = msg.type

                msgs.append(Message(type=new_type, note=msg.note,
                                    time=new_time, channel=msg.channel))

            elif msg.type == 'program_change':
                program_change.channel = msg.channel
                program_change.program = msg.program
        if msgs:
            pianorolls.append(Pianoroll(pitch_range, program_change, msgs))
    return pianorolls


def roll_to_message(track, pianoroll, pattern, storage, pause):
    pitches = []
    for i, _ in enumerate(pianoroll.pitch_range):
        if pattern[i] == 1:
            pitches.append(pianoroll.pitch_range[i])

    if pitches:
        c = pianoroll.program_change.channel

        if pause > 0:
            track.append(Message('note_on', note=pitches[0], time=MIN_DUR * pause, channel=c))
            pad = 1
        else:
            pad = 0

        for z in pitches[pad:]:
            track.append(Message('note_on', note=z, time=0, channel=c))

        track.append(Message('note_off', note=pitches[0], time=MIN_DUR * storage, channel=c))

        for z in pitches[1:]:
            track.append(Message('note_off', note=z, time=0, channel=c))


def pianoroll_to_messages(pianoroll):
    track = MidiTrack()
    track.append(pianoroll.program_change)
    storage, pause = 0, 0
    roll = pianoroll.roll
    pattern = []
    count = 0
    while roll:
        if not pattern:
            pattern = roll[0]
            roll.pop(0)
        else:
            if roll[0][-1] == 1:
                roll.pop(0)
                count += 1
            else:
                if sum(pattern) == 0:
                    pause = count
                else:
                    storage = count
                if storage > 0:
                    roll_to_message(track, pianoroll, pattern, storage, pause)
                    pause, storage = 0, 0
                pattern = roll[0]
                count = 1
                roll.pop(0)
    if sum(pattern) == 0:
        pause = count
    else:
        storage = count
    if storage > 0:
        roll_to_message(track, pianoroll, pattern, storage, pause)
    return track


class Polyphonic_pianoroll:
    def __init__(self, pianorolls):
        self.pianorolls = pianorolls
        self.to_polyphonic_pianoroll()

    polyphonic_pianoroll = []

    def to_polyphonic_pianoroll(self):
        max_l = 0
        for t in self.pianorolls:
            max_l = max(max_l, len(t.roll))
        for t in self.pianorolls:
            if len(t.roll) < max_l:
                t.roll.extend([[0] * len(t.roll[0])] * (max_l - len(t.roll)))
        polyphonic_pianoroll = []
        for x in range(max_l):
            new = []
            for t in self.pianorolls:
                new.extend(t.roll[x])
            polyphonic_pianoroll.append(new)
        self.polyphonic_pianoroll = polyphonic_pianoroll

    def back_to_pianorolls(self):
        for t in self.pianorolls:
            t.roll = []
        for p in self.polyphonic_pianoroll:
            for t in self.pianorolls:
                t.roll.append(p[:len(t.pitch_range) + 1])
                p = p[len(t.pitch_range) + 1:]


def save_midi(tracks, name):
    mid = MidiFile()
    for t in tracks:
        # print(t.pitch_range)
        # print(t.roll)
        mid.tracks.append(pianoroll_to_messages(t))
    mid.save(name)
