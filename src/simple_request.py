import requests
import argparse

from midi import Polyphonic_pianoroll, save_midi, parse_midi


def simple_request(filename, url, min_dur):
    melody = open(filename, "rb")
    payload = {"melody": melody}
    r = requests.post(url, files=payload).json()
    tracks = parse_midi(filename, min_dur)
    song = Polyphonic_pianoroll(tracks)
    pianoroll = list(map(tuple, song.polyphonic_pianoroll))
    if r["success"]:
        for item in r["predictions"]:
            song.polyphonic_pianoroll = item
            song.back_to_pianorolls()
            save_midi(song.pianorolls, 'generated-%s.mid' % r["predictions"].index(item))
    else:
        print("Request failed")
        print(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_filename", type=str, required=True,
                        help="Name of input MIDI file.")
    parser.add_argument("--min_dur", type=int, default=40,
                        help="Min dur value")
    args = parser.parse_args()

    rest_api_url = "http://localhost:5000/predict"
    simple_request(args.midi_filename, rest_api_url, args.min_dur)
