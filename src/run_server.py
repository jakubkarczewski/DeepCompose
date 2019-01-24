import flask
from net import Net
import argparse
import os
from midi import Polyphonic_pianoroll, parse_midi, save_midi
app = flask.Flask(__name__)
model = None


@app.route("/predict", methods=["POST"])
def predict():
    temp_filename = 'my_file.mid'
    data = {"success": False}
    net = app.config.get('net')
    if flask.request.method == "POST":
        if flask.request.files.get("melody"):
            melody = flask.request.files["melody"].read()
            file = open(temp_filename, 'wb')
            file.write(melody)
            file.close()
            preds = []
            data["predictions"] = net.run_inference(temp_filename, app.config.get('model_path'), mode="rest_api_mode",
                                                    predictions=preds)
            os.remove(temp_filename)
            data["success"] = True

    return flask.jsonify(data)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_filename", type=str, required=True,
                        help="Name of input MIDI file.")
    parser.add_argument("--min_dur", type=int, default=40,
                        help="Min dur value")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Size of training batch")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="echo the string you use here")
    parser.add_argument("--gen_len", type=int, default=10,
                        help="echo the string you use here")
    parser.add_argument("--gen_num", type=int, default=5,
                        help="echo the string you use here")
    parser.add_argument("--weights_path", type=str,
                        help="echo the string you use here")
    args = parser.parse_args()

    my_net = Net(args.min_dur, args.batch_size, args.epochs, args.gen_len, args.gen_num)
    app.config['model_path'] = args.weights_path
    app.config['net'] = my_net
    app.run()
