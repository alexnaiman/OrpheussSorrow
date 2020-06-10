from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
# from model import MidiModelUtils
import tensorflow as tf
import keras
from orpheus_model import OrpheusModel

app = Flask(__name__)
cors = CORS(app, resources={
            r"/generate_song": {"origins": "http://localhost:3000"}})

model = None

# some Tensorflow configs for working with Keras
config = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)
session = tf.Session(config=config)
keras.backend.set_session(session)


@app.route('/generate_song', methods=['GET', 'POST'])
def get_songs_by_features():
    data = request.json
    features, emotion_data, threshold = OrpheusModel.parse_json_data(data)

    with session.as_default():
        with session.graph.as_default():
            return jsonify(model.get_song_by_features(features, emotion_data, threshold))


if __name__ == '__main__':
    # Loading our main model before running our server
    model = OrpheusModel()
    model.load_model()
    model.load_normalized_values()
    app.run()
