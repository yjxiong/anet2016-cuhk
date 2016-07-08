"""
A minimal server for web demo of action recognition
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from pyActionRec.action_classifier import ActionClassifier
from pyActionRec.anet_db import ANetDB
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'tmp/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# init action classifier
# init classifier
models = [('models/resnet200_anet_2016_deploy.prototxt',
           'models/resnet200_anet_2016.caffemodel',
           1.0, 0, True),
          ('models/bn_inception_anet_2016_temporal_deploy.prototxt',
           'models/bn_inception_anet_2016_temporal.caffemodel',
           0.2, 1, False)
          ]

cls = ActionClassifier(models)
db = ANetDB.get_db("1.3")
lb_list = db.get_ordered_label_list()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ['avi', 'mp4', 'webm', 'mkv']


@app.route("/")
def main():
    return render_template('index.html')


@app.route("/upload_video", methods=['POST'])
def upload_video():
    if 'video_file' not in request.files:
        return jsonify(error='upload not found'), 500, {'ContentType': 'application/json'}

    upload_file = request.files['video_file']
    if upload_file.filename == '':
        return jsonify(error='the file has no name'), 500, {'ContentType': 'application/json'}

    if upload_file and allowed_file(upload_file.filename):
        filename = secure_filename(upload_file.filename)

        # first save the file
        savename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        upload_file.save(savename)

        # classify the video
        try:
            scores, frm_scores = cls.classify(savename)
        except:
            return jsonify(error='classification failed'), 500, {'ContentType': 'application/json'}
        finally:
            # clear the file
            os.remove(savename)

        idx = np.argsort(scores)[::-1]

        top_3_results = []
        for i in xrange(3):
            k = idx[i]
            top_3_results.append('{}: {}'.format(lb_list[k], scores[k]))

        # return the result in json
        return jsonify(error=None, results=top_3_results), 200, {'ContentType': 'application/json'}

    else:
        return jsonify(error='empty or not allowed file'), 500, {'ContentType': 'application/json'}

if __name__ == "__main__":
    # run the Flask app
    app.debug = True
    app.run()
