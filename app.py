from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import yolo_detector
import trick_classifier
import optical_flow
import database

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)

        video = request.files['video']

        if video.filename == '':
            return redirect(request.url)

        if video and allowed_file(video.filename):
            filename = secure_filename(video.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video.save(video_path)

            skater, skateboard = yolo_detector.detect_objects(video_path)
            trick, flipped = trick_classifier.predict_flip(video_path)
            direction = optical_flow.track_direction(video_path)

            # Log results in database
            database.store_result(filename, trick, flipped, direction)

            return render_template('result.html',
                                   trick=trick,
                                   flipped=flipped,
                                   direction=direction)

    return render_template('home.html')

@app.route('/stats')
def stats():
    data = database.get_all_results()
    return render_template('stats.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
