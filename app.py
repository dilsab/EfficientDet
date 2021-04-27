import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import io
import base64

from detect import detect_objects

ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'img' not in request.files:
            flash('Request has no image. Please add image.')
            return redirect(request.url)
        img = request.files['img']
        if img and allowed_file(img.filename):
            filename = secure_filename(img.filename)

            # Temp save image for object detection
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(save_path)

            detection_result = detect_objects(save_path)

            buf = io.BytesIO()
            detection_result['image'].savefig(buf, format='jpg', bbox_inches='tight')
            img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            os.remove(save_path)

            return render_template('objects_detected.html', img=img_data, detection_result=detection_result)

    return render_template('index.html')
