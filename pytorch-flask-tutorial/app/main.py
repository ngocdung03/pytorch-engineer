# Steps:
#1. Create a directory and virtual env:
# mkdir pytorch-flask-tutorial
# cd pytorch-flask-tutorial
# python3 -m venv venv
# . venv/bin/activate  (on current directory containing the environment file)
# (to find path of activated venv: echo $VIRTUAL_ENV)
#Create this main.py file
    # 1 load image
    # 2 image -> tensor
    # 3 prediction
    # 4 return json
#Inside app directory: Set 2 environment variables
# export FLASK_APP=main.py
# export FLASK_ENV=development
# flask run
#Quit server: Ctrl+C
from flask import Flask, request, jsonify
from app.torch_utils import transform_image, get_prediction
app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})
        
    try:
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        prediction = get_prediction(tensor)
        data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
        return jsonify(data)
    except:
        return jsonify({'error': 'error during prediction'})

    # return jsonify({'result': 1})