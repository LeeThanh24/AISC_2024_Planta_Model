import os
from flask import Flask, jsonify, request
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Tạo thư mục uploads để lưu ảnh
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def hello_world():
    return "hello world"


def uploadImage(path):
    try:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file:
            # print(os.path)
            destination = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(destination)
            result = predict(path, destination)
            return result, 200
    except Exception as e:
        print(e)
        return "Error in uploading file", 500


@app.route('/plants/predict', methods=['POST'])
def uploadImagePlants():
    return uploadImage('/plants/predict')


@app.route('/deseases/predict', methods=['POST'])
def uploadImageDeseases():
    return uploadImage('/deseases/predict')


def predict(mode, path):
    print("my mode ", mode)
    if mode == '/deseases/predict':
        model = YOLO('best_deseases.pt')  # load desease model
    else:
        model = YOLO('best.pt')  # load plant model
    results = model(path)  # predict on an image
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()

    maxIndex = np.argmax(probs)

    relevant =[names_dict[probs.index(x)] for x in probs if x != probs[maxIndex]]

    result_name = names_dict[np.argmax(probs)]

    predict_result = {"name": result_name, "relevant":relevant}
    return jsonify(predict_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8000)
