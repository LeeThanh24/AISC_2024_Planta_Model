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
    # employees = [{'id': 2, 'name': 'Ashley'}, {'id': 2, 'name': 'Kate'}, {'id': 3, 'name': 'Joe'}]
    # model = YOLO('best.pt')  # load a custom model
    # #
    # results = model('image/dataTest.jpg')  # predict on an image
    # names_dict = results[0].names
    # probs = results[0].probs.data.tolist()
    # print(np.argmax(probs))

    # print(names_dict)
    # print(probs)
    # categories = {
    #     "CayTrauBa": "Cây Trầu Bà",
    #     "CayLuoiHo": "Cây Lưỡi Hổ",
    #     "CayKimTien": "Cây Kim Tiền",
    #     "CayKimNgan": "Cây Kim Ngân",
    #     "CayThuongXuan": "Cây Thường Xuân",
    #     "CayThietMocLan": "Cây Thiết Mộc La ",
    #     "CayVanNienThanh": "Cây Vạn Niên Thanh",
    #     "CayPhuQuy": "Cây Phú Quý",
    #     "CayLanY": "Cây Lan Ý",
    #     "CayXuongRong": "Cây Xương Rồng",
    #     "CayCoCanh": "Cây Có Cánh",
    #     "CayDaNgocMinhChau": "Cây Dạ Ngọc Minh Châu"
    # }
    # print(names_dict[np.argmax(probs)])
    # result_name = names_dict[np.argmax(probs)]
    # predict_result = {"name": result_name}
    # return jsonify(predict_result)
    return "hello world"

@app.route('/predict', methods=['POST'])
def uploadImage():
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
            result = predict(destination)
            return result , 200
    except Exception as e:
        print(e)
        return "Error in uploading file", 500

def predict(path):
    model = YOLO('best.pt')  # load a custom model
    results = model(path)  # predict on an image
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    result_name = names_dict[np.argmax(probs)]
    predict_result = {"name": result_name}
    return jsonify(predict_result)

if __name__ == '__main__':
    app.run(host ='0.0.0.0', debug=False, port=8000)
