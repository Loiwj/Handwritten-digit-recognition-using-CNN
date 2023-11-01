from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow import keras
from PIL import Image
import base64
import io  # Import thư viện io

app = Flask(__name__)
model = keras.models.load_model('mnist_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['imageData'].split(",")[1]
    image_bytes = base64.b64decode(image_data)
    
    # Đọc ảnh từ dữ liệu bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0

    image = image.reshape((1, 28, 28))
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    return jsonify({'predicted_class': int(predicted_class)})


if __name__ == '__main__':
    app.run(debug=True)
