import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load tập dữ liệu MNIST
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Tiền xử lý dữ liệu (chuẩn hóa giá trị pixel về khoảng [0, 1])
train_images, test_images = train_images / 255.0, test_images / 255.0

# Xây dựng mô hình mạng nơ-ron
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Biến đổi hình ảnh 28x28 thành mảng 1 chiều 784
    layers.Dense(128, activation='relu'),  # Lớp ẩn có 128 nơ-ron với hàm kích hoạt ReLU
    layers.Dropout(0.2),  # Áp dụng dropout để tránh overfitting
    layers.Dense(10)  # Lớp đầu ra có 10 nơ-ron tương ứng với 10 số từ 0 đến 9
])

# Biên dịch mô hình
model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# Đào tạo mô hình
model.fit(train_images, train_labels, epochs=10)

# Đánh giá mô hình trên tập kiểm tra
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Accuracy on test data: {test_acc}')

# Lưu mô hình đã đào tạo vào tệp
model.save('mnist_model.h5')
print("Model saved as mnist_model.h5")
