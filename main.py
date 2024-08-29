import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("Welcome to the NeuralNine (c) Handwritten Digits Recognition v0.1")

model_path = 'handwritten_digits.h5'

def fix_activation_function(config):
    if isinstance(config.get('activation'), dict) and config['activation']['class_name'] == 'function':
        config['activation'] = 'softmax'
    return config

if os.path.exists(model_path):

    with tf.keras.utils.custom_object_scope({'fix_activation_function': fix_activation_function}):
        model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
else:
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=3)

    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

    model.save(model_path)
    print("Model trained and saved successfully.")

image_number = 1
while os.path.isfile(f'digits/digit{image_number}.png'):
    try:
        img = cv2.imread(f'digits/digit{image_number}.png', cv2.IMREAD_GRAYSCALE)
        img = np.invert(img)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28)
        prediction = model.predict(img)
        print(f"The number is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except Exception as e:
        print(f"Error reading image: {e}. Proceeding with next image...")
        image_number += 1
