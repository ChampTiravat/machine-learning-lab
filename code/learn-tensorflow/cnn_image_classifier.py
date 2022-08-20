import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images / 255

test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(
    train_images,
    train_labels,
    epochs=5,
)

scores = model.evaluate(test_images, test_labels)

results = model.predict(test_images)


print("result[0] : {}".format(results[0]))
print("test_labels[0]: {}".format(test_labels[0]))

# print("scores: {}".format(scores))
# print("results: {}".format(results))
