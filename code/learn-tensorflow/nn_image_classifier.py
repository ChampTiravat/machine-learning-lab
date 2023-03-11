import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.datasets import fashion_mnist as dataset
from tensorflow.keras.callbacks import Callback


config = {
    'EPOCHS': 50,

    # ADAM is 2nd-order optimization, give better performance than SGD.
    'OPTIMIZER': 'adam',

    # use this loss function when output has mutiple values.
    'LOSS_FUNCTION': 'sparse_categorical_crossentropy',

    'SHOW_IMAGE': False,
}


# Retrieve Fashion MNIST data and split them into training & testing sets.
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()


# Data normalization so the model can train faster and perform better.
train_images = train_images // 255.0
test_images = test_images // 255.0


model = Sequential([
    # Convert 28x28 pixels 2-dimensional array into 1-dimensional array.
    Flatten(input_shape=(28, 28)),

    # Randomly set number of neurons in this layer to 128.
    Dense(units=128, activation=tf.nn.relu),

    # Using `softmax` to pick the best prediction value out of the 10 values.
    Dense(units=10, activation=tf.nn.softmax),
])


model.compile(
    optimizer=config['OPTIMIZER'],
    loss=config['LOSS_FUNCTION'],
    metrics=['accuracy']  # Display accuracy in training procedure.
)


# Display architecture of the model on the terminal.
model.summary()


# Using keras.Callback to set conditions to stop training procedure.
class TrainingCallBack(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print('Stop training since the accuracy is above 95%')
            self.model.stop_training = True  # Stop training the model


callbacks = TrainingCallBack()


# Training the model on the training set.
model.fit(
    train_images,
    train_labels,
    epochs=config['EPOCHS'],
    #callbacks=[callbacks],
)


# Running the model against the testing set.
model.evaluate(test_images, test_labels)


# Perform classification.
classifications = model.predict(test_images)


fashion_mnist_class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
print(fashion_mnist_class_names)


if not config['SHOW_IMAGE']:
    # Display result without image.

    label = fashion_mnist_class_names[test_labels[0]]
    classification_result = classifications[0]

    print(classification_result)
    print('Prediction: {}'.format(label))

else:
    # Display result with images.

    for i in range(10):
        classification_result = classifications[i]
        result_image = test_images[i]
        label = fashion_mnist_class_names[test_labels[i]]

        print(classification_result)
        plt.title('Prediction: {}'.format(label))
        plt.imshow(result_image)
        plt.show()
