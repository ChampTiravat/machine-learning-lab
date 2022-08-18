import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

layer1 = Dense(units=1, input_shape=[1])

model = Sequential([layer1])

model.compile(optimizer='sgd', loss='mean_squared_error')

x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(x, y, epochs=500)

print('Result: {}'.format(model.predict([10.0])))
print('Weights of the 1st layer: {}'.format(layer1.get_weights()))
