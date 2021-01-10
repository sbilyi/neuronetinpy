# https://youtu.be/oCXh_GFMmOE
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat= keras.utils.to_categorical(y_train, 10)
y_test_cat= keras.utils.to_categorical(y_test, 10)

# plt.figure(figsize=(10, 5))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i], cmap = plt.cm.binary)
# plt.show()

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

model.evaluate(x_test, y_test_cat)

n=2
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(f'The number is: {np.argmax(res)}')
plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

predicted = model.predict(x_test)
predicted = np.argmax(predicted, axis=1)

print(predicted)
print(predicted[:20])
print(y_test[:20])

mask = predicted == y_test
print(mask[:10])

x_false = x_test[~mask]
p_false = predicted[~mask]

print(x_false.shape)

for i in range(5):
    print('Actual value :' + str(y_test[i]))
    print('Net value :' + str(p_false[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()

