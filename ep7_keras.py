#https://youtu.be/BQg9OZdzLLE
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Dense

c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46.4, 59, 71.6, 100.4])

model = keras.Sequential()
model.add(Dense(units=1, input_shape=(1, ), activation='linear'))
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))

history = model.fit(c, f, epochs=800, verbose=False)

plt.plot(history.history['loss'])
plt.grid(True)
plt.show()

print(model.predict([12]))
print(model.get_weights())