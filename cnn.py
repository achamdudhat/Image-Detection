import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import  keras
import tensorflow.keras.layers as tfl
from tensorflow.keras import datasets, models, layers
from tensorflow.python.framework import ops


# Load and preprocess the dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# Normalize the pizels
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(len(X_train), 28, 28, 1)
X_test = X_test.reshape(len(X_test), 28, 28, 1)
#print(len(X_train))
#print(len(X_test))
#print(len(y_train))
#print(len(y_test))
#plt.matshow(X_train[1])
#plt.show()
#print(X_train.shape) # 60000,28,28
#print(X_test.shape) #10000,28,28

# Convolving, pooling and adding layers with its activation
model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

# Compile the model and train
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5,validation_data=(X_test, y_test))

# Make predictions on the test set
#y_predicted = model.predict(X_test)
#y_predicted_labels = [np.argmax(i) for i in y_predicted]
#print(y_predicted_labels)
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print("\nTest accuracy: %.1f%%" % (100.0 * test_acc))