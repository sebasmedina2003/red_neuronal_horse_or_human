import tensorboard
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2

datos, metadatos = tfds.load('horses_or_humans', as_supervised=True, with_info=True)
IMG_SIZE = 200
EPOCHS = 10
BATCH_SIZE = 32

datos_entrenamiento = []
x = [] 
y = [] 
for i, (imagen, label) in enumerate(datos['train']):
    imagen = cv2.resize(imagen.numpy(), (IMG_SIZE, IMG_SIZE))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = imagen.reshape(IMG_SIZE, IMG_SIZE, 1)

    datos_entrenamiento.append([imagen, label])
    x.append(imagen)
    y.append(label)

x = np.array(x).astype('float') / 255
y = np.array(y)

modeloDenso = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    keras.layers.Dense(150, activation='relu'),
    keras.layers.Dense(150, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid') # Salida binaria
])

modeloCNN = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

modeloCNN2 = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

modeloDenso.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
    )

modeloCNN.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
    )

modeloCNN2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
    )

tensorboard_denso = keras.callbacks.TensorBoard(log_dir='logs/denso')
modeloDenso.fit(x, y, batch_size=BATCH_SIZE, validation_split=0.15, epochs=EPOCHS, callbacks=[tensorboard_denso])

tensorboard_cnn = keras.callbacks.TensorBoard(log_dir='logs/cnn')
modeloCNN.fit(x, y, batch_size=BATCH_SIZE, validation_split=0.15, epochs=EPOCHS, callbacks=[tensorboard_cnn])

tensorboard_cnn2 = keras.callbacks.TensorBoard(log_dir='logs/cnn2')
modeloCNN2.fit(x, y, batch_size=BATCH_SIZE, validation_split=0.15, epochs=EPOCHS, callbacks=[tensorboard_cnn2])

modeloDenso.save('bin/modeloDenso.h5')
modeloCNN.save('bin/modeloCNN.h5')
modeloCNN2.save('bin/modeloCNN2.h5')

