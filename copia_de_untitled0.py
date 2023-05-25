import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.layers.regularization.spatial_dropout3d import Dropout
from tensorflow.keras.callbacks import TensorBoard


datos, metadatos = tfds.load("fashion_mnist", as_supervised = True, with_info = True)

metadatos

datos_entrenamiento = []

TAMANO_IMAGEN = 224

for i, (imagen, etiqueta) in enumerate(datos["train"].take(15)):
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMAGEN, TAMANO_IMAGEN))
    
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    plt.subplot(3, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen, cmap='gray')
    plt.xlabel(etiqueta.numpy())
    plt.tight_layout()

plt.show()



for i, (imagen, etiqueta) in enumerate(datos["train"]):
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMAGEN, TAMANO_IMAGEN))  # cambiamos de tamaño la imagen
    imagen = imagen.reshape(TAMANO_IMAGEN, TAMANO_IMAGEN, 1)  # agregamos la dimensión del canal (escala de grises)
    datos_entrenamiento.append([imagen, etiqueta])

print(datos_entrenamiento[3])

X = [] # Ejemplos
y = [] # Las respectivas etiquetas

for imagen, etiqueta in datos_entrenamiento:
  X.append(imagen)
  y.append(etiqueta)

y

import numpy as np

# Normalización

X = np.array(X).astype(float)/255

y = np.array(y)

y

X.shape


modeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100, 1)),
    tf.keras.layers.Dense(150, activation="relu"),
    tf.keras.layers.Dense(150, activation="relu"),
    tf.keras.layers.Dense(1, activation ="sigmoid") # Una neurona porque la salida es binaria
])


modeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),    
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

modeloDenso.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

modeloCNN.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)



# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs

tensorboardCNN = TensorBoard(log_dir = "logs/CNN")

modeloCNN.fit(
    X, y,
    validation_split = 0.15,
    epochs = 3,
    callbacks = [tensorboardCNN]
)

modeloCNN.save("fashion-mnist_CNN.h5")

!pip install tensorflowjs

!mkdir carpeta_salida

!tensorflowjs_converter --input_format keras fashion-mnist_CNN.h5 carpeta_salida/
