

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu, input_shape= x_train.shape[1:]))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3)

val_loss,val_acc = model.evaluate(x_test,y_test)
print(val_loss,val_acc)

path_file = 'Q:\\Programacao\\Python\\PyCharm-Projetos\\Deep-Learning-Mendonca\\src\\models\\'

model.save(path_file+'epic_num_reader.model')

new_model = tf.keras.models.load_model(path_file+'epic_num_reader.model')

predictions = new_model.predict(np.array(x_test))

print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()
##print(x_train[0])