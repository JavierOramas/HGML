# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()


# %%
def preprocess(x,y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)

    return x,y

def create_dataset(xs, ys, n_classes=10):
    ys = tf.one_hot(ys, depth=n_classes)

    return tf.data.Dataset.from_tensor_slices((xs,ys))         .map(preprocess)         .shuffle(len(ys))         .batch(128)


# %%
train_ds = create_dataset(x_train,y_train)
val_ds = create_dataset(x_val, y_val)


# %%
model = keras.Sequential([
    keras.layers.Reshape(
        target_shape=(28 * 28,), input_shape=(28,28)
    ),
    keras.layers.Dense(
        units=256, activation='relu'
    ),
    keras.layers.Dense(
        units=192, activation='relu'
    ),
    keras.layers.Dense(
        units=128, activation='relu'
    ),
    keras.layers.Dense(
        units=10, activation='softmax'
    )
])


# %%
model.compile(
    optimizer='adam',
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_ds.repeat(),
    epochs=10,
    steps_per_epoch=500,
    validation_data=val_ds.repeat(),
    validation_steps=2
)


# %%
predictions = model.predict(val_ds)
best_predictions = [i.argmax() for i in predictions]
correct = 0
for i in range(len(best_predictions)):
    if best_predictions[i] == y_val[i]:
        correct += 1
print(correct/len(y_val)*100)


# %%
model.save('fashion_mnist(10.4).h5')


# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim((0, 1)) # Uncomment this when showing you model for pay raise
plt.legend(['train', 'test'], loc='upper left');


# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim((1.5, 2))
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# %%
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'green'
  else:
    color = 'red'
  
  plt.xlabel("Predicted: {} {:2.0f}% (True: {})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


# %%
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# %%
i = 200
plot_image(i, predictions, y_val, x_val)


