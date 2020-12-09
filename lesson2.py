# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
from tensorflow import keras

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


