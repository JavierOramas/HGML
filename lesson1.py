# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import keras
from keras import layers
import tensorflow as tf
import numpy as np
import pandas as pd


# %%
lin_reg = keras.Sequential([
    layers.Dense(1,activation='linear', input_shape = [1]),
])

optimizer = tf.keras.optimizers.RMSprop(0.001)

lin_reg.compile(
    loss='mse',
    optimizer=optimizer,
    metrics=['mse']
)


# %%
def plot_error(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mse'],
            label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
            label = 'Val Error')

  plt.legend()
  plt.show()


# %%
data = np.array([
  [4,2],
  [4,10],
  [7,4],
  [7,22],
  [8,16],
  [9,10],
  [10,18],
  [10,26],
  [10,34],
  [11,17],
  [11,28],
  [12,14],
  [12,20],
  [12,24],
  [12,28],
  [13,26],
  [13,34],
  [13,34],
  [13,46],
  [14,26],
  [14,36],
  [14,60],
  [14,80],
  [15,20],
  [15,26],
  [15,54],
  [16,32],
  [16,40],
  [17,32],
  [17,40],
  [17,50],
  [18,42],
  [18,56],
  [18,76],
  [18,84],
  [19,36],
  [19,46],
  [19,68],
  [20,32],
  [20,48],
  [20,52],
  [20,56],
  [20,64],
  [22,66],
  [23,54],
  [24,70],
  [24,92],
  [24,93],
  [24,120],
  [25,85]
])


# %%
speed = data[:,0]
distance = data[:,1]


# %%
history = lin_reg.fit(
    x=speed,
    y = distance,
    shuffle=True,
    epochs=1000,
    validation_split=0.2,
    verbose=0
)


# %%
import matplotlib.pyplot as plt

pred_distance = lin_reg.predict(speed)

plt.plot(speed,pred_distance, color='red') 
plt.scatter(speed,distance)
plt.show()


# %%
plot_error(history)


# %%
def build_nn():
    net = keras.Sequential([
        layers.Dense(32,activation='relu', input_shape = [1]),
        layers.Dense(16, activation='relu'),
        layers.Dense(1),
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    net.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mse', 'accuracy']
    )

    return net


# %%
net = build_nn()

history = net.fit(
    x = speed,
    y = distance, 
    shuffle=True,
    epochs=1000,
    validation_split=0.2,
    verbose=0
)


# %%
net_dist = net.predict(speed)
plt.plot(speed, net_dist, color='red')
plt.plot(speed, pred_distance, color='orange')
plt.scatter(speed,distance)

plt.show()


# %%
plot_error(history)


# %%
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10
)


# %%
net = build_nn()

history = net.fit(
    x= speed,
    y= distance,
    shuffle= True,
    epochs= 1000,
    validation_split= 0.2,
    verbose= 0,
    callbacks= [early_stop] 
)


# %%
plot_error(history)


# %%
net.save('simple_nn.h5')


# %%
simple_net= keras.models.load_model('simple_nn.h5')


# %%
simple_net.summary()


