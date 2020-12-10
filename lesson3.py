# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
import joblib

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 16,10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# %%
df = pd.read_csv('airbnb_nyc_2019.csv')


# %%
df.shape


# %%
sns.distplot(df.price)


# %%
sns.distplot(np.log1p(df.price))


# %%
sns.countplot(x='room_type', data=df)


# %%
sns.countplot(x='neighbourhood_group', data=df)


# %%
sns.distplot(df.number_of_reviews)


# %%
corr_matrix = df.corr()
price_corr = corr_matrix['price']
price_corr.iloc[price_corr.abs().argsort()]


# %%
missing = df.isnull().sum()
missing[missing > 0].sort_values(ascending=False)


# %%
df = df.drop(['id', 'name', 'host_id', 'host_name', 'reviews_per_month', 'last_review', 'neighbourhood'], axis=1)


# %%
df


# %%
x = df.drop('price', axis=1)
y = np.log1p(df.price.values)


# %%
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.compose import make_column_transformer

transformer = make_column_transformer(
    (MinMaxScaler(), [
        'latitude', 'longitude', 'minimum_nights', 
        'number_of_reviews', 'calculated_host_listings_count', 'availability_365'
    ]
),
(OneHotEncoder(handle_unknown='ignore'), ['neighbourhood_group', 'room_type'])
)


# %%
transformer.fit(x)


# %%
x = transformer.transform(x)


# %%
x_train, x_test, y_train, y_test =    train_test_split(x,y,test_size=0.2,random_state=RANDOM_SEED)


# %%
model = keras.Sequential()
model.add(keras.layers.Dense(
    units=64,
    activation='relu',
    input_shape = [x_train.shape[1]]
    ))
model.add(keras.layers.Dropout(rate=0.3))
model.add(keras.layers.Dense(
    units=32, activation='relu'
))
model.add(keras.layers.Dropout(rate=0.5))

model.add(keras.layers.Dense(1))


# %%
model.compile(
    optimizer = keras.optimizers.Adam(0.0001),
    loss = 'mae',
    metrics = ['mae']
)


# %%
BATCH_SIZE = 32

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_mae',
    mode='min',
    patience=10
)

history = model.fit(
    x=x_train,
    y=y_train,
    shuffle=True,
    epochs=100, 
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)


# %%
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model accuracy')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.ylim((0, 3)) # Uncomment this when showing you model for pay raise
plt.legend(['train mae', 'val mae'], loc='upper left');


# %%
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

y_pred = model.predict(x_test)


# %%
print(f'MSE {mean_squared_error(y_test, y_pred)}')
print(f'RMSE {np.sqrt(mean_squared_error(y_test, y_pred))}')
print(f'R2 {r2_score(y_test, y_pred)}')


# %%
joblib.dump(transformer, 'data_transformes.joblib')
model.save('airbnb_price_pred.h5')


