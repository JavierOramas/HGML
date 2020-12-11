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

sns.set(style='whitegrid', palette='muted', font_scale=1)

# rcParams['figure.figsize'] = 10, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# %%
def plot_scaling_result(data, scaled_data, scaling_title, x_lim=(-5, 5)):

  scaled_df = pd.DataFrame(
      scaled_data, 
      columns=['Normal', 'Exponential', 'Uniform']
  )

  fig, (ax1, ax2) = plt.subplots(ncols=2)
  ax1.set_xlim((-300, 300))
  ax1.set_title('No Scaling')
  sns.kdeplot(data.Normal, ax=ax1)
  sns.kdeplot(data.Exponential, ax=ax1)
  sns.kdeplot(data.Uniform, ax=ax1)

  ax2.set_xlim(x_lim)
  ax2.set_title(scaling_title)
  sns.kdeplot(scaled_df.Normal, ax=ax2)
  sns.kdeplot(scaled_df.Exponential, ax=ax2)
  sns.kdeplot(scaled_df.Uniform, ax=ax2);


# %%
data = pd.DataFrame(
    {
        'Normal': np.random.normal(100,50,1000),
        'Exponential': np.random.exponential(25,1000),
        'Uniform': np.random.uniform(-150,-50,1000)
    }
)


# %%
from sklearn.preprocessing import MinMaxScaler

min_max_scaled = MinMaxScaler(feature_range=(0,1)).fit_transform(data)
# plot_scaling_result(data,min_max_scaled,'Min Max Scaler')
plot_scaling_result(data, min_max_scaled, 'Min-Max Scaling', (-1.5, 1.5))


# %%
from sklearn.preprocessing import StandardScaler

stand_scaled = StandardScaler().fit_transform(data)
plot_scaling_result(data, stand_scaled, 'Standard Scaling', (-7, 7))


# %%
from sklearn.preprocessing import RobustScaler

robust_scaled = RobustScaler().fit_transform(data)
plot_scaling_result(data, robust_scaled, 'Robust Scaling', (-7, 7))


# %%
property_type = np.array(['House', 'Unit', 'Townhouse', 'House', 'Unit']).reshape(-1, 1)


# %%
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder().fit(property_type)
labels = enc.transform(property_type)
labels.flatten()


# %%
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False).fit(property_type)
one_hots = enc.transform(property_type)
one_hots


# %%
n_rooms = np.array([1, 2, 1, 4, 6, 7, 12, 20])
pd.cut(n_rooms, bins=[0, 3, 8, 100], labels=["small", "medium", "large"])


# %%
dates = pd.Series(["1/04/2017", "2/04/2017", "3/04/2017"])
pd_dates = pd.to_datetime(dates)
pd_dates.dt.dayofweek

# %% [markdown]
# # Melbourne House Pricing
# 
# ## Data from Kaggle: https://www.kaggle.com/anthonypino/melbourne-housing-market/

# %%
df = pd.read_csv('assets/MELBOURNE_HOUSE.csv')
df.shape


# %%
missing = df.isnull().sum()
missing[missing > 0].sort_values(ascending=False)


# %%
df = df.dropna()


# %%
sns.distplot(df.Rooms.dropna());


# %%
df['Date'] = pd.to_datetime(df.Date)
df['SaleDayOfWeek'] = df.Date.dt.dayofweek
sns.countplot(df.SaleDayOfWeek);


# %%
sns.countplot(df.Rooms);


# %%
X = df[['Rooms', 'Distance', 'Propertycount', 'Postcode']]
y = np.log1p(df.Price.values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

from sklearn.ensemble import GradientBoostingRegressor

base_model = GradientBoostingRegressor(learning_rate=0.3, n_estimators=150, random_state=RANDOM_SEED).fit(X_train, y_train)

base_model.score(X_test, y_test)


# %%
y_base_pred = base_model.predict(X_test)


# %%
df['Size'] = pd.cut(df.Rooms, bins=[0, 2, 4, 100], labels=["Small", "Medium", "Large"])
df = df.drop(['Address', 'Date'], axis=1)


# %%
X = df.drop('Price', axis=1)
y = np.log1p(df.Price.values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


# %%
from sklearn.compose import make_column_transformer

transformer = make_column_transformer(    
    (RobustScaler(), ['Distance', 'Propertycount', 'Postcode','Rooms']),
    (OneHotEncoder(handle_unknown="ignore"), ['Size', 'SaleDayOfWeek', 'Type', 'Method', 'Regionname']),
    (OrdinalEncoder(
        categories=[X.CouncilArea.unique(), X.SellerG.unique(), X.Suburb.unique()], 
        dtype=np.int32
      ), ['CouncilArea', 'SellerG', 'Suburb']
    ),
)


# %%
transformer.fit(X_train)

X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)


# %%
X.shape


# %%
from sklearn.ensemble import GradientBoostingRegressor

final_model = GradientBoostingRegressor(learning_rate=0.3, n_estimators=150, random_state=RANDOM_SEED).fit(X_train, y_train)
final_model.score(X_test, y_test)


# %%
y_final_pred = final_model.predict(X_test)


# %%
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(22, 10))

ax1.set_xlim((0, 6000000))
ax1.set_ylim((0, 6000000))
ax1.set_title('No Preprocessing')
sns.scatterplot(np.expm1(y_test), np.expm1(y_base_pred), ax=ax1)

ax2.set_xlim((0, 6000000))
ax2.set_ylim((0, 6000000))
ax2.set_title('With Preprocessing')
sns.scatterplot(np.expm1(y_test), np.expm1(y_final_pred), ax=ax2);


