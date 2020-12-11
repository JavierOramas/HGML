# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# %%
RANDOM_SEED = 42


# %%
auto_df = pd.read_csv("auto.csv")
auto_df.shape


# %%
keep_columns = ['mpg', 'horsepower', 'weight', 'acceleration', 'origin']


# %%
auto_df['is_american'] = (auto_df.origin == 1).astype(int)


# %%
def create_regression_dataset(
    df,
    columns=['weight', 'horsepower', 'mpg']
):
    all_columns = columns.copy()
    all_columns.append('acceleration')

    print(df.columns)
    reg_df = df[all_columns]
    
    reg_df = StandardScaler().fit_transform(reg_df[all_columns])
    reg_df = pd.DataFrame(reg_df, columns=all_columns)

    return reg_df[columns], reg_df.acceleration

def create_classification_dataset(df):
    columns = ['mpg', 'weight', 'horsepower']
    
    x = df[columns]
    x = StandardScaler().fit_transform(x)
    x = pd.DataFrame(df, columns=columns) 

    return x, df.is_american


# %%
from sklearn.model_selection import KFold, cross_val_score

def eval_model(model, x,y,score):
    cv = KFold(n_splits=10, random_state=RANDOM_SEED)
    results = cross_val_score(model,x,y,cv=cv, scoring=score)
    return np.abs(results.mean())

def eval_classifier(model, x,y):
    return eval_model(model, x,y, score='accuracy')

def eval_regressor(model, x,y):
    return eval_model(model,x,y,score='neg_mean_squared_error')


# %%
from sklearn.linear_model import LinearRegression

x,y = create_regression_dataset(auto_df, columns=['horsepower'])

reg = LinearRegression()
eval_regressor(reg,x,y)


# %%
x,y = create_regression_dataset(auto_df)

reg = LinearRegression()
eval_regressor(reg,x,y)


# %%
from sklearn.linear_model import Ridge
x,y = create_regression_dataset(auto_df)
reg = Ridge(alpha=0.0005, random_state=RANDOM_SEED)

eval_regressor(reg,x,y)


# %%
from sklearn.linear_model import LogisticRegression

x,y = create_classification_dataset(auto_df)
clf = LogisticRegression(solver='lbfgs')
eval_classifier(clf,x,y)


# %%
from sklearn.neighbors import KNeighborsClassifier

x,y = create_classification_dataset(auto_df)

clf = KNeighborsClassifier(n_neighbors=24)
eval_classifier(clf,x,y)


# %%
from sklearn.naive_bayes import GaussianNB

x,y = create_classification_dataset(auto_df)
clf = GaussianNB()
eval_classifier(clf,x,y)


# %%
from sklearn.tree import DecisionTreeRegressor

x,y = create_regression_dataset(auto_df)

reg = DecisionTreeRegressor()
eval_regressor(reg,x,y)


# %%
from sklearn.ensemble import RandomForestRegressor

x,y = create_regression_dataset(auto_df)
reg = RandomForestRegressor(n_estimators=50)
eval_regressor(reg,x,y)


# %%
from sklearn.ensemble import GradientBoostingRegressor

x,y = create_regression_dataset(auto_df)

reg = GradientBoostingRegressor(n_estimators=100)
eval_regressor(reg,x,y)


# %%
from sklearn.svm import SVR

x,y = create_regression_dataset(auto_df)

reg = SVR(gamma='auto', kernel='rbf', C=4.5)
eval_regressor(reg,x,y)


