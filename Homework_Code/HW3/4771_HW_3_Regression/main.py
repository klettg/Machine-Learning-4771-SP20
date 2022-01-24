# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import csv
import os

import scipy.io
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
import sklearn.neighbors as neighbors
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer


def read_data(file_path):
    mat = scipy.io.loadmat(file_path)
    train_x_data = mat['trainx']  # variable in mat file
    train_x_columns = [f'col_{num}' for num in range(len(train_x_data[0]))]
    train_x_rows = [f'index_{num}' for num in range(len(train_x_data))]
    train_x_df = pd.DataFrame(train_x_data, columns=train_x_columns, index=train_x_rows)
    train_y_data = mat['trainy']
    y_column = ['year']
    train_y_index = [f'index_{num}' for num in range(len(train_y_data))]
    train_y_df = pd.DataFrame(train_y_data, columns=y_column, index=train_y_index)
    test_x_data = mat['testx']  # variable in mat file
    test_x_columns = [f'col_{num}' for num in range(len(test_x_data[0]))]
    test_x_rows = [f'index_{num}' for num in range(len(test_x_data))]
    test_x_df = pd.DataFrame(test_x_data, columns=test_x_columns, index=test_x_rows)
    #total_df = pd.concat([x_df, y_df], axis=1)
    # print(total_df.head())
    return train_x_df, train_y_df, test_x_df

#ignore this function
def knn():
    train_x_df, train_y_df, test_x_df = read_data("/Users/Griffin/Downloads/MSdata.mat")
    max_year = train_y_df['year'].max()
    min_year = train_y_df['year'].min()
    y_input = train_y_df.year
    x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(train_x_df, y_input, test_size=0.33, random_state=42)
    knn = neighbors.KNeighborsClassifier(n_neighbors=20)
    knn.fit(x_train_split, y_train_split)
    test_pred = knn.predict(x_test_split)
    test_pred[test_pred > max_year] = max_year
    test_pred[test_pred < min_year] = min_year
    mae = mean_absolute_error(y_test_split, test_pred)
    print(mae)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_x_df, train_y_df, test_x_df = read_data("/Users/Griffin/Downloads/MSdata.mat")
    max_year = train_y_df['year'].max()
    min_year = train_y_df['year'].min()
    y_input = train_y_df.year
    x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(train_x_df, y_input, test_size=0.1, random_state=42)

    r= SGDRegressor(loss='huber', epsilon=.05, max_iter=10000, tol=1e-5, penalty='l1', alpha=0)
    regPipe = make_pipeline(PowerTransformer(), r)
    regPipe.fit(train_x_df, y_input)
    pred = regPipe.predict(x_test_split)
    pred[pred > max_year] = max_year
    pred[pred < min_year] = min_year
    pred = np.rint(pred)
    mae = mean_absolute_error(y_test_split, pred)

    pred_real = regPipe.predict(test_x_df)
    pred_real[pred_real > max_year] = max_year
    pred_real[pred_real < min_year] = min_year
    pred_real = np.rint(pred_real)
    file = open('results_sgdr.csv', 'w')
    file.write('dataid,prediction\n')
    for i in range(len(pred_real)):
        file.write("{}, {} \n".format(i+1, pred_real[i]))
    file.close()
    print(mae)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
