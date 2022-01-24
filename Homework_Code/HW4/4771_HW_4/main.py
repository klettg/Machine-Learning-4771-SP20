import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as sklearn_data

# Code used for C.i
# (I rewrote parts of the lloyd's algorithm in other .py for part vi to run in more clean way and for the L,W,D,V
# Decomposition. Run different datasets via commented code on lines 100-104

def execute_lloyds(df, k):
    max_x = df[xval].max()
    min_x = df[xval].min()
    max_y = df[yval].max()
    min_y = df[yval].min()
    means_curr = []
    means_old = []
    for i in range(k):
        means_curr.append([random.uniform(min_x, max_x), random.uniform(min_y, max_y), f"ClusterLabel{i}"])
        means_old.append([random.uniform(min_x, max_x), random.uniform(min_y, max_y), f"ClusterLabel{i}"])
    row_indices = [f'index_{num}' for num in range(len(means_old))]
    means_curr_df = pd.DataFrame(means_curr, columns=['X_val', 'Y_val', clusterlabel], index=row_indices)
    means_old_df = pd.DataFrame(means_old, columns=['X_val', 'Y_val', clusterlabel], index=row_indices)
    while not has_converged(means_old_df, means_curr_df):
        #set curr -> old
        means_old_df = means_curr_df
        # assign to nearest cluster
        df = assign_nearest_cluster(means_old_df, df)
        #find new centers
        means_curr_df = find_centers(df)

    means_curr_df['clusterColor'] = means_curr_df.apply(lambda x: 1 if x.iloc[2] == 'ClusterLabel1' else 0, axis=1)
    df['clusterColor'] = df[clusterlabel].apply(lambda x: 1 if x == 'ClusterLabel1' else 0)
    colors = np.where(df.clusterColor > 0, 'r', 'k')
    colors_means = np.where(means_curr_df.clusterColor > 0, 'r', 'k')
    ax = df.plot.scatter(xval,yval,
                    c=colors)
    means_curr_df.plot.scatter(xval,yval,
                    c=colors_means, ax=ax, marker='x')
    plt.show()


def has_converged(means_old_df, means_curr_df):
    means_old_clean = means_old_df.drop(columns=[clusterlabel])
    means_curr_clean = means_curr_df.drop(columns=[clusterlabel])
    old_means_set = set()
    curr_means_set = set()
    for index, row in means_old_clean.iterrows():
        x = tuple(row)
        old_means_set.add(x)
    for index, row in means_curr_clean.iterrows():
        x = tuple(row)
        curr_means_set.add(x)
    return old_means_set == curr_means_set


def assign_nearest_cluster(old_means_df, df):
    df = df.drop(columns=clusterlabel)

    df[clusterlabel] = df.apply(lambda row: euclidean_dist(row, old_means_df), axis=1)
    return df


def euclidean_dist(row, old_means_df):
    #row = row.to_frame()
    #row = row.transpose()
    distances = np.linalg.norm(row[column_labels] - old_means_df[column_labels], axis=1)
    index = np.argmin(distances)
    label = old_means_df.iloc[index,:]
    label = label[clusterlabel]
    return label


def find_centers(df):
    updatedMeans = df.groupby([clusterlabel]).mean()
    means_curr = []
    for index, row in updatedMeans.iterrows():
        means_curr.append([row.iloc[0], row.iloc[1], index])
    row_indices = [f'index_{num}' for num in range(len(means_curr))]
    updatedMeans_df = pd.DataFrame(means_curr, columns=['X_val', 'Y_val', clusterlabel], index=row_indices)
    return updatedMeans_df

def make_rectangle(min_x, min_y, max_x, max_y):
    rec = []
    x, y = min_x, min_y
    for dx, dy in (1, 0), (0, 1), (-1, 0), (0, -1):
        while x in range(min_x, max_x+1) and y in range(min_y, max_y+1):
            rec.append((x, y))
            x += dx
            y += dy
        x -= dx
        y -= dy
    return np.asarray(rec)


if __name__ == '__main__':
    xval = 'X_val'
    yval = 'Y_val'
    column_labels = [xval, yval]
    clusterlabel = 'Cluster_Label'
    #---Different datasets can be uncommented and run from here#
    #data = [[1, 0, 'test'], [-1, 0, 'test'], [0, 1, 'test'], [0, -1, 'test'], [3, 0, 'test'], [-3, 0, 'test'], [0, 3, 'test'], [0, -3, 'test']]
    #data, dataclusters = sklearn_data.make_moons(n_samples=100, noise=.02, random_state=1)
    #big_rectangle = make_rectangle(-5, -5, 5, 5)
    #small_rectangle = make_rectangle(-2, -2, 2, 2)
    #data = np.concatenate((big_rectangle, small_rectangle), axis=0)
    data, dataclusters = sklearn_data.make_circles(n_samples=100, noise=.02, random_state=1)
    row_indices = [f'index_{num}' for num in range(len(data))]
    df = pd.DataFrame(data, columns=[xval, yval], index=row_indices)
    df[clusterlabel] = 'test'
    k = 2
    execute_lloyds(df, k)