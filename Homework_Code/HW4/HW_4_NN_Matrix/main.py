import sklearn.datasets as sklearn_data
import sklearn.neighbors as sklearn_nn
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt


#Code for part vi
# code for different datasets commented on line 81-84

def compute_euclidean_dist(k, X, curr_sample, centroids):
    distances = np.zeros(k)
    for j in range(0, k):
        distances[j] = np.sqrt(np.sum(np.power(X[curr_sample, :] - centroids[j], 2)))
    return distances

def find_new_centroids(classifications, X, curr_centroids):
    new_centroids = np.zeros((k, total_features))
    new_loss = 0
    for i in range(0, k):
        data_points_curr_label = np.where(classifications == i)
        labeled_data_centroids = X[data_points_curr_label]
        new_centroids[i] = labeled_data_centroids.mean(axis=0)

        for curr_sample in range(0, labeled_data_centroids.shape[0]):
            new_loss += np.sum(np.power(labeled_data_centroids[curr_sample, :] - curr_centroids[i], 2))
    return new_loss, new_centroids

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



def execute_lloyds(X, k, number_samples):
    delta = 0
    tolerance = .0001
    max_iterations = 100
    classifications = np.zeros(number_samples, dtype=np.int64)

    Initial_Label = np.random.choice(number_samples, k)

    curr_centroids = X[Initial_Label, :]


    for curr_iter in range(0, max_iterations):
        for curr_sample in range(0, number_samples):
            distances = compute_euclidean_dist(k,X,curr_sample,curr_centroids)
            classifications[curr_sample] = np.argmin(distances)

        new_loss, new_centroids = find_new_centroids(classifications,X,curr_centroids)

        diff_in_loss = np.abs(delta - new_loss)
        # Stop if we are at basically the same curr_centroids
        if diff_in_loss <= tolerance:
            return classifications

        # update the variables for loss and centers
        delta = new_loss
        curr_centroids = new_centroids

    return classifications


def plot_data(data, cluster_labels):
    plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, s=3)
    plt.show()


if __name__ == '__main__':
    nsamples = 300
    k = 2
    nn = 4
    #big_rectangle = make_rectangle(-5, -5, 5, 5)
    #small_rectangle = make_rectangle(-2, -2, 2, 2)
    #data = np.concatenate((big_rectangle, small_rectangle), axis=0)
    #data, dataclusters = sklearn_data.make_moons(n_samples=nsamples, noise=.011, random_state=1)
    data, dataclusters = sklearn_data.make_circles(n_samples=nsamples, noise=.011, random_state=1)

    G_r = sklearn_nn.kneighbors_graph(data, nn).toarray()
    nsamples = len(data)
    W = np.empty(shape=(nsamples,nsamples))
    for i in range(nsamples):
        for j in range(nsamples):
            W[i][j] = int(max(G_r[i][j], G_r[j][i]))
    D = np.zeros((nsamples,nsamples))
    for i in range(nsamples):
        D[i][i] = W[i].sum()
    L = D - W
    eigenValues, eigenVectors = linalg.eig(L)

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]

    V = eigenVectors[:,-k:]
    number_samples = V.shape[0]
    total_features = V.shape[1]
    classifications = execute_lloyds(V, k, number_samples)
    plot_data(data, classifications)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
