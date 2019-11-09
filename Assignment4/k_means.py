"""
Deepank Agrawal
17CS30011
Assignment 4 - KMeans Clustering

- Run command:
$ python3 <file-name>

- Make sure that train data is persent in the same folder
"""

import numpy as np
import pandas as pd

def load_dataset(data_path):
    """
    loads the data set and returns the data & class labels
    """
    data = pd.read_csv(data_path, squeeze=True)
    label = data[data.columns[-1]]
    classes = label.unique()
    
    return data, classes


def distance(point, center):
    """
    returns euclidean distance between two data points
    """
    p = point[0:4].astype('float64')
    c = center[0:4].astype('float64')
    
    return np.linalg.norm(p - c)


def update_centroid(centers, clusters, data):
    """
    returns the mean center of clusters
    """
    for i, c_d in enumerate(clusters):
        average = [0, 0, 0, 0]
        for idx in c_d:
            average = average + data.values[idx][0:4]
        centers.values[i] = average / (len(c_d) + 1e-7)
                    
    return centers


def train_kmeans(data, classes, centroids, NUM_ITR=10):
    """
    trains the kmeans for NUM_ITR iterations and returns cluster & corres. mean
    """
    print('Training KMeans...')
    for i in range(NUM_ITR):
        print('Iteration #: {}'.format(i + 1))
        clusters = [np.array([], dtype=np.int64) for i in range(classes.size)]
        for idx in data.index:
            dist = np.array([])
            # find the nearest cluster mean
            for c in centroids.values:
                dist = np.append(dist, distance(data.values[idx], c))
            num = np.argmin(dist)
            clusters[num] = np.append(clusters[num], idx)
        # update mean of clusters
        centroids = update_centroid(centroids, clusters, data)
    
    return clusters, centroids 


def calc_jaccard_index(data, clusters, classes):
    """
    calculate Jaccard distance between every pair of ground truth and predicted clusters
    """
    ground_truth_clusters = []
    for c in classes:
        ground_truth = data[data[data.columns[-1]].values == c].index
        ground_truth_clusters.append(np.array(ground_truth))
        
    jaccard_distance = np.zeros([len(ground_truth_clusters), len(clusters)])
    
    # calculate Jaccard Index
    for i, gtc in enumerate(ground_truth_clusters):
        for j, c in enumerate(clusters):
            union = np.union1d(gtc, c).shape[0]
            intersection = np.intersect1d(gtc, c).shape[0]
            jaccard_distance[j, i] = (union - intersection) / (union + 1e-7)
    
    return jaccard_distance


def print_(dist):
    """
    prints the Jaccard distances
    """
    tabs = '\t'*2
    ws = ' '*5

    print('\n------------Jaccard Distance------------')
    print(tabs + 'Ground Truth Clusters')
    for i, dis in enumerate(dist):
        print('Cluster {}:'.format(i), end=' ')
        for d in dis:
            print('{0:.4f}'.format(d), end=ws)
        print('\n')


def main():
    data, classes = load_dataset('./dataset/dataset.csv')
    
    # sample random cluster means
    sample = data.sample(n = classes.size, random_state=0)
    centroids = sample.drop(columns=[data.columns[-1]])
    
    # train the KMeans
    clusters, centroids = train_kmeans(data, classes, centroids)
    
    print('\n---------------------Cluster means----------------------')
    for idx, c in enumerate(centroids.values):
        print('Cluster {}: {}'.format(idx, c))

    # find Jaccard distances
    jaccard_distance = calc_jaccard_index(data, clusters, classes)
    
    print_(jaccard_distance)


if __name__ == '__main__':
    main()
