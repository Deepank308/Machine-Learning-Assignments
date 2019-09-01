"""
Deepank Agrawal
17CS30011
Assignment 3
Naive Bayes Classifier with Laplacian smoothing

- Run command:
$ python <file-name>

- Make sure that train and test data is persent in the same folder
"""

import numpy as np
import pandas as pd

def load_dataset(train_path, test_path):

    train = pd.read_csv(train_path, squeeze=True)
    
    test = pd.read_csv(test_path, squeeze=True)

    return train, test

def split_class(data, label, classes):
    """
    split the dataset classwise
    """
    class_data = []
    for c in classes:
        class_data.append(data[data[label] == c])

    return class_data

def train_model(data):
    """
    Calculates class probability and conditional probabilities for each attribute type
    """
    label = data.columns.values[0]
    attr_size = data.columns.shape[0] - 1

    # find the number of classes
    classes = data[label].unique()
    classes_data = split_class(data, label, classes)
    apriori_prob = []
    classes_cond_prob = []

    # probability calculation
    for class_data in classes_data:

        class_cond = []
        class_size = class_data.shape[0] + 5

        for attr in range(1, attr_size + 1):

            attribute = class_data.columns.values[attr]
            attr_data = class_data[attribute].value_counts()

            for i in range(1, 6):

                if i not in attr_data.keys():
                    attr_data[i] = 0

            # laplacian smoothing
            attr_data = attr_data.sort_index() + 1
            attr_data = attr_data / class_size
            class_cond.append(attr_data)

        apriori_prob.append(class_data.shape[0]*1.0 / data.shape[0])
        classes_cond_prob.append(class_cond)

    return classes_cond_prob, apriori_prob

def predict(data, class_cond_prob, apriori_prob):
    """
    calculates accuracy over given test set
    """
    label = data.columns.values[0]
    label = data[label]

    class_pred_prob = np.ones((data.shape[0], len(class_cond_prob)))
    for idx, row in data.iterrows():
        for c in range(len(class_cond_prob)):
            for attr, r_data in enumerate(row):
                if attr == 0: continue
                class_pred_prob[idx][c] *= class_cond_prob[c][attr - 1][r_data]
    
    class_pred_prob *= apriori_prob
    pred_label = np.argmax(class_pred_prob, axis=1)

    return pred_label

def main():

    train_data, test_data = load_dataset('data2_19.csv', 'test2_19.csv')
    class_cond_prob, apriori_prob = train_model(train_data)

    # predict on train set
    train_pred = predict(train_data, class_cond_prob, apriori_prob)
    train_acc = np.sum(train_pred == train_data[train_data.columns.values[0]])*100.0
    print('Train Accuracy: {} %'.format(train_acc / train_data.shape[0]))

    # predict on test set
    test_pred = predict(test_data, class_cond_prob, apriori_prob)
    test_acc = np.sum(test_pred == test_data[test_data.columns.values[0]])*100.0
    print('Test Accuracy: {} %'.format(test_acc / test_data.shape[0]))



if __name__ == '__main__':
    main()