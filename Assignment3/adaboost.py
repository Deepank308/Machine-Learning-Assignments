"""
Deepank Agrawal
17CS30011
Assignment 3 - Adaboost algorithm using ID3 Decision trees as weak classifier.

- Run command:
$ python <file-name>

- Use python2 to run the code
- Make sure that train and test data is persent in the dataset folder
"""

import numpy as np
import csv

np.random.seed(0)

def load_data(data_path, test=False):
    """
    Load the data from csv file in (features, data-points) fashion
    """
    attributes = []
    target = []
    attr_dict = {}
    with open(data_path, 'rt') as f:
        data = csv.reader(f)
        flag = 0
        for row in data:
            if flag == 0 and test == False:
                for c in range(len(row) - 1):
                    attr_dict.update({c :row[c]})
                attributes = [[] for _ in range(len(row) - 1)]
                flag = 1
                continue
            if flag == 0 and test == True:
                attributes = [[] for _ in range(len(row) - 1)]
                flag = 1
            for i in range(len(row) - 1):
                attributes[i].append(row[i])
            target.append(row[-1])

    data = attributes[:]
    data.append(target)
    neglect = np.zeros(len(attributes))

    return data, attr_dict, attributes, neglect


def calc_entropy(target_val):
    """
    Calculates entropy using summation(-p_i*log(p_i)) for the given target set
    """
    el, count = np.unique(target_val, return_counts=True)
    entropy = np.sum([((-1.0*count[i])/np.sum(count))*np.log2((1.0*count[i])/np.sum(count)) for i in range(len(el))])

    return entropy


def calc_info_gain(data, attr_split_num):
    """
    Takes data and attribute to split about as input and return the information gain
    """
    total_entropy = calc_entropy(data[-1])

    el, count = np.unique(data[attr_split_num], return_counts=True)
    weighted_entropy = 0
    for i in range(len(el)):
        target = []
        for p in range(len(data[-1])):
            if data[attr_split_num][p] == el[i]:
                target.append(data[-1][p])
        weighted_entropy += ((1.0*count[i])/np.sum(count))*calc_entropy(target)

    info_gain = total_entropy - weighted_entropy

    return info_gain

    
def build_tree(data, orig_data, attr_dict, attributes, neglect, parent_node=None):
    """
    Takes in dataset and attributes as input and build the decision tree recursively
    """
    # if all are 'yes' or 'no'
    if len(np.unique(data[-1])) <= 1:
        return np.unique(data[-1])[0]
    elif len(data) == 0:
        return np.unique(orig_data[-1])[np.argmax(np.unique(orig_data[-1], return_counts=True)[1])]
    # if no attribute left
    elif all(i == 1 for i in neglect):
        return parent_node
    # else build tree recursively
    else:
        parent_node = np.unique(data[-1])[np.argmax(np.unique(data[-1], return_counts=True)[1])]
        item_gain = []
        # calculate max information gain for each attribute
        for i in range(len(attributes)):
            if neglect[i] == 0:
                item_gain.append(calc_info_gain(data, i))
            else:
                item_gain.append('-inf')
        max_gain_attr = np.argmax(item_gain)
        best_attr = attributes[max_gain_attr]

        neglect[max_gain_attr] = 1
        # sub tree
        tree = {attr_dict[max_gain_attr]:{}}
        # build the sub-trees
        for value in np.unique(data[max_gain_attr]):
            sub_data = [[] for _ in range(len(data))]
            for p in range(len(data[max_gain_attr])):
                if data[max_gain_attr][p] == value:
                    for i in range(len(data)):
                        sub_data[i].append(data[i][p])
            subtree = build_tree(sub_data, orig_data, attr_dict, attributes, np.copy(neglect), parent_node)

            tree[attr_dict[max_gain_attr]][value] = subtree
        
        return tree


def print_(tree, tabs_size=0):
    tabs = '\t'*tabs_size
    if type(tree) != dict:
        print(tabs + tree)
        return
    for key in tree:
        for k in tree[key]:
            print(tabs + key + ' --> ' + k + ':')
            print_(tree[key][k], tabs_size + 1)
    return

def predict_label(tree_, attr_dict, data, d):
    """
    Predict the label of a single data point
    """
    if type(tree_) == dict:
        key = tree_.keys()[0]
    else:
        return tree_
    while True:
        try:
            idx = attr_dict.keys()[attr_dict.values().index(key)]
            k = data[idx][d]
            tree_ = tree_[key][k]
            if type(tree_) != dict:
                return tree_
            key = tree_.keys()[0]
        except:
            tree_ = 'no'
            return tree_

    return -1


def predict_tree(data, tree, attr_dict, targets):
    """
    Predict the label of each data point and return the error count
    """
    correct_count = 0

    data_size = len(data[-1])
    attr_size = len(data) - 1

    # print(data_size)
    for d in range(data_size):
        y_pred = predict_label(tree, attr_dict, data, d)
        if y_pred == targets[d]:
            correct_count += 1

    return (data_size*1.0 - correct_count) / data_size



def main():
    # load training data
    data, attr_dict, attributes, neglect = load_data('./data3_19.csv')

    sample_size = int(len(data[0])*0.5)
    NUM_ITR = 3
    targets = data[-1]
    weights = np.ones(len(data[0]), dtype=float) / len(data[0])

    trees = []
    indices = np.arange(len(data[0]))
    alpha_list = []

    # run iterations to train the weak classifiers
    for i in range(NUM_ITR):
        sample = np.random.choice(indices, sample_size, p = weights, replace=False)
        
        # prepare the sample set
        sample_data = [[] for _ in range(len(data))]
        for s in sample:
            for d in range(len(data)):
                sample_data[d].append(data[d][s])

        # train the classifier
        attributes = sample_data[:-1]
        neglect_ = np.copy(neglect)
        tree = build_tree(sample_data, sample_data, attr_dict, attributes, neglect_)

        # calculate alpha
        sample_targets = sample_data[-1]
        e = predict_tree(sample_data, tree, attr_dict, sample_targets)
        alpha = 0.5*np.log((1 - e) / (e + 1e-6))
        alpha_list.append(alpha)

        # update the weights
        for s in sample:
            y_pred = predict_label(tree, attr_dict, data, s)

            if y_pred == targets[s]:
                weights[s] = weights[s]*np.exp(-1.0*alpha)
            else:
                weights[s] = weights[s]*np.exp(alpha)

        sum_weights = np.sum(weights)
        weights = weights / sum_weights

        # save the classifier
        trees.append(tree)

    # load the test set
    data, _, _, _ = load_data('./test3_19.csv', test=True)
    targets = data[-1]
    pos_count = 0
    # predict on the test set
    for d in range(len(data[-1])):
        pred = 0
        for i in range(NUM_ITR):
            y_pred = predict_label(trees[i], attr_dict, data, d)
            if y_pred == targets[d]:
                pred += alpha_list[i]
            else:
                pred -= alpha_list[i]
        
        if pred >= 0:
            pos_count += 1

    print('Test Accuracy: {}%'.format((100.0*pos_count) / len(data[-1])))

if __name__ == '__main__':
    main()