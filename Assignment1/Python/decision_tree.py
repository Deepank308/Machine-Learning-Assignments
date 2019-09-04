"""
Deepank Agrawal
17CS30011
Assignment 1 - Decision Tree using ID3 algorithm

- Run command:
$ python <file-name>

- Make sure that train and test data is persent in the dataset folder
"""

import numpy as np
import csv

def load_data(data_path):
    """
    Load the data from csv file in (features, data-points) fashion
    """
    attributes = []
    target = []
    attr_dict = {}
    with open(data_path, 'rt') as f:
        data = csv.reader(f)
        i = 0
        for row in data:
            if i == 0:
                for c in range(len(row) - 1):
                    attr_dict.update({c :row[c]})
                attributes = [[] for _ in range(len(row) - 1)]
                i = 1
                continue
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
                if data[maxc_gain_attr][p] == value:
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


def predict(data, tree, attr_dict):
    correct_count = 0
    data_size = len(data[-1])
    attr_size = len(data) - 1
    for d in range(data_size):
        tree_ = tree
        key = tree_.keys()[0]
        while True:
            idx = attr_dict.keys()[attr_dict.values().index(key)]
            k = data[idx][d]
            tree_ = tree_[key][k]
            if type(tree_) != dict:
                if tree_ == data[-1][d]:
                    correct_count += 1
                break
            key = tree_.keys()[0]
    
    print('Accuracy: {}%'.format((100.0*correct_count) / data_size))
    return



def main():
    # load data
    data, attr_dict, attributes, neglect = load_data('../dataset/dataset.csv')

    # build and print tree
    tree = build_tree(data, data, attr_dict, attributes, neglect)
    print_(tree)

    # test the decision tree on given dataset
    predict(data, tree, attr_dict)


if __name__ == '__main__':
    main()