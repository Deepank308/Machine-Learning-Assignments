//
//  main.cpp
//  DesicionTree
//
//  Created by Ayush Tiwari, 17CS10056 on 21/08/19.
//  Copyright © 2019 Ayush Tiwari, 17CS10056. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cstdlib>
#include <random>

using namespace std;

// Global variable declaration begins

vector<string> col_header;
vector<vector<string> > train_data;
vector<vector<string> > test_data;

// Global variable declaration ends

/**
 *  Splits a string at positions where the given character is found
 *
 *  @param line string to be split
 *  @param c character at which to split
 *  @return words that the string was split into
 */
vector<string> split(string line, char c) {

    vector<string> words;

    string::iterator iter = line.begin();
    string::iterator pos;

    while(iter!=line.end()) {
        pos = find(iter, line.end(), c);
        words.push_back(string(iter, pos));
        if(pos==line.end()) break;
        iter = pos+1;
    }

    return words;
}

/**
 *  Finds the unique values in a given column
 *
 *  @param rows is the table
 *  @param col is the column number
 *  @return values of the distinct elements
 */
vector<string> unique_values(vector<vector<string> > &rows, int col) {

    set<string> values;

    for(int i=0; i<rows.size(); i++)
        values.insert(rows[i][col]);

    return vector<string>(values.begin(), values.end());
}


/**
 *  Counts the total number of distinct labels in a dataset
 *
 *  @param rows is the table
 *  @return count of the distinct labels
 */
map<string, int> class_counts(vector<vector<string> > &rows) {

    map<string, int> count;

    for(int i=0; i<rows.size(); i++)
        count[rows[i].back()]++;

    return count;
}

//// Criteria on which divisions will take place
class Criteria {

    int column;
    string value;

public:
    Criteria(int column, string value) {
        this->column = column;
        this->value = value;
    }

    /**
     * @return true if the given example satisfies this criteria
     */
    bool match(vector<string> example) {
        string val = example[this->column];
        return val == this->value;
    }

    operator std::string() const{
        string condition = "==";
        return col_header[this->column] + string(" ") + condition + string(" ") + this->value + string("?");
    }
};

/**
 *  Partitions the dataset based on the given criteria
 *
 *  @param rows is the dataset
 *  @param criteria is the criteria on which division takes place
 *  @param true_rows is the reference in which the rows that satisfy the criteria will be placed
 *  @param false_rows is the reference in which the rows that don't satisfy the criteria will be placed
 */
void data_partition(vector<vector<string> > &rows, Criteria *criteria, vector<vector<string> > &true_rows, vector<vector<string> > &false_rows) {

    for(int i=0; i<rows.size(); i++) {

        if(criteria->match(rows[i]))
            true_rows.push_back(rows[i]);
        else
            false_rows.push_back(rows[i]);
    }
}

/**
 *  Calculates the gini index of a given sample
 *
 *  @param rows is the sample
 *  @return impurity is the gini index for the given sample
 */
double gini(vector<vector<string> > rows) {

    map<string, int> counts = class_counts(rows);
    double impurity = 1.0;

    map<string, int>::iterator iter = counts.begin();

    while(iter!=counts.end()) {

        double prob_of_label = iter->second / float(rows.size());
        impurity -= prob_of_label*prob_of_label;

        iter++;
    }
    return impurity;
}

/**
 * Calculates the information gain provided the divisions
 *
 * @param left is the subset that does not satisfy the partitioning criteria
 * @param right is the subset that does satisfy the partitioning criteria
 * @param current is the current impurity
 */
double info_gain(vector<vector<string> > left, vector<vector<string> > right, double current) {

    double p = float(left.size())/(left.size()+right.size());

    return current - p * gini(left) - (1-p) * gini(right);

}

/**
 *  Finds the best criteria for splitting using information gain
 *
 *  @param rows is the sample
 *  @param best_gain is the reference which will store the best case info_gain
 *  @return best_criteria is the best criteria for splitting
 */
Criteria* findBestSplit(vector<vector<string> > rows, double &best_gain) {

    best_gain = 0;
    Criteria *best_criteria = new Criteria(0, "");
    double current_uncertainity = gini(rows);

    unsigned long n_features = rows[0].size()-1;

    for(int col=0; col<n_features; col++) {
        vector<string> values = unique_values(rows, col);

        for(auto &val:values) {

            Criteria *criteria = new Criteria(col, val);
            vector<vector<string> > true_rows, false_rows;

            data_partition(rows, criteria, true_rows, false_rows);

            if(true_rows.size()==0 || false_rows.size()==0) continue;

            double gain = info_gain(true_rows, false_rows, current_uncertainity);

            if(gain >= best_gain) {
                best_gain = gain;
                *best_criteria = *criteria;
            }

        }
    }

    return best_criteria;
}

////Structure describing a tree node
struct TreeNode {
    Criteria *criteria;
    TreeNode *left, *right;
    map<string, int> predictions;
    bool isLeaf;

    /**
     * Constructor for the leaves of the tree
     */
    TreeNode(map<string, int> predictions) {
        this->predictions = predictions;
        this->criteria = NULL;
        this->left = NULL;
        this->right = NULL;
        this->isLeaf = true;
    }

    /**
     *  Constructor for the internal nodes
     */
    TreeNode(Criteria *criteria, TreeNode *left, TreeNode *right) {
        this->criteria = criteria;
        this->left = left;
        this->right = right;
        isLeaf = false;
    }
};

/**
 *  Builds the tree recursively
 *
 *  @param rows is the sample
 *  @param depth is the depth of the node to be built
 *  @param max_depth is for tree pruning
 */
TreeNode *build_tree(vector<vector<string> > &rows, int depth, int max_depth=100) {

    double gain = 0;
    Criteria *criteria = findBestSplit(rows, gain);

    if(gain == 0 || depth > max_depth) return new TreeNode(class_counts(rows));

    vector<vector<string> > right_rows, left_rows;
    data_partition(rows, criteria, right_rows, left_rows);

    TreeNode *right_branch = build_tree(right_rows, depth+1, max_depth);
    TreeNode *left_branch = build_tree(left_rows, depth+1, max_depth);

    return new TreeNode(criteria, right_branch, left_branch);

}

/**
 *  Prints the prediction map properly
 *
 *  @param predictions takes the prediction map of a leaf
 */
void print_predictions(map<string, int> &predictions) {
    map<string, int>::iterator iter = predictions.begin();

    cout << "{";
    while(iter!=predictions.end()) {
        cout << iter->first << " : " << iter->second << ", ";
        iter++;
    }
    cout << "\b\b" << "}";
}

/**
 *  Prints the tree in an awesome way
 *
 *  @param root is the root of the tree/subtree
 *  @param spacing is the no. of leading spaces to be added
 */
void print_tree(TreeNode *root, string spacing) {

    if(root->isLeaf==true) {
        cout << (spacing + string("Predict "));
        print_predictions(root->predictions);
        cout << endl;
        return;
    }

    cout << (spacing + "Is " + string(*(root->criteria))) << endl;
    cout << (spacing + string("--> True:")) << endl;
    print_tree(root->left, spacing + "  ");

    cout << (spacing + string("--> False:")) << endl;
    print_tree(root->right, spacing + "  ");

}

/**
 *  Generates a random test data
 *
 *  @param rows is the sample
 *  @return res is the feature vector
 */
vector<string> gen_test(vector<vector<string> > &rows) {

    unsigned long int n_features = rows[0].size() - 1;
    vector<string> res, unique_features;

    for(int col=0; col<n_features; col++) {
        unique_features = unique_values(rows, col);
        unsigned long int r = random()%(unique_features.size());
        res.push_back(unique_features[r]);
    }

    return res;
}

/**
 *  Predicts the class of a given feature row
 *
 *  @param root is the root of the built tree
 *  @param features is the feature values for predicting
 *  @return a map of the form {no : count, yes : count}
 */
map<string, int> predict_class(TreeNode *root, vector<string> features) {

    if(root->isLeaf) return root->predictions;

    if(root->criteria->match(features)) return predict_class(root->right, features);
    return predict_class(root->left, features);
}

/**
 *  Reads and stores the data in global variable training_set
 *
 *  @param filepath is the path of the dataset file
 */
void pre_process(string filepath) {

    ifstream file;
    file.open(filepath);

    string line;
    getline(file, line);

    col_header = split(line, ',');

    while(getline(file, line))
        train_data.push_back(split(line, ','));

}

/**
 *  Prints a sample dataset for visualisation
 */
void print_sample(int n) {

    for(auto &e : col_header) cout << e << "\t";
    cout << endl;

    set<int> sample;

    while(sample.size() < (n < train_data.size() ? n:train_data.size()))
        sample.insert(rand()%train_data.size());


    for(set<int>::iterator iter=sample.begin(); iter!=sample.end(); iter++) {
        for(auto &e:train_data[*iter]) cout << e << "\t";
        cout << endl;
    }

}

////Driver function
int main(int argc, const char * argv[]) {

    pre_process("dataset.csv");                             // fill the global variable training_set
    print_sample(5);                                        // print a sample of dataset for                                                              // visualisation

    TreeNode *root = build_tree(train_data, 0, 50);         // construct the tree

    print_tree(root, string(""));                           // print the tree

    cout << endl;

    for(int i=0; i<5; i++) {
        vector<string> test = gen_test(train_data);
        cout << "Testing data " << i << ") ";
        for(int i=0; i<test.size(); i++) {
            cout << test[i] << " ";
        }
        cout << "\n";
        cout << "Prediction : ";
        map<string, int> pred = predict_class(root, test);
        print_predictions(pred);
        cout << "\n\n";
    }

    return 0;
}
