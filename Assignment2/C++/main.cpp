//
//  main.cpp
//  naive_bayes
//
//  Created by Ayush Tiwari, 17CS10056 on 02/09/19.
//

/*
  To compile : g++ -std=c++14 17CS10056_2.cpp
*/

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <map>
#include <iomanip>

using namespace std;

vector<string> col_header;

vector<vector<int> > train_feat;
vector<int> train_labels;

vector<vector<int> > test_feat;
vector<int> test_labels;

vector<int> class_count(2);

vector<double> p_class(2);

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

void get_table(vector<vector<int> > &table, int col) {

    for(int i=0; i<train_feat.size(); i++) {
        int feat_val = train_feat[i][col];
        int label_val = train_labels[i];

        table[feat_val-1][label_val]++;
    }

}

void construct_tables(vector<vector<vector<int> > >  &tables) {

    int num_feat = int(train_feat[0].size());

    tables.resize(num_feat);

    for(int i=0; i<num_feat; i++) {

        tables[i].resize(5);
        for(int j=0; j<5; j++)
            tables[i][j].resize(2);

        get_table(tables[i], i);

    }

}

void print_table(vector<vector<int> > &table, int feat) {

    cout << "    " << "  0" << " " << "  1" << "\n\n";

    cout << string(12, '-') << "\n\n";

    for(int i=0; i<table.size(); i++) {
        cout << setw(3) << i+1 << " ";
        for(int j=0; j<table[i].size(); j++) {
            cout << setw(3) << table[i][j] << " ";
        }
        cout << endl;
    }

}

void print_tables(vector<vector<vector<int> > > &tables) {

    for(int i=0; i<tables.size(); i++) {

        cout << string(15, '*') << "\n\n";
        cout << "     " << string("X") + to_string(i+1) << "\n\n";
        print_table(tables[i], i);
        cout << "\n\n";

    }

}

int predict(vector<vector<vector<int> > > &tables, vector<int> features) {


    vector<double> prob(2);
    prob[0]=1;
    prob[1]=1;

    for(int i=0; i<features.size(); i++) {
        prob[0] *= double(tables[i][features[i]-1][0]+1)/(class_count[0]+2);
    }

    for(int i=0; i<features.size(); i++) {
        prob[1] *= double(tables[i][features[i]-1][1]+1)/(class_count[1]+2);
    }


    prob[0] *= (double(class_count[0])/train_feat.size());
    prob[1] *= (double(class_count[1])/train_feat.size());

    return prob[0]>prob[1]? 0:1;

}

void predict_all(vector<vector<vector<int> > > &tables) {

    int correct = 0;
    int total = 0;

    cout << "Predictions\n";

    for(int i=0; i<test_feat.size(); i++) {

        cout << string(15, '*') << "\n\n";

        int prediction = predict(tables, test_feat[i]);

        for(int j=0; j<test_feat[i].size(); j++) {
            cout << test_feat[i][j] << " ";
        }
        cout << "\n\n" << "Predicted : " << prediction << endl;
        cout << "Actual : " << test_labels[i] << "\n\n";

        if(test_labels[i] == prediction) correct++;

        total++;
    }

    cout << string(15, '*') << "\n\n";

    cout << "Correct : " << correct << " / " << total << endl;
    cout << "Accuracy : " << double(correct)/total << endl;

}

void pre_process(string train_filepath, string test_filepath) {

    ifstream file;
    file.open(train_filepath);

    string line;
    getline(file, line);

    col_header = split(line.substr(1, line.size()-3), ',');

    while(getline(file, line)) {
        vector<string> data_string = split(line.substr(1, line.size()-3), ',');
        vector<int> data_int;

        for(auto &data:data_string) {
            data_int.push_back(data[0]-'0');
        }

        if(data_int[0]==0) class_count[0]++;
        else class_count[1]++;

        train_labels.push_back(data_int[0]);
        train_feat.push_back(vector<int>(data_int.begin()+1, data_int.end()));
    }

    file.close();

    file.open(test_filepath);
    getline(file, line);        // Skipping the header

    while(getline(file, line)) {
        vector<string> data_string = split(line.substr(1, line.size()-3), ',');
        vector<int> data_int;

        for(auto &data:data_string) {
            data_int.push_back(data[0]-'0');
        }

        test_labels.push_back(data_int[0]);
        test_feat.push_back(vector<int>(data_int.begin()+1, data_int.end()));
    }

    file.close();
}

int main(int argc, const char * argv[]) {

    pre_process("dataset/train.csv", "dataset/test.csv");

    vector<vector<vector<int> > > tables;
    
    construct_tables(tables);
    print_tables(tables);
    predict_all(tables);

    return 0;
}
