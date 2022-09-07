#!/usr/bin/env python
'''
This script contains various functions used in this project
Author: Sai Venkata Krishnaveni Devarakonda
Date: 04/10/2022
'''

import utilities
str_path_1b_program = './data_2c2d3c3d_program.txt'

#splitting data into train and test data sets
str_path_1b_program = './data_2c2d3c3d_program.txt'
features,targets = utilities.Load_data(str_path_1b_program)
arr3d_train_features = features[0:90]
arr3d_test_features = features[90:120]
arr1d_train_targets = targets[0:90]
arr1d_test_targets = targets[90:120]


X_train, X_test, y_train, y_test = arr3d_train_features,arr3d_test_features,arr1d_train_targets,arr1d_test_targets

for i in range(1,9):
    clf = utilities.DecisionTree()
    clf.fit(X_train, y_train,i)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    acc1 = utilities.accuracy(y_train, y_train_pred)
    acc = utilities.accuracy(y_test, y_test_pred)
    print('Train Accuracy for d=' +str(i)+' is '+str(acc1))
    print('Test Accuracy for d=' +str(i)+' is '+str(acc))
    print('\n')