#!/usr/bin/env python
'''
This script contains various functions used in this project
Author: Sai Venkata Krishnaveni Devarakonda
Date: 04/08/2022
'''

import numpy as np
import math
import re
import random


# Loading homework1 train data separated as features and targets
def Load_data(str_path_1b_program):
    '''
    This function loads train data(demographic data height,weight,age) from homework1 and separates features and targets
    inputs: str path to train data.txt
    outputs: numpy arrays of targets,features
    '''
    # initialize empty lists to gather features and targets
    features = []
    targets = []
    # read lines in txt file as string
    with open(str_path_1b_program) as f:
        for line in f:
            data = line
            # remove parenthesis
            data_tmp = re.sub(r"[\([{})\]]", "", data)
            # extract list of 1 feature
            lsFeature_tmp = [float(data_tmp.split(',')[0]), float(data_tmp.split(',')[1]), int(data_tmp.split(',')[2])]
            # extract target
            lsTarget_tmp = [data_tmp.split(',')[3][1]]
            features.append(lsFeature_tmp)
            targets.append(lsTarget_tmp)
    features = np.array(features)
    targets = np.array(targets)
    return features,targets


# calculating entropy
def entropy1(array_target_labels):
    '''
    This function calculates entropy for given labels
    inputs: numpy array of labels
    outputs: entropy
    '''
    unique, counts = np.unique(array_target_labels, return_counts=True)
    total = 0
    entrop = 0
    for i in range(len(counts)):
        total = total + counts[i]
    for i in range(len(counts)):
        p = (counts[i] / total)
        entrop = entrop + (p * math.log2(p))
    e = -(entrop)
    return e


class DecisionNode:
    '''
    This class contain functions required by decision node
    '''
    def __init__(
            self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    '''
       This class contain all functions required by decision tree
    '''
    def __init__(self):
        self.root = None

    def fit(self, features, labels, max_depth):
        '''
        This function fits the model
        Input  : numpy of array of Train feature and train labels and depth of the decision tree
        Output :
        '''
        self.root = self._build_decision_tree(features, labels, max_depth, depth=0)

    def predict(self, X):
        '''
        This function predicts the label
        Input  : numpy of array of test features
        Output : predicted labels for given test features
        '''
        pred = []
        for x in X:
            pred1 = self._traverse_tree(x, self.root)
            pred.append(pred1)
        return np.array(pred)

    def _best_feature_split(self, features, labels):
        '''
        This function makes best feature split from all features
        Input : numpy of array of train features and train labels
        Output : the column number of best feature and the corresponding best threshold for that feature
                and leftchild indices and right child indices
        '''
        ls_split_val = []
        ls_gains = []
        for i in range(features.shape[1]):
            val, IG, _, _ = self._get_split_info(features[:, i], labels)
            ls_split_val.append(val)
            ls_gains.append(IG)
        best_IG = np.max(ls_gains)
        arrGains = np.asarray(ls_gains)
        # feature with highest IG
        ind = np.where(arrGains == best_IG)
        ind = ind[0][0]
        _, _, lsLeft_ind, lsRight_ind = self._get_split_info(features[:, ind], labels)
        best_feature = features[:, ind]
        best_val = ls_split_val[ind]
        return ind, best_val, lsLeft_ind, lsRight_ind

    # for each feature, we are getting thresholds, and identifying the splitting value
    def _get_split_info(self, feature, labels):
        '''
        This function calculates IG for single feature threshold value and identifies best threshold out of them
        Input : Numpy array of train features and train labels
        Output : Best threshold value of that feature,Best IG  of that feature,Left indices and right indices
        '''
        # sort the values of feature
        arrF_sorted = np.sort(feature)
        ls_c = []
        lsIG = []
        # calculate mid values
        for i in range(len(arrF_sorted) - 1):
            mid = (arrF_sorted[i] + arrF_sorted[i + 1]) / 2
            ls_c.append(mid)
        arr_c = np.array(ls_c)

        # eval IG for each val of threshold
        for i in range(len(arr_c)):
            c = arr_c[i]
            # check all values of feature with root and place in corresponding subtree
            lsLeft_features = []
            lsLeft_labels = []
            lsRight_features = []
            lsRight_labels = []
            lsLeft_ind = []
            lsRight_ind = []
            for j in range(len(feature)):
                if (feature[j] <= c):
                    lsLeft_features.append(feature[j])
                    lsLeft_labels.append(labels[j])
                else:
                    lsRight_features.append(feature[j])
                    lsRight_labels.append(labels[j])
            lsLabels_child = np.array([lsLeft_labels, lsRight_labels], dtype=object)
            lsIG.append(self._information_gain(labels, lsLabels_child))
        IG_max = np.max(lsIG)
        # index of location where IG is MAX
        ind_maxIG = np.where(lsIG == IG_max)
        dval = arr_c[ind_maxIG[0][0]]
        # left and right indexes for maximum IG threshold
        for j in range(len(feature)):
            if (feature[j] <= dval):
                lsLeft_ind.append(j)
            else:
                lsRight_ind.append(j)
        return (dval, IG_max, lsLeft_ind, lsRight_ind)

    # build decision tree
    def _build_decision_tree(self, features, labels, max_depth, depth=0):
        '''
        This function builds decision tree
        Input: Numpy array of train features and train labels,depth of decision tree
        Output : Decision tree
        '''
        n_labels = len(np.unique(labels))
        # check the stoppinf criteria
        if (depth >= max_depth or n_labels <= 1):
            leaf = self._most_common_label(labels)
            return DecisionNode(value=leaf)
            # return leaf
        col, best_val, lsLeft_ind, lsRight_ind = self._best_feature_split(features, labels)

        leftfeatures = features[lsLeft_ind]
        leftlabels = labels[lsLeft_ind]
        left_subtree = self._build_decision_tree(leftfeatures, leftlabels, max_depth, depth + 1)

        rightfeatures = features[lsRight_ind]
        rightlabels = labels[lsRight_ind]
        right_subtree = self._build_decision_tree(rightfeatures, rightlabels, max_depth, depth + 1)
        return DecisionNode(col, best_val, left_subtree, right_subtree)
        # return most_frequent_label()

    # calculating information gain
    def _information_gain(self, starting_labels, split_labels):
        '''
        This function calculates information gain
        Input : numpy array of actual labels and left and right child labels
        Output : Information gain
        '''
        info_gain = entropy1(starting_labels)
        for branched_subset in split_labels:
            info_gain -= len(branched_subset) * entropy1(branched_subset) / len(starting_labels)
        return info_gain

    def _traverse_tree(self, x, DecisionNode):
        '''
        This function traverses through the decision tree
        Input : test feature, decision node
        Output : Left or right or leaf nodes
        '''
        if DecisionNode.is_leaf_node():
            return DecisionNode.value

        if x[DecisionNode.feature] <= DecisionNode.threshold:
            return self._traverse_tree(x, DecisionNode.left)
        return self._traverse_tree(x, DecisionNode.right)

    def _most_common_label(self, labels):
        '''
        This function finds the most common label
        Input : labels
        Output : most common label
        '''
        vals, counts = np.unique(labels, return_counts=True)
        if (len(vals) == 0):
            return
        else:
            mode_value = np.argwhere(counts == np.max(counts))
            most_frequent_label = vals[mode_value][0][0]
        return most_frequent_label


def accuracy(y_true, y_pred):
    '''
    This function calculates accuracy
    Input : actual and expected labels
    Output : accuracy
    '''
    count = 0
    for i in range(len(y_true)):
        if (y_true[i] == y_pred[i]):
            count = count + 1
    accuracy = count / len(y_true) * 100
    return accuracy

####################################################### function required for bagging  #################################################
def randomsample(features):
    '''
    This function gets random samples from original train feature set
    '''

    ls_randomsample = random.choices(list(enumerate(features)),k=len(features))
    arr_randomsample = np.array(ls_randomsample,dtype=object)
    ind = [x for x in arr_randomsample[:,0]]
    sample =[x for x in arr_randomsample[:,1]]
    return ind,np.array(sample)

def most_common_label(labels):
    '''
      This function finds the most common label
      Input : labels
      Output : most common label
      '''
    vals, counts = np.unique(labels, return_counts=True)
    mode_value = np.argwhere(counts == np.max(counts))
    most_frequent_label = vals[mode_value][0][0]
    return most_frequent_label

def bagging(trainfeatures,trainlabels,max_depth,num_samples,testfeature):
    '''
    This function does bagging by using original decision tree
    '''
    random.seed(5)
    np.random.seed(5)
    predictions = []
    predict_labels = np.zeros((len(testfeature),num_samples),dtype=str)
    dt = DecisionTree()
    for i in range(num_samples):
        index,randsample_features = randomsample(trainfeatures)
        randsample_labels = trainlabels[index]
        dt.fit(randsample_features,randsample_labels,max_depth)
        predict_labels[:,i] = dt.predict(testfeature)
    for i in range(len(predict_labels)):
        p = most_common_label(predict_labels[i])
        predictions.append(p)
    return predictions

###########################################################################################Functions required for Adaboost #############################################################

#calculate weights for data items
def data_items_weights(weights,labels,pred_labels,alpha):
    #compute weight to data items
    wts = np.zeros([len(weights),1])
    for i in range(len(weights)):
        wts[i] = weights[i] * (np.exp(-alpha * labels[i] * pred_labels[i]))
    wts_sum = sum(wts)
    for i in range(len(wts)):
        wts[i] = wts[i]/wts_sum
    return wts

#compute classifier weight
def classifier_weight(error):
    #compute classifier weight
    alpha = (1/2) * (math.log((1-error)/(error +1e-10 )))
    #print('classifier wt is'+str(alpha))
    return alpha

#compute delta
def delta(labels,pred_labels):
    #comparing predicted labels with actual labels
    delta = np.zeros([len(pred_labels),1])
    for i in range(len(pred_labels)):
        if(labels[i] == pred_labels[i]):
            delta[i] = 1
        else:
            delta[i] = 0
    return delta.ravel()

#compute error of the classifier
def classifier_error(weights,delta):
    #compute error of the classifier
    denominator = 0
    numerator = 0
    for i in range(len(weights)):
        denominator = denominator + weights[i]
        numerator = numerator + (weights[i]*(1-delta[i]))
    #for i in range(len(delta)):
        #numerator = numerator + (weights[i]*(1-delta[i]))
    error = numerator/denominator
    #print('error is'+str(error))
    return np.float64(error)

def adaboost(trainfeatures,trainlabels,max_depth,num_times,test_features):
    predict_labels = np.zeros([len(test_features),num_times])

    #adding weights column to features
    weights = np.zeros([len(trainfeatures),1])
    for i in range(len(trainfeatures)):
        weights[i] = 1/len(trainfeatures)
    arrtrain_features = np.append(trainfeatures,weights,axis=1)

    X_train = trainfeatures
    dt = DecisionTree()
    for i in range(num_times):
        dt.fit(X_train, trainlabels,max_depth)
        y_train_pred = dt.predict(trainfeatures)
        arrdelta = delta(trainlabels,y_train_pred)
        #compute error of the classifier
        e = classifier_error(weights,arrdelta)
        error = np.round(e,5)

        if(error<0.5):
            #compute classifier weight
            alpha = classifier_weight(error)
            #compute weights for data items
            weights = data_items_weights(weights,trainlabels,y_train_pred,alpha)
            #modifying weights column in features
            arrtrain_features[:,3] = weights[:,0]

            #calculate cummulative sum upper
            cum_sum_up = np.cumsum(weights)
            cum_sum_up = np.reshape(cum_sum_up,(len(cum_sum_up),1))
            cum_sum_low = cum_sum_up - weights
            cum_sum_low = np.reshape(cum_sum_low,(len(cum_sum_low),1))
            arrtrain_features1 = np.append(arrtrain_features,cum_sum_low,axis=1)
            arrtrain_features2 = np.append(arrtrain_features1,cum_sum_up,axis=1)

            indices = []
            for k in range(len(arrtrain_features2)):
                a = np.random.random()
                for j in range(len(arrtrain_features2)):
                    if(arrtrain_features2[j,5] > a and a > arrtrain_features2[j,4]):
                        indices.append(j)

            X_train = trainfeatures[indices]

            # predict labels for test feature
            y_test_pred = dt.predict(test_features)
            y_test_pred_updated = alpha * y_test_pred
            predict_labels[:,i] = y_test_pred_updated

    predictions = []
    for i in range(predict_labels.shape[0]):
        p = 0
        for j in range(predict_labels.shape[1]):
            p = p + predict_labels[i,j]
        predictions.append(np.sign(p))
    #print(accuracy(testlabels, predictions))
    return predictions
