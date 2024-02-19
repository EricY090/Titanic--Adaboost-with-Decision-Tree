### I pledge my honor that I have abided by the Stevens Honor System
### Hw3 - Eric  Yang

import numpy as np
import math
from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


### Return the prediciton of testing data using decision tree classfiaction
### Use a decision tree with only depth of 1 as the weak classifier
def decision_tree(x_tr, y_tr,  weight):
    model = DecisionTreeClassifier(max_depth = 1)
    model.fit(x_tr, y_tr, sample_weight = weight)
    y_predict = model.predict(x_tr)
    return model, np.array([y_predict])


### Return alpha in the Adaboost
def alpha(w, y_tr, y_predict):
    w = np.array([w])
    err = w @ (np.not_equal(y_tr, y_predict)).astype(int)
    err = err /np.sum(w)
    alpha = 0.5*math.log((1-err)/err)
    return alpha



### Update the weight
def update(y_tr, predict, w, N, a):
    z = sum(w)
    w = w/z
    w = np.array([w])
    k = np.exp((-a)* (np.equal(y_tr, predict)).astype(int))
    w = w*(k.T)
    return w.ravel()                        ### Flatten back into 1D array for the usage of decision tree function


### Update the prediction including the new tree and alpha
def pred(tree, alpha, x, predict):
    t = (np.array([tree.predict(x)])).T
    update = alpha*t
    predict = predict+update
    return predict


### Calulate Accuracy
def accuracy(predict, y_tst):
    predict = np.sign(predict)          ### Transform into -1 or 1, and same dimension with the testing data
    correct = np.sum((np.equal(predict, y_tst)).astype(int))
    return correct/y_tst.shape[0]



### Adaboost using decision tree classfiers as weak classifiers
def Adaboost(x_tr, y_tr, rounds, x_tst, y_tst):
    N = x_tr.shape[0]           ### Number of training examples
    tree_collection = []
    alpha_collection = []
    u = np.array([1/N] * N)    ### Initial weight, 1D array
    train_acc = []              ### Obtain the accuracy as the error metrix, for the training and testing data in 500 rounds
    test_acc = []
    train_predict = (np.array([(y_tr.shape[0])*[0]])).reshape(y_tr.shape[0], 1)
    test_predict = (np.array([(y_tst.shape[0])*[0]])).reshape(y_tst.shape[0], 1)
    
    for i in range(rounds):
        t, predict = decision_tree(x_tr, y_tr, u)       ### Get the weak classifier, and prediction from it
        predict = predict.reshape(N, 1)
        tree_collection.append(t)
        a = alpha(u, y_tr, predict)
        alpha_collection.append(a)

        ### Obtain accuracy after each round for prediction of traing and testing data
        train_predict = pred(tree_collection[-1], alpha_collection[-1], x_tr, train_predict)
        test_predict = pred(tree_collection[-1], alpha_collection[-1], x_tst, test_predict)
        train_acc.append(accuracy(train_predict, y_tr))
        test_acc.append(accuracy(test_predict, y_tst))
        
        u = update(y_tr, predict, u, N, a)              ### Update the weight for next round
        
    return train_acc, test_acc
        
    


if __name__ == "__main__":
    
    ### Import data
    dataframe = read_csv("train.csv")
    
    ### Data Processing
    ### Get the features we want and the corresponding surviving result
    data_x = dataframe[["Pclass", "Sex", "SibSp", "Parch", "Embarked"]].to_numpy()
    data_y = dataframe[["Survived"]].to_numpy()

    ### Transform attribute "Embarked" into (0, 1, 2), and "Sex" into (0, 1)
    for i in range(data_x.shape[0]):
        if data_x[i][-1] == 'S': data_x[i][-1]= 0
        elif data_x[i][-1] == 'C': data_x[i][-1]= 1
        elif data_x[i][-1]== 'Q': data_x[i][-1] = 2
        else:                                   ### For removing the instance with empty Embarked data
            data_x[i][-1] = 100
            data_y[i][0] = 100
        if data_x[i][1] == 'male': data_x[i][1]= 0
        else: data_x[i][1]= 1
    ### Remove the instance with empty Embarked data
    data_x = np.array([x for x in data_x if x[-1] != 100])
    data_y = np.array([x for x in data_y if x[0] != 100])
    
    data_y[data_y == 0] = -1                    ### Transform the non-survived data from 0 into -1
    
    ### Make data as float
    data_x = data_x.astype(float)
    data_y = data_y.astype(float)
    

    ### Divide into 80% training and 20% testing
    data_x, data_y = shuffle(data_x, data_y)
    x_train = data_x[: int((data_x.shape[0])*0.8)]
    x_test = data_x[int((data_x.shape[0])*0.8):]
    y_train = data_y[: int((data_y.shape[0])*0.8)]
    y_test = data_y[int((data_y.shape[0])*0.8):]


    ### Model Training-----------------------------------------------------------------------------------------###
    rounds = 500
    ### Return the accuracy calcualted for the predcition of training and testing data after each round
    train_acc, test_acc = Adaboost(x_train, y_train, rounds, x_test, y_test)


    ### Accuracy for the best round in testing data
    print(f"Accuracy: {max(test_acc)}")

    
    ### Plot the accuracy for training and testing data for 500 rounds
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.show()
    

    
    
   
    
