import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
import os, sys
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline 
import torch.nn as nn

import torch
from torch import Tensor

criterion = nn.MSELoss()

# ---------------------------
def least_square(X, y):
    x = X[:,np.newaxis] # (1000,1)
    y = y[:, np.newaxis]
    A = np.vstack([X, np.ones(len(X))]).T
    
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    return m,c

def least_square2(X, y):
    x = X[:,:,np.newaxis] # (1000,1)
    y = y[:, np.newaxis]
    
    A = np.hstack([X, np.ones(len(X))[:,np.newaxis]])
    
    m1, m2, c = np.linalg.lstsq(A, y, rcond=None)[0]
    #print("m1, m2, c", m1, m2, c)
    return m1, m2, c
# ---------------------------

class KNN_linear_regression_1d():

    def __init__(self, x, y, k):
        self.x = x
        self.y = y
        self.test_x = None
        self.k = k

    def predict(self, test_x):
        self.test_x = test_x
        dists = self.compute_distances(test_x)
        test_labels = self.predict_labels(dists, k = self.k)
        return test_labels

    def compute_distances(self, test_x):

        self.test_x = test_x
        num_test = test_x.shape[0]
        num_train = self.x.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sum((test_x[i]-self.x[j])**2)**0.5
        return dists

    def predict_labels(self, dists):
        num_test = dists.shape[0]
        k = self.k
        y_pred = np.zeros(num_test)
        loss = 0
        for i in range(num_test):
            sorted_index = sorted(range(len(self.x)),key = lambda x:dists[i][x])
            knn_X = [self.x[n] for n in sorted_index[0:k]]
            knn_Y = [self.y[n] for n in sorted_index[0:k]]

            m, c = least_square(np.array(knn_X), np.array(knn_Y))
            y_pred[i] = m*self.test_x[i] + c
            # calculate loss
            loss += (y_pred[i] - knn_Y[0])**2
        loss = (loss**0.5)/num_test
        print("k = ",self.k,", loss = ", loss)
            
        return y_pred

def dataset2(datapath,k=20):
    data = np.load(datapath)
    X = data['X'] 
    y = data['y']

    test_x = []
    num_test = 1000
    for i in range(1000):
        test_x.append((random.uniform(-3,3), random.uniform(-3,3)))#[:,None]
    
    test_x = np.array(test_x)
    train_data = list(zip(X[:,0], X[:,1]))
    newX1 = np.array(X[:,0])
    newX2 = np.array(X[:,1])
    num_test = test_x.shape[0]#newX1.shape[0] * newX2.shape[0]
    num_train = len(train_data)
    # compute distance
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
            dists[i][j]= np.sum((test_x[i]-train_data[j][0])**2+(test_x[i]-train_data[j][1])**2)**0.5
    # predict test_x
    
    y_pred = np.zeros(num_test)
    
    loss = 0
    for i in range(num_test):
        sorted_index = sorted(range(len(X)),key = lambda x:dists[i][x])
        knn_X1 = [newX1[n] for n in sorted_index[0:k]]
        knn_X2 = [newX2[n] for n in sorted_index[0:k]]
        knn_Y = [y[n] for n in sorted_index[0:k]]
        distance = [dists[i][n] for n in sorted_index[0:k]] 
        
        train_data = np.array(list(zip(knn_X1, knn_X2)))
        knn_Y = np.array(knn_Y)

        m1, m2, c = least_square2(train_data, knn_Y)
        y_pred[i] = m1 * test_x[i][0] + m2 * test_x[i][1] + c
        loss += (y_pred[i]-knn_Y[0])**2
   
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(newX1, newX2, y)
    #ax.plot_trisurf(knn_X1, knn_X2, knn_Y,label='h=0')
    ax.plot_trisurf(test_x[:,0].tolist(), test_x[:,1].tolist() , y_pred.tolist(),label='h=0')
    plt.show()

    # wrong
    # loss = criterion(torch.Tensor(np.asarray(y_pred)), torch.Tensor(y))
    print("loss on training data = ", loss**0.5/len(y_pred))
    return y_pred



def dataset1(datapath):
    data = np.load(datapath)
    X = data['X'] 
    y = data['y']
    y = y.tolist()
    X = X.tolist()

    plt.figure()
    plt.scatter(X, y)
    #plt.show()

    sorted_index = sorted(range(len(X)), key=lambda x:X[x])
    newY = [y[i]for i in sorted_index]
    newX = [X[i] for i in sorted_index]
    plt.figure()
    plt.scatter(newX, newY)
    #plt.show()


    newX = np.asarray(newX)
    newY = np.asarray(newY)

    knnLR = KNN_linear_regression_1d(newX, newY, 1000)

    x_test = np.linspace(newX[0],newX[-1],len(newX))[:,None]
    dists = knnLR.compute_distances(x_test)


    
    y_pred = knnLR.predict_labels(dists)
    #loss = criterion(torch.Tensor(y_pred), torch.Tensor(newY))
    #print("loss on training data = ", loss/len(y_pred))

    
    #print(np.shape(x_test), np.shape(np.asarray(y_pred)))
    plt.figure()
    plt.scatter(newX, newY)
    plt.plot(x_test,np.asarray(y_pred), c='r')
    #plt.show()


    return y_pred



if __name__ =='__main__':
    # data_path = 'data/data2.npz'
    # k = 1000
    # y_pred = dataset2(data_path, k)

    #print("prediction",y_pred)
    data_path = 'data/data1.npz'
    dataset1(data_path)