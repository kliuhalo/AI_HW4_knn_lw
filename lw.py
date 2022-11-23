import numpy as np
import os, sys
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
import random
from tqdm import tqdm
import torch.nn as nn
import torch

criterion = nn.MSELoss()

def weight_matrix(point, X, tau, m):
    w = np.mat(np.eye(m))
    for i in range(m): 
        xi = X[i] 
        d = (-2 * tau * tau) 
        w[i, i] = np.exp(np.dot((xi-point), (xi-point).T)/d) 
        
   
    return w

def weight_matrix2(point, X, tau, m):
    w = np.mat(np.eye(m)) 
    for i in range(m): 
        w[i,i] = max(0, 1-pow(2*np.linalg.norm(X[i]-point)/tau,2))
    
    return w

def predict1d(newX, newY,tau):
    m = newX.shape[0]
    X = np.vstack([newX, np.ones(m)]).T
    newY = newY[:, np.newaxis]

    # predict
    predicts = []
    x_test = np.linspace(newX[0],newX[-1],len(newX))[:,None]
    x_test = x_test.tolist()
   
    for i in range(len(x_test)):
        point = np.array([float(x_test[i][0]), 1])
        
        Wm = weight_matrix(point, X, tau, m)

        theta = np.linalg.pinv(X.T*(Wm * X))*(X.T*(Wm * newY)) 
        pred = np.dot(point, theta)
        predicts.append(pred)
    predicts = np.squeeze(np.asarray(predicts))
    plt.figure()
    plt.scatter(newX, newY)

    plt.plot(x_test, predicts, c = 'r')
    plt.show()
    return predicts

def dataset1(data_path, tau=0.5):
    data = np.load(data_path)
    X = data['X'].tolist()
    y = data['y'].tolist()
    

    sorted_index = sorted(range(len(X)), key=lambda x:X[x])
    newY = np.asarray([y[i]for i in sorted_index])
    newX = np.asarray([X[i] for i in sorted_index])

    predicts = predict1d(newX, newY, tau)

    return predicts


def dataset2(data_path, tau=1):

    data = np.load(data_path)
    X = data['X'] 
    y = data['y']
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X[:,0], X[:,1], y)
    plt.show()
    
    test_x = []
    num_test = 1000
    for i in range(1000):
        test_x.append((random.uniform(-3,3), random.uniform(-3,3)))#[:,None]
    test_x = np.array(test_x)
    train_data = list(zip(X[:,0], X[:,1]))
    num_train = len(train_data)
    # compute distance
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
            dists[i][j]= np.sum((test_x[i]-train_data[j][0])**2+(test_x[i]-train_data[j][1])**2)**0.5
    # predict test_x
    m = num_train
    X = np.hstack([X, np.ones(m)[:,np.newaxis]])
    
    matrixY = np.array(y[:,np.newaxis])
    predicts = []
    for i in tqdm(range(test_x.shape[0])):
        
        point = np.concatenate((test_x[i],np.array([1])),axis=0)
        
        Wm = weight_matrix2(point, X, tau,m)
        Wm = weight_matrix(point, X, tau, m )
        
        theta = np.linalg.pinv(X.T*(Wm * X))*(X.T*(Wm * matrixY)) 
       
        pred = np.dot(point, theta)

        predicts.append(pred)
    
    predicts = np.squeeze(np.asarray(predicts))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X[:,0], X[:,1], y)
    ax.plot_trisurf(test_x[:,0].tolist(), test_x[:,1].tolist() , predicts.tolist(),label='h=0')
    plt.show()

if __name__=='__main__':
    # data_path = 'data/data1.npz'
    # tau = 1
    #predicts = dataset1(data_path, tau)
        
    data_path = 'data/data2.npz'
    tau = 0.08
    predicts = dataset2(data_path, tau)
