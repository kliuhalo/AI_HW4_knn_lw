import numpy as np
import random
import matplotlib.pyplot as plt
import os, sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import linear_regress_func 
import lw

from mpl_toolkits import mplot3d 



def lw_try_dataset2():

    data = np.load('data/data2.npz')
    X = data['X'] 
    y = data['y']
    newX1 = X[:,0]
    newX2 = X[:,1]
    sorted_index = sorted(range(len(X)), key=lambda x:(X[x,0],X[x,1]))
    newY = [y[i]for i in sorted_index]
    newX1 = [newX1[i] for i in sorted_index]
    newX2 = [newX2[i] for i in sorted_index]
    
    test_x = []
    num_test = 1000
    for i in range(1000):
        test_x.append((random.uniform(-3,3), random.uniform(-3,3)))#[:,None]
    test_x = np.array(test_x)
    train_data = list(zip(newX1, newX2))
    newX1 = np.array(newX1)
    newX2 = np.array(newX2)
    num_test = test_x.shape[0]#newX1.shape[0] * newX2.shape[0]
    num_train = len(train_data)
    # compute distance
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
            dists[i][j]= np.sum((test_x[i]-train_data[j][0])**2+(test_x[i]-train_data[j][1])**2)**0.5
    # predict test_x
    k = 100
    m = num_train
    # print("X.shape",X.shape,m)
    # print(X[:,:].shape)
    # print(np.ones(m)[:,np.newaxis].shape)
    X = np.hstack([X, np.ones(m)[:,np.newaxis]])
    # print("X.shape",X.shape)
    y_pred = np.zeros(num_test)
    newY = np.array(newY)
    matrixY = np.array(newY[:,np.newaxis])
    predicts = []
    for i in range(test_x.shape[0]):
        
        # con = np.array([1]*test_x[i].shape[0])[:,np.newaxis]
        #print("test_x.shape",test_x.shape, con.shape)
        point = np.concatenate((test_x[i],np.array([1])),axis=0)
        
        tau = 2
        Wm = lw.weight_matrix2(point, X, tau,m)
        
        theta = np.linalg.pinv(X.T*(Wm * X))*(X.T*(Wm * matrixY)) 
        # print("shape",Wm.shape,X.shape,theta.shape)
        
        pred = np.dot(point, theta)
        
        predicts.append(pred)
    predicts = np.squeeze(np.asarray(predicts))
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(newX1, newX2, newY)
    ax.plot_trisurf(test_x[:,0].tolist(), test_x[:,1].tolist() , predicts.tolist(),label='h=0')
    plt.show()

def other_2(X1, X2, y):
    x1 = X1[:,np.newaxis] # (1000,1)
    x1_2 = [X1[i]**2 for i in range(len(X1))]
    x2 = X1[:,np.newaxis]
    x2_2 = [X2[i]**2 for i in range(len(X2))]
    y = y[:, np.newaxis]
    x1 = np.array(x1)
    x2 = np.array(x2)
    x1_2 = np.array(x1_2)[:,np.newaxis]
    x2_2 = np.array(x2_2)[:,np.newaxis]
    print(x1.shape, x2.shape, x1_2.shape, x1_2.shape)
    a = np.concatenate((x1, x1_2), axis = 1)
    b = np.concatenate((x2, x2_2),axis = 1)
    X = np.concatenate((a,b),axis=1)
    A = np.concatenate((X,np.ones(len(X))[:,np.newaxis]),axis=1)
    print(X.shape, A.shape)
    #A = np.vstack([X, np.ones(len(X))]).T
    m1, m2, m3, m4, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print(m1,m2,m3,m4,c)
    return m1, m2, m3, m4, c




def other_method(X, y):
    x = X[:,np.newaxis] # (1000,1)
    y = y[:, np.newaxis]
    b = [X[i]**2 for i in range(len(X))]
    
    b= np.array(b)
    b = b[:,np.newaxis]
    x = np.array(x)
    print(x.shape, b.shape)
    X2 = np.concatenate((b, x), axis = 1).T
    A = np.vstack([X2, np.ones(len(X))]).T

    
    m1, m2, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print(m1, m2,c)
    return m1, m2 ,c

def other_method_dataset1():
    data = np.load('data/data1.npz')
    X = data['X'] 
    y = data['y']
    y = y.tolist()
    X = X.tolist()
    # plt.figure()
    # plt.scatter(X, y)
    # plt.show()

    sorted_index = sorted(range(len(X)), key=lambda x:X[x])
    newY = [y[i]for i in sorted_index]
    newX = [X[i] for i in sorted_index]

    newX = np.asarray(newX)
    newY = np.asarray(newY)

    x_test = np.linspace(newX[0],newX[-1],len(newX))[:,None]
    m1 , m2, c = other_method(newX, newY)
    
    y_pred = [(m1*(newX[i]**2)+m2*newX[i]+c) for i in range(len(x_test))]
    plt.figure()

    plt.scatter(newX, newY)
    plt.plot(newX, y_pred, c='r')
    plt.show()



if __name__=="__main__":
    # datapath = 'data/data2.npz'
    # data = np.load(datapath)
    # X = data['X'] 
    # y = data['y']
    # newX1 = X[:,0]
    # newX2 = X[:,1]

    # sorted_index = sorted(range(len(X)), key=lambda x:(X[x,0],X[x,1]))
    # newY = np.array([y[i]for i in sorted_index])
    # newX1 = np.array([newX1[i] for i in sorted_index])
    # newX2 = np.array([newX2[i] for i in sorted_index])
    # test_x = []
    # num_test = 1000
    # for i in range(1000):
    #     test_x.append((random.uniform(-3,3), random.uniform(-3,3)))#[:,None]
    # test_x = np.array(test_x)

    # m1, m2, m3, m4, c= other_2(newX1, newX2, newY)
    # y_pred = [(m1*(newX1[i]**2)+m2*newX1[i])+m3*(newX2[i]**2)+m3*newX2[i]+c for i in range(len(test_x))]
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_trisurf(newX1, newX2, newY)
    # y_pred = np.array(y_pred)
    # y_pred = y_pred.squeeze().tolist()
    # print(y_pred)
    # print(test_x[:,0].shape, np.array(y_pred).shape)
    # ax.plot_trisurf(test_x[:,0].tolist(), test_x[:,1].tolist() , y_pred,label='h=0')
    # plt.show()
    other_method_dataset1()