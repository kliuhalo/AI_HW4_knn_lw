
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Linear regression
def sklearn_func(X,y):
    model = LinearRegression(fit_intercept=True)
    x = X[:,np.newaxis] # (1000,1)

    plt.figure()
    # plt.plot(X,y)
    # plt.show()
    plt.scatter(X, y)
    model.fit(x, y)
    predict_y = model.predict(x)
    plt.scatter(X, predict_y, color='r')
    #plt.show()


# polynomial regression


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))
def sklearn_func2(X,y):

    x = X[:,np.newaxis] # (1000,1)

    deg = 1
    fig, ax = plt.subplots(4,4)
    plt.figure()
    for i in range(4):
        for j in range(4):
            ypred=PolynomialRegression(degree=deg).fit(x,y).predict(x)
            ax[i][j].scatter(X,y)
            ax[i][j].scatter(X, ypred,marker = ".")

            plt.title('degree '+str(deg) )
            deg += 1
    #plt.show()

# -------------
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


def least_square3(X, y):
    X = np.array(X)
    y = np.array(y)
    x = X[:,np.newaxis] # (1000,1)
    n = x.shape[1]
    r = np.linalg.matrix_rank(x)
    print(n,r)

    U, sigma, VT = np.linalg.svd(x, full_matrices=False)
    D_plus = np.diag(np.hstack([1/sigma[:r], np.zeros(n-r)]))
    V = VT.T
    X_plus = V.dot(D_plus).dot(U.T)
    w = X_plus.dot(y)
    error = np.linalg.norm(x.dot(w) - y, ord=2) ** 2

    np.linalg.lstsq(x, y)

    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, w*x, c='red')

    plt.show()


def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
  
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
  
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
  
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
  
    return (b_0, b_1)




def plot_regression_line(X, y, b):
    x = X[:,np.newaxis] # (1000,1)
    # plotting the actual points as scatter plot
    plt.scatter(X, y, color = "m",
               marker = "o")#, s = 30)
  
    # predicted response vector
    y_pred = b[0] + b[1]*x
  
    # plotting the regression line
    plt.plot(X, y_pred, color = "g")
  
    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')
  
    # function to show plot
    #plt.show()


if __name__ =="__main__":
    data = np.load('data/data1.npz')
    X = data['X'] 
    y = data['y']
    y = y.tolist()
    X = X.tolist()
    plt.figure()
    plt.scatter(X, y)
    plt.show()
    least_square3(X, y)