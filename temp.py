import numpy as np
import matplotlib.pyplot as plt
import os, sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import linear_regress_func

data = np.load('data/data2.npz')
data = np.load('data/data2.npz')
X = data['X'] 
y = data['y']

num_test=1000
test_x = np.array([ [i,i+1]for i in range(num_test)])
print("test_x", test_x.shape)


y_pred = np.zeros(num_test)
for i in range(num_test):
    train_data = np.array([[2,2],[3,3],[4,4]])
    knn_Y = np.array([6,8,10])
    m1, m2, c =linear_regress_func.least_square2(train_data, knn_Y)
    y_pred[i] = m1 * test_x[i][0] + m2 * test_x[i][1] + c

print(y_pred)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(test_x[:,0].tolist(),test_x[:,1].tolist() ,y_pred)
plt.show()


