import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
import scipy.ndimage as ndimag

def ldaLearn(X,y):
    
    left_part, right_part = np.hsplit(X, 2)
    uniq_y = np.unique(y)
    x_mean = ndimag.mean(left_part, labels=y, index=uniq_y)
    y_mean = ndimag.mean(right_part, labels=y, index=uniq_y)
    #Stack arrays in sequence vertically (row wise).
    means = np.vstack((x_mean, y_mean))
    
    covmat = np.cov(X.T)
    # Return covmat and the means mat
    return means,covmat

def qdaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD

    
    
    #Split an array into multiple sub-arrays horizontally (column-wise).
    left_part, right_part = np.hsplit(X, 2)
    uniq_y = np.unique(y)
    #Calculate the mean of the values of an array at labels.
    #labels: All elements sharing the same label form one region over which the mean of the elements is computed.
    #index: Labels of the objects over which the mean is to be computed
    x_mean = ndimag.mean(left_part, labels=y, index=uniq_y)
    y_mean = ndimag.mean(right_part, labels=y, index=uniq_y)
    #Stack arrays in sequence vertically (row wise).
    means = np.vstack((x_mean, y_mean))


    # stack the labels to X
    mat = np.hstack((X, y))

    # uniq_y: array([ 1.,  2.,  3.,  4.,  5.])
    C1 = mat[mat[:, 2] == uniq_y[0], :]
    C2 = mat[mat[:, 2] == uniq_y[1], :]
    C3 = mat[mat[:, 2] == uniq_y[2], :]
    C4 = mat[mat[:, 2] == uniq_y[3], :]
    C5 = mat[mat[:, 2] == uniq_y[4], :]

    # last column is the label, remove it
    C1 = C1[:, :2]
    C2 = C2[:, :2]
    C3 = C3[:, :2]
    C4 = C4[:, :2]
    C5 = C5[:, :2]

    covmats = []
    covmats.append(np.cov(C1.T))
    covmats.append(np.cov(C2.T))
    covmats.append(np.cov(C3.T))
    covmats.append(np.cov(C4.T))
    covmats.append(np.cov(C5.T))
    return means, covmats


#%%

def ldaTest(means, covmat, Xtest, ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    
    
    # det: Compute the determinant of an array.    
    my_det = 1/np.sqrt(2 * np.pi) * (np.linalg.det(covmat)**2)
    
    # all ones : (100, 5)
    temp_matrix = np.ones((Xtest.shape[0],means.shape[1]))

    for i in range(0, Xtest.shape[0]):
        for j in range(0, means.shape[1]):

            m1 = (np.transpose(Xtest[i, :] - means[:, j].T))
            m1 = np.asmatrix(m1)
            m2 = np.linalg.inv(covmat)
            temp = np.dot(m1, m2)
            temp = -.5 * np.dot(temp, m1.T)
            temp = np.exp(temp[0,0]) * my_det
            temp_matrix[i, j] = temp

    # Returns the indices of the maximum values along an axis
    ypred = np.argmax(temp_matrix, axis=1) + 1
    ypred = np.asmatrix(ypred).T

    count = 0
    for x in range(0,Xtest.shape[0]):
        if ypred[x, 0] == ytest[x, 0]:
            count+=1
    total = Xtest.shape[0]
    acc = count/total
    return acc, ypred



    #%%
def qdaTest(means, covmats, Xtest, ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    # temp var with all ones
    temp_matrix = np.ones((Xtest.shape[0],means.shape[1]))

    for i in range(0, Xtest.shape[0]):
        for j in range(0, means.shape[1]):

            m1 = (np.transpose(Xtest[i, :] - means[:, j].T))
            m1 = np.asmatrix(m1)
            
            # inv(a): Compute the (multiplicative) inverse of a matrix.
            cov = covmats[j]
           

            temp = np.dot(m1, np.linalg.inv(cov))
            temp = -.5 * np.dot(temp, m1.T)
            
            # det: Compute the determinant of an array. 
            my_det = 1/ (np.power((2 * np.pi),len(means)/2) * np.sqrt(np.linalg.det(cov)))
            temp = np.exp(temp) * my_det
            temp_matrix[i, j] = temp

    ypred = np.argmax(temp_matrix, axis=1) + 1
    ypred = np.asmatrix(ypred).T

    count = 0
    for x in range(0,Xtest.shape[0]):
        if ypred[x, 0] == ytest[x, 0]:
            count += 1
    total = Xtest.shape[0]
    acc = count/total
    return acc, ypred

def learnOLERegression(X,y):
	
    # IMPLEMENT THIS METHOD  
    prod1 = np.dot(X.T,X)
    prod2 = np.dot(X.T,y)
    w = np.dot(np.linalg.inv(prod1), prod2) 
    return w

def learnRidgeRegression(X,y,lambd):
    zeroM = X.shape[0]
    oneM =X.shape[1]  
    calXtrans = X.T 
    calX = np.dot(calXtrans, X) 
    calY = np.dot(calXtrans, y) 
    identityM = np.identity(calX.shape[0]) 
    cal = lambd * identityM 
    calFinal =  calX + cal 
    calInver = np.linalg.inv(calFinal)
    w = np.dot(calInver, calY)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    prod1 = Xtest.dot(w)
    prod2 = ytest-(prod1)
    prod3 = prod2.T.dot(prod2) 
    
    return prod3 / ytest.size

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

   # IMPLEMENT THIS METHOD                                                                                                       

    
    # ERROR
    w_T = np.asmatrix(w).T # transpose
    # number of samples
    # ERROR

    mul = lambd * np.dot(w_T.T, w_T)
    diff = y - np.dot(X, w_T)
    diff = np.square(diff)
    num2 = np.sum(diff) + mul
    error = .5 * num2
    # ERROR GRAD

    mull = lambd *  w_T
    mul = np.dot(X.T,np.asmatrix(X))
    Y = np.dot(X.T,np.asmatrix(y))
    diff = np.dot(mul,w_T) - Y
   
    error_grad = diff + mull
    
    #Remove single-dimensional entries from the shape of an array.
    error_grad = np.squeeze(np.asarray(error_grad))
    
    return error, error_grad

    # return error, error_grad
def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    end = p + 1  # integer
    count = 1
    temp = x.shape[0]  # array
    Xd = np.ones((temp, 1))
    while (count < end):
        Xd = np.column_stack((Xd, np.power(x, count)))
        count = count + 1

    return Xd


#%%
# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'), encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
