import numpy as np
import matplotlib.pyplot as plt

def prepare_data(filename):
    
    # extract data from the file
    points = np.genfromtxt(filename, delimiter=',')
    
    # create a matrix of only factors
    X = points[:, :-1]
    
    # add free coefficients as 1
    X = np.hstack(
        (np.ones((X.shape[0], 1)),X))
    
    # create a vector of outputs
    y = points[:, -1]
    
    return points, X, y
    

def linear_regression(X, y):
    '''
    calculate weights = (X.T @ X)^(-1) @ X.T @ y
    where '@' denotes a matrix product
    '''
    weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    
    return weights

def compute_error(weights, X, y):
    '''
        Computes Error = 1/N * ||X @ weights  - y||^2
    '''
    N = X.shape[0]
    
    return np.linalg.norm(np.dot(X, weights) - y)**2 / N

def plot_results(points, weights):
    '''
    plot the results if we have only 1 factor
    '''
    if points.shape[1] == 2:
        plt.scatter(points[:, 0], points[:, 1], color = 'r')
        x = np.linspace(min(points[:, 0]), max(points[:, 0]), 100)
        y = list(map(lambda a: weights[0] + weights[1] * a, x))
        plt.plot(x, y, color = 'b')
        plt.show()
        
def sklearn_realization(points):
    '''
    Let's see sklearn realization and calculate an error for it
    '''
    from sklearn import linear_model, metrics

    X = points[:, :-1]
    y = points[:, -1]
    
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    
    # get prediction
    y_predicted = regr.predict(X)
       
    # calculate error
    error = metrics.mean_squared_error(y, y_predicted)
    
    return error
       
def main():
    # define the name of the file
    filename = 'E:\DataRootUniversity\Module4\data_linear_regression.csv'
    
    # create the matrix of inputs and the vector of outputs
    points, X, y = prepare_data(filename)
    
    # calculate the weights = (w_0, w_1, ..., w_n)
    weights = linear_regression(X, y)
    
    # cumpute the error
    error = compute_error(weights, X, y)
    
    print('weights = {}\nerror = {}'.format(weights, error))
    
    plot_results(points, weights)
    
    # let's compare with sklearn results
    error_sklearn = sklearn_realization(points)
    print('Sklearn results:\nerror = {}'.format(error_sklearn))
    
main()