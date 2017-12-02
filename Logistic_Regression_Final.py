import numpy as np
import matplotlib.pyplot as plt

eps = 5e-2
num_observations = 5000

def generate_data(num_observations):
    np.random.seed(12)
    
    x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
    
    simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
    simulated_labels = np.hstack((np.zeros(num_observations),
                                  np.ones(num_observations)))
    return simulated_separableish_features, simulated_labels

def visualize_data(simulated_separableish_features, simulated_labels):
    plt.figure(figsize=(12,8))
    plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
                c = simulated_labels, alpha = .3)
    plt.show()

def sigmoid(x):
    '''1 / (1 + e^(-x))'''
    return 1 / (1 + np.exp(-x))

def log_likelihood(features, target, weights):
    '''
        U = sum(target * weights_tr * features - log(1 + exp(weights_tr * features)))
    '''
    scores = np.dot(features, weights)
    return np.sum(target * scores - np.log(1 + np.exp(scores)))

def grad(features, target, predictions):
    '''
        grad(U) = features_tr * (target - predictions)
    '''
    diff = target - predictions
    return np.dot(features.T, diff)

def hesse(features, predictions):
    '''
        hesse(U) = - features_tr * W * features,
        where W = diag(predictions)
    '''
    W = np.diag(predictions)   
    return np.dot(np.dot(features.T, W), features)

def stop_condition(features, target, predictions):
    return np.linalg.norm(grad(features, target, predictions)) <= eps

def logistic_regression(features, target):
    # add free coefficients as 1
    features = np.hstack(
        (np.ones((features.shape[0], 1)),features))
    
    # initialize weights, scores and predictions
    weights = np.zeros(features.shape[1])
    scores = np.dot(features, weights)
    predictions = sigmoid(scores)
    
    # iterative process
    # first 500 iterations - Newton-Raphson
    # next iterations - gradient descent
    step = 0
    
    while not stop_condition(features, target, predictions):

        # Update weights
        gradient = grad(features, target, predictions)

        if step < 500:
            hess = hesse(features, predictions)
            weights += np.dot(np.linalg.inv(hess), gradient)
            if step % 100 == 0:
                print('log_likelihood = ',log_likelihood(features, target, weights))
                print('||gradient|| = ', np.linalg.norm(gradient), '\n')
        else:
            weights += 5e-5 * gradient
            if step % 10000 == 0:
                print('log_likelihood = ',log_likelihood(features, target, weights))
                print('||gradient|| = ', np.linalg.norm(gradient), '\n')
                
        # Update scores and predictions
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)
        
        # Update step
        step += 1
        
    return weights

def show_correctness(simulated_separableish_features, simulated_labels, predicts):
    plt.figure(figsize = (12, 8))
    plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
                c = predicts == simulated_labels, alpha = .7, s = 50)
    plt.show()
    

def main():
    # Generate data
    simulated_separableish_features, simulated_labels = generate_data(num_observations)
    
    #Visualize generated data
    visualize_data(simulated_separableish_features, simulated_labels)
    
    # Calculate weights
    weights = logistic_regression(simulated_separableish_features, simulated_labels)
    
    # Add free coefficients as 1
    data_with_ones = np.hstack((np.ones((simulated_separableish_features.shape[0], 1)),
                                     simulated_separableish_features))
    # Add z-scores
    final_scores = np.dot(data_with_ones, weights)
    
    # Calculate predictions
    predicts = np.round(sigmoid(final_scores))
    
    # Show correctness of classifying
    show_correctness(simulated_separableish_features, simulated_labels, predicts)
    
    return weights
    
main()


