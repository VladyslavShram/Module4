import matplotlib.pyplot as plt
import numpy as np

def generate_data():
    points = np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
              (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
              (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))
    return points

def initialize_centroids(points, k):
    '''
        Selects k random points as initial
        points from dataset
    '''
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

def closest_centroid(points, centroids):
    '''
        Returns an array containing the index to the nearest centroid for each point
    '''
    dists = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(dists, axis=0)

def move_centroids(points, closest, centroids):
    '''
        Returns the new centroids assigned from the points closest to them
    '''
    return np.array([points[closest==i].mean(axis=0) for i in range(centroids.shape[0])])

def main():
    num_iterations = 8
    
    # Number of centroids
    k = 3 
        
    # Generate data
    points = generate_data()
    
    # Initialize centroids
    centroids = initialize_centroids(points, k)
    
    # Run iterative process
    for i in range(num_iterations):
        closest = closest_centroid(points, centroids)
        centroids = move_centroids(points, closest, centroids)
    
    
    return centroids, closest_centroid(points, centroids)

centroids, clusterized = centroids, closest_centroid(points, centroids)
print('centroids:\n', centroids)
print('clusterized:\n', clusterized)
