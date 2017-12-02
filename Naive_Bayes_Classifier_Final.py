import numpy as np

def prepare_data(filename):
    # extract data from the file
    return np.genfromtxt(filename, delimiter=',')

def split_dataset(dataset, split_ratio):
    # Training set size
    train_size = int(dataset.shape[0] * split_ratio)
    
    # List of randomly chosen indicies
    indices = np.random.permutation(dataset.shape[0])
    
    # Split indicies for training and test set by trainSize
    training_idx, test_idx = indices[:train_size], indices[train_size:]
    
    # Create training and test sets by indicies
    training, test = dataset[training_idx,:], dataset[test_idx,:]
    
    return training, test

def separate_by_class(dataset):
    # Here we limit our classes to 0 and 1
    # You need to generalize this for arbitrary number of classes
    # Me: Done!
    dictionary = {}
    values = set(dataset[:, -1])
    
    for val in values:
        dictionary[val] = dataset[dataset[:, -1]==val]
    
    return dictionary

def summarize(dataset):
    # Calculate means and standart deviations with one degree of freedom for each attribute
    # We do it by column which is axis 1
    # Also we remove last elements (guess why?) - Because the last elements are labels
    means = dataset.mean(axis=0)[:-1]
    stds = dataset.std(axis=0, ddof=1)[:-1]
    
    return means, stds

def summarize_by_class(dataset):
    # Divide dataset by class and summarize it
    separated = separate_by_class(dataset)
    
    summaries = {}
    
    for class_value, instances in separated.items():
        summaries[class_value] = summarize(instances)
    
    return summaries

def calculate_probability(x, mean, stdev):
    # Calculate probability by x, mean and std
    # 1/(sqrt(2pi)*std)*exp(-(x-mean)^2/(2std^2))
    return np.prod(np.exp(-(x - mean)**2 / (2 * stdev**2)) / (np.sqrt(2 * np.pi) * stdev))

def calculate_class_probabilities(summaries, input_vector):
    # Calculate probabilities for input vector from test set
    probabilities = {}
    
    for class_value, class_summaries in summaries.items():
        
        means = class_summaries[0]
        stds  = class_summaries[1]
        
        # Calculate corresonding probabilities and multiply them
        probabilities[class_value] = calculate_probability(input_vector[:-1], means, stds)
        
    return probabilities

def predict(summaries, input_vector):
    # Calculate probabilities
    probabilities = calculate_class_probabilities(summaries, input_vector)
    
    # Init values of probability and label
    best_label, best_prob = None, -1
    
    # Check probability of which class is better
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    
    return best_label

def get_predictions(summaries, test_set):   # optimize with numpy
    # For each probability find optimal labels
    # optimized with map
    predictions = list(map(lambda x: predict(summaries, x), test_set))

    return predictions

def get_accuracy(test_set, predictions):
    # Check accuracy
    correct = 0
    
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0

def main():
    # Set initial data
    filename = 'E:\DataRootUniversity\Module4\data_naive_bayes_classifier.csv'
    
    # Set split ratio
    split_ratio = 0.67
    
    # Load dataset and return numpy array
    dataset = prepare_data(filename)
    
    # Split dataset
    training_set, test_set = split_dataset(dataset, split_ratio)
    
    # Log row amounts
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(training_set), len(test_set)))
    
    # Prepare model
    summaries = summarize_by_class(training_set)
    
    # Test model
    predictions = get_predictions(summaries, test_set)
    
    accuracy = get_accuracy(test_set, predictions)
    
    print('Accuracy: {0}%'.format(accuracy))
    
main()