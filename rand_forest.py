"""
Random Forest Implementation with Bagging
@Author Jason St. George 2018
Acknowledgement: Jason Brownlee @ machinelearningmastery.com for original decision tree design inspiration

"""
from sys import argv
from random import seed
from random import randrange
from csv import reader
from math import ceil
from math import sqrt
from math import inf


def get_data(fn):
    """
    Import data from csv file
    :param fn: filename
    :return: list of lists with contents of csv
    """
    data = []
    with open(fn, 'r') as file:
        csv_reader = reader(file)

        for row in csv_reader:
            if not row:
                continue
            data.append(row)

    return data


def convert_to_float(data, column):
    """
    Convert string values in list of lists to type float
    :param data: list of lists (string)
    :param column: feature
    """
    for row in data:
        row[column] = float(row[column].strip())


def convert_to_int(data, column):
    """
    Convert labels to integer values (assuming they are float)
    :param data: list of lists
    :param column: index of labels
    :return: lookup table of Y labels
    """
    class_values = [row[column] for row in data]
    unique = set(class_values)
    lookup = {}

    for i, value in enumerate(unique):
        lookup[value] = i

    for row in data:
        row[column] = lookup[row[column]]

    return lookup


def cross_validation_split(data, n_folds):
    """
    Get CV split based on desired number of k-folds
    :param data: list of lists
    :param n_folds: int value of k-folds to split
    :return: list of split data
    """
    split = []
    copy = list(data)
    fold_size = int(len(data) / n_folds)
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(copy))
            fold.append(copy.pop(index))
        split.append(fold)
    return split


def test_split(index, value, data):
    """
    Get test split based on feature and value
    :param index: feature index in list (col)
    :param value: feature value to split on
    :param data: list of lists
    :return: test split for each branch
    """
    left, right = [], []

    for row in data:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)

    return left, right


def accuracy(actual, predicted):
    """
    Compare predicted values with actual labels in dataset
    :param actual: Y labels
    :param predicted: Output of model
    :return: Ratio of correctly to incorrectly predicted values
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / float(len(actual)) * 100.


def evaluate_algorithm(data, algorithm, n_folds, *args):
    """
    Main code that trains the model on k-folds and returns an accuracy metric for the current dataset
    :param data: current k-fold (list of lists)
    :param algorithm: Model to train (random forest)
    :param n_folds: Number of k-folds
    :param args: Remaining arguments: max_depth, min_size, sample_size, num_trees
    :return: Accuracy of trained model on cross-validation sets
    """
    folds = cross_validation_split(data, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = []

        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        acc = accuracy(actual, predicted)
        scores.append(acc)

    return scores


def gini_index(groups, class_values):
    """
    Computes the gini index of the proposed split.  Measures the entropy of the split and returns
     a real-valued number between 0 and 1, where 0 represents a perfect split and 1 represents an even (50/50) split.
    :param groups: dictionary of left and right groups
    :param class_values: set of class labels
    :return: gini index for the proposed split
    """
    gini = 0.
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue

            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))

    return gini


def get_split(data, n_features):
    """
    Greedy algorithm that chooses the best available split
    :param data: list of lists
    :param n_features: number of features to include in split
    :return: dictionary of best feature index, best feature value to split on, and all groups
    """
    class_values = list(set(row[-1] for row in data))
    b_index, b_value, b_score, b_groups = inf, inf, inf, None
    features = []
    while len(features) < n_features:
        index = randrange(len(data[0]) - 1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in data:
            groups = test_split(index, row[index], data)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups

    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def to_leaf(group):
    """
    Create a leaf node containing the current group
    :param group: current list of elements
    :return: majority vote prediction for this leaf
    """
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, n_features, depth):
    """
    Recursively split and build the tree until we reach terminating conditions
    :param node: Dictionary of current groups
    :param max_depth: How deep should the tree get
    :param min_size: Terminating condition for bins
    :param n_features: Number of features 
    :param depth: Current depth of the tree
    """
    left, right = node['groups']
    del (node['groups'])

    if not left or not right:
        node['left'] = node['right'] = to_leaf(left + right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_leaf(left), to_leaf(right)
        return

    if len(left) <= min_size:
        node['left'] = to_leaf(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)

    if len(right) <= min_size:
        node['right'] = to_leaf(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)


def build_tree(train, max_depth, min_size, n_features):
    """
    Create root node and call splitting function to build the tree recursively
    :param train: training data
    :param max_depth: Maximum depth for the tree
    :param min_size: Terminating condition for bin size to prevent overfitting
    :param n_features: Number of features
    :return: Trained model (decision tree)
    """
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


def predict(node, row):
    """
    Recursively traverse the tree to make a prediction given the current example
    :param node: Current node (level in the tree)
    :param row: Testing example
    :return: The predicted value of the bin/leaf node
    """
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def subsample(data, ratio):
    """
    Create a random subsample from the dataset with replacement
    :param data: list of lists (examples)
    :param ratio: percentage of data to subsample
    :return: Subsampled data
    """
    sample = []
    n_sample = round(len(data) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(data))
        sample.append(data[index])
    return sample


def bagging_predict(trees, row):
    """
    Make a prediction for each decision tree and aggregate the results
    :param trees: list of decision trees to predict with
    :param row: current example
    :return: predicted value
    """
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    """
    Main random forest algorithm that creates multiple trees and makes predictions with bagging
    :param train: Training data
    :param test: Testing data
    :param max_depth: Maximum depth the trees should reach
    :param min_size: Minimum bin (leaf) size
    :param sample_size: Size of subsamples desired
    :param n_trees: Number of trees in the forest
    :param n_features: Number of features in the dataset
    :return: predicted values from bagging
    """
    trees = []
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return predictions


def main():
    fn = 'data_banknote_auth.csv'
    n_trees = 10
    n_folds = 5
    max_depth = 8
    min_size = 1
    sample_size = 1.0
    seed(1)

    if len(argv) > 1:
        assert len(argv) == 6, "Usage: rand_forest.py file n_trees n_folds max_depth min_size"
        fn = argv[1]
        n_trees = int(argv[2])
        n_folds = int(argv[3])
        max_depth = int(argv[4])
        min_size = int(argv[5])

    data = get_data(fn)

    n_features = ceil(sqrt(len(data[0]) - 1))

    for i in range(0, len(data[0]) - 1):
        convert_to_float(data, i)

    lookup_table = convert_to_int(data, len(data[0]) - 1)

    scores = evaluate_algorithm(data, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    print('K-Fold scores: %s' % scores)
    print('Overall accuracy: %.3f%%' % (sum(scores) / float(len(scores))))


if __name__ == "__main__":
    main()