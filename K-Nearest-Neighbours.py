import pandas as pd
import numpy as np


def data_setup():
    '''
    Reads the two files creating two lists of lists.
    Each sublist in the big list represents a different datapoint.
    26 sublists for the main_data, 188 sublists for the train_data
    '''
    train = pd.read_csv('train_data(1).txt', header=None)
    main = pd.read_csv('main_data(1).txt', header=None)
    train_data = np.array(train).tolist()
    main_data = np.array(main).tolist()
    return train_data, main_data


def dist_vect(item, train_data):
    '''
    Arguments: item = a datapoint of the main data
               train_data =  a list of lists as generated in data_setup function

    Function calculates the Euclidean distance of each item in the main_data file from all the training data
    and finds the class of the specific item from each training data. The two values are stored in a tuple.
    We end up with 188 different distances for each main data/item.
    So, we have a list of 188 different tuples for each main data/item.
    '''
    dv = []
    for element in train_data:
        sqr_sum = 0.0
        for i in range(len(item)):
            sqr_sum += (item[i] - element[i])**2
        dist = np.sqrt(sqr_sum)
        dv.append((dist, element[7]))
    dv.sort()

    return dv


def decide_class_wk(dv, k):
    '''
    Returns the class of the item with distance dv
    to all items in the training dataset,
    by weighted voting amongst its k nearest neighbours.
    '''
    # Sort list dv by ascending distance, extract k smallest elements and their class
    smallest_dist = sorted(dv)[:k]

    # Get list of (class ID, weight) tuples
    weights = [(d[1], 1 / d[0]) for d in smallest_dist]

    # Calculate score for each candidate class.
    # We use a dictionary {class ID: score}.
    scores = {}
    for c, w in weights:
        if c not in scores.keys():
            # New candidate -- add it to the list.
            scores[c] = w
        else:
            # Existing candidate -- add w to its score.
            scores[c] += w

    # Sort the dictionary by descending scores (returns a list of tuples)
    vote = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Return highest scoring candidate
    return vote[0][0]


# -------- MAIN PROGRAM ---------
train_data, main_data = data_setup()
knn = [3, 8, 25]
for k in knn:
    new_classes = []
    for ele in range(len(main_data)):
        dv = dist_vect(main_data[ele], train_data)
        new_classes.append(decide_class_wk(dv, k))
    print('Weighted classes for k={}: {}\n'.format(k, new_classes))
