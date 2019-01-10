from K_Nearest_Neighbours import data_setup, decide_class_wk


def dist_vectL1(item, train_data):
    '''
    Arguments: item = a datapoint of the main data
               train_data =  a list of lists as generated in data_setup function

    Function calculates the norm-1 of each item in the main_data file from all the training data
    and finds the class of the specific item from each training data. The two values are stored in a tuple.
    We end up with 188 different distances for each main data/item.
    So, we have a list of 188 different tuples for each main data/item.
    '''
    class_val = []
    dist = []
    abs_sum = 0
    for element in train_data:
        for i in range(len(item)):
            abs_sum += abs(item[i] - element[i])
        dist.append(abs_sum)
        class_val.append(element[7])
        dvL1 = list(zip(dist, class_val))
        abs_sum = 0
    dvL1.sort()

    return dvL1


# -------- MAIN PROGRAM --------

train_data, main_data = data_setup()
knn = [3, 8, 25]
for k in knn:
    class_values = []
    for ele in range(len(main_data)):
        dv = dist_vectL1(main_data[ele], train_data)
        class_values.append(decide_class_wk(dv, k))

    print("Classes of the main data:", class_values)
